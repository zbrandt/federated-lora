"""
Check ffalora.rank_rule.recommended_rank's predictions against the actual
accuracy-maximizing rank observed in the grid sweep
(results/sweep/grid_sweep_*.json) -- the ground truth the rule was only
ever provisionally calibrated against a single anchor point plus one
corroborating cell, per rank_rule.py's docstring.

Meaningful once at least two ranks have `--min-seeds` completed runs at
the same epsilon; safe to run against a partial grid sweep before it
finishes -- epsilons without enough coverage yet are skipped and reported
as such, not silently treated as a match.

Usage:
    uv run python validate_rank_rule.py
    uv run python validate_rank_rule.py --min-seeds 3   # peek early, noisier
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from ffalora.config import Config
from ffalora.rank_rule import GLUE_TRAIN_SIZES, recommended_rank


def _final_accuracy(result: dict, tail: int) -> float:
    history = result["history"]
    window = history[-tail:] if len(history) >= tail else history
    return sum(entry["accuracy"] for entry in window) / len(window)


def load_grid(results_dir: Path, tail: int) -> dict[float, dict[int, list[float]]]:
    """{epsilon: {rank: [final_accuracy per seed]}} read straight from the grid sweep's own output files."""
    by_eps_rank: dict[float, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for path in sorted(results_dir.glob("grid_sweep_r*_eps*_seed*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        config = result["config"]
        rank = config["lora_rank"]
        client_epsilons = config["client_epsilons"]
        eps = client_epsilons[0] if client_epsilons else None
        by_eps_rank[eps][rank].append(_final_accuracy(result, tail))
    return by_eps_rank


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tail", type=int, default=5, help="rounds averaged into each run's final accuracy")
    parser.add_argument("--results-dir", default="results/sweep")
    parser.add_argument("--min-seeds", type=int, default=5,
                         help="a rank only counts as 'measured' at a given epsilon once it has this many seeds")
    args = parser.parse_args()

    by_eps_rank = load_grid(Path(args.results_dir), args.tail)
    config = Config()
    shard_size = GLUE_TRAIN_SIZES[config.dataset_task] / config.num_clients

    print(f"{'epsilon':>8} | {'measured best':>13} | {'rule pick':>9} | ranks measured -> mean accuracy")
    hits = 0
    scored = 0
    for eps in sorted(by_eps_rank, key=lambda value: (value is None, value)):
        rank_means = {
            rank: sum(accs) / len(accs)
            for rank, accs in by_eps_rank[eps].items()
            if len(accs) >= args.min_seeds
        }
        if len(rank_means) < 2:
            print(f"{eps!s:>8} | (skipped -- fewer than 2 ranks have {args.min_seeds}+ seeds yet)")
            continue

        measured_best = max(rank_means, key=rank_means.get)
        candidate_ranks = tuple(sorted(rank_means))
        try:
            predicted = recommended_rank(config, eps, shard_size, candidate_ranks=candidate_ranks)
        except ValueError as error:
            print(f"{eps!s:>8} | measured best={measured_best} | rule can't predict: {error}")
            continue

        scored += 1
        hit = predicted == measured_best
        hits += hit
        pretty = {rank: round(acc, 3) for rank, acc in sorted(rank_means.items())}
        print(f"{eps!s:>8} | {measured_best:>13} | {predicted:>9} | {pretty}  [{'OK' if hit else 'MISS'}]")

    if scored:
        print(f"\n{hits}/{scored} epsilon points matched the rule's prediction ({100 * hits / scored:.0f}%)")
    else:
        print("\nnothing scoreable yet -- need at least 2 ranks with --min-seeds seeds each, at the same epsilon")


if __name__ == "__main__":
    main()
