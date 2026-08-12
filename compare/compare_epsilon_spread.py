"""
Phase 1: does personalizing each client's LoRA rank to its own DP epsilon (via
ffalora.rank_rule.recommended_rank) beat picking one global rank from the
population's mean epsilon, once clients stop sharing a single privacy budget?

For each spread level, half the clients get mean_eps - delta and half get
mean_eps + delta, so the population mean epsilon is exactly mean_eps at every
spread level -- only how far individual clients deviate from it changes.
"personalized" gives each client the rank recommended_rank() would pick for its
own epsilon; "uniform" gives every client the rank recommended_rank() would
pick knowing only the population mean. Both run under rank-aware aggregation
(ffalora/server/aggregate.py) so a real personalization effect isn't confounded
with the naive-aggregation dilution bug idea #1 fixed.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from ffalora.config import Config
from ffalora.rank_rule import GLUE_TRAIN_SIZES, recommended_rank
from sweep import _run_one


# Half the clients get mean_eps - delta, half get mean_eps + delta; population mean stays exactly mean_eps.
# spread=0 means delta=0 (every client identical); spread=1 means delta reaches all the way down to eps_min.
def make_epsilon_spread(num_clients: int, mean_eps: float, spread: float, seed: int, eps_min: float = 0.5) -> list[float]:
    if spread <= 0:
        return [mean_eps] * num_clients

    max_delta = min(spread * mean_eps, mean_eps - eps_min)
    rng = np.random.default_rng(seed)
    half = num_clients // 2
    deltas = rng.uniform(0.0, max_delta, size=half)
    epsilons = [mean_eps + d for d in deltas] + [mean_eps - d for d in deltas]
    if num_clients % 2 == 1:
        epsilons.append(mean_eps)
    rng.shuffle(epsilons)
    return epsilons


# Each client's rank as recommended_rank() would pick it for that client's own epsilon.
def personalized_ranks(config: Config, client_epsilons: list[float], shard_size: float, candidate_ranks: tuple[int, ...]) -> list[int]:
    return [recommended_rank(config, eps, shard_size, candidate_ranks) for eps in client_epsilons]


# The single global rank recommended_rank() would pick knowing only the population's mean epsilon.
def uniform_ranks(config: Config, mean_eps: float, num_clients: int, shard_size: float, candidate_ranks: tuple[int, ...]) -> list[int]:
    rank = recommended_rank(config, mean_eps, shard_size, candidate_ranks)
    return [rank] * num_clients


def _final_accuracy(result: dict, tail: int) -> float:
    history = result["history"]
    window = history[-tail:] if len(history) >= tail else history
    return sum(entry["accuracy"] for entry in window) / len(window)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-clients", type=int, default=20)
    parser.add_argument("--client-sample-rate", type=float, default=0.2)
    parser.add_argument("--mean-eps", type=float, default=2.0)
    parser.add_argument("--eps-min", type=float, default=0.5)
    parser.add_argument("--spreads", type=float, nargs="+", default=[0.0, 0.33, 0.66, 1.0])
    parser.add_argument("--candidate-ranks", type=int, nargs="+", default=[2, 4, 6, 8])
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--tail", type=int, default=5)
    parser.add_argument("--task", default="sst2")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="tiny config for a fast correctness check")
    parser.add_argument("--results-dir", default="results/epsilon_spread")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    candidate_ranks = tuple(args.candidate_ranks)
    num_clients = 4 if args.smoke else args.num_clients
    shard_size = GLUE_TRAIN_SIZES[args.task] / num_clients

    accuracies: dict[tuple[float, str], list[float]] = defaultdict(list)

    for spread in args.spreads:
        for mode in ("uniform", "personalized"):
            for seed in args.seeds:
                base = Config(
                    seed=seed,
                    dataset_task=args.task,
                    num_clients=num_clients,
                    client_sample_rate=args.client_sample_rate,
                    aggregation_strategy="rank_aware",
                    rounds=2 if args.smoke else args.rounds,
                    local_epochs=1 if args.smoke else None,
                    train_split="train[:1%]" if args.smoke else "train",
                )

                client_epsilons = make_epsilon_spread(num_clients, args.mean_eps, spread, seed, args.eps_min)
                if mode == "personalized":
                    ranks = personalized_ranks(base, client_epsilons, shard_size, candidate_ranks)
                else:
                    ranks = uniform_ranks(base, args.mean_eps, num_clients, shard_size, candidate_ranks)

                config = replace(base, client_epsilons=client_epsilons, client_ranks=ranks, lora_rank=max(ranks))

                tag = f"epsspread_{mode}_spread{spread:g}_meaneps{args.mean_eps:g}_seed{seed}"
                result = _run_one(config, tag, results_dir, args.force)
                accuracies[(spread, mode)].append(_final_accuracy(result, args.tail))

    print(f"mean_eps={args.mean_eps:g}, candidate_ranks={candidate_ranks}\n")
    print(f"{'spread':>7} | {'mode':>12} | {'n seeds':>7} | {'mean acc':>8} | {'std':>6}")
    for spread in args.spreads:
        for mode in ("uniform", "personalized"):
            accs = accuracies[(spread, mode)]
            mean = sum(accs) / len(accs)
            std = (sum((a - mean) ** 2 for a in accs) / len(accs)) ** 0.5
            print(f"{spread:>7g} | {mode:>12} | {len(accs):>7} | {mean:>8.4f} | {std:>6.4f}")


if __name__ == "__main__":
    main()
