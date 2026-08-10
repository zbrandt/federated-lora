"""
Two controlled ablations over FFA-LoRA (see ffalora/lora_layer.py), each
varying exactly one axis at a time so any accuracy trend can be attributed
to that axis alone:

1. `rank` sweep -- DP held constant (or off) across every run, `lora_rank`
   (r_max, homogeneous across clients) varied between runs.
2. `dp`   sweep -- rank held constant across every run, the shared per-client
   DP epsilon varied between runs (including a `none` no-DP baseline point).
3. `grid` sweep -- full rank x epsilon interaction grid (every combination,
   not one axis at a time). Tests whether the accuracy-maximizing rank
   SHIFTS as epsilon changes, which the two marginals above can't show:
   each of those holds the other axis fixed at a single value, so a rank
   effect that only appears at small epsilon (as DP noise-amplification
   theory predicts -- optimal rank should move to SMALLER ranks as epsilon
   shrinks, since noise added to B scales into the reconstructed update
   alongside signal that a wider rank can't outrun) wouldn't show up there.

Every run is a full ffalora.build(config) + server.run(), so this is exactly
as expensive as running main.py that many times -- there is no shortcut.
Run with --smoke first to confirm plumbing before committing to a real
(likely multi-hour, and for `grid`, likely multi-day) sweep.

Usage
-----
    # fixed epsilon=4.0, rank in {2,4,6,8}, 2 seeds each
    uv run python sweep.py rank --dp-epsilon 4.0 --ranks 2 4 6 8 --seeds 42 43

    # fixed rank=8, epsilon in {none, 1, 2, 4, 8}, 2 seeds each
    uv run python sweep.py dp --rank 8 --epsilons none 1 2 4 8 --seeds 42 43

    # both, with default sweep points
    uv run python sweep.py both --seeds 42

    # full rank x epsilon interaction grid, large logical batch split into
    # GPU-memory-safe physical sub-batches under DP
    uv run python sweep.py grid \
        --ranks 2 4 8 16 32 64 --epsilons 1 2 4 8 --seeds 0 1 2 3 4 \
        --batch-size 512 --max-physical-batch-size 16 --epochs 3

    # fast plumbing check before a real sweep
    uv run python sweep.py rank --smoke

Each run's raw per-round JSON is cached to `results/sweep/<tag>.json` (same
schema as `main.py run`, so `python main.py plot results/sweep` also works
if you want per-round curves instead of the summary trend below) and
skipped on a re-run unless `--force` is passed. A trend plot -- accuracy
(mean of the last `--tail` rounds) vs. the swept variable, mean +/- std
across seeds -- is written to `figures/sweep/`.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ffalora import build
from ffalora.config import GLUE_TASKS, Config
from ffalora.rank_rule import GLUE_TRAIN_SIZES, recommended_rank


def _parse_epsilon(token: str) -> float | None:
    return None if token.strip().lower() == "none" else float(token)


def _fmt(value: float) -> str:
    return f"{value:g}"


def _base_config(args: argparse.Namespace, seed: int) -> Config:
    config = Config(seed=seed, dataset_task=args.task)
    if args.smoke:
        config = replace(config, rounds=2, num_clients=4, train_split="train[:1%]")
    if args.rounds is not None:
        config = replace(config, rounds=args.rounds)
    if args.num_clients is not None:
        config = replace(config, num_clients=args.num_clients)
    return config


def _run_one(config: Config, tag: str, results_dir: Path, force: bool) -> dict:
    path = results_dir / f"{tag}.json"
    if path.exists() and not force:
        print(f"[sweep] {tag}: cached at {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    print(f"[sweep] {tag}: running ({config.rounds} rounds, {config.num_clients} clients)...", flush=True)
    server = build(config)
    history = server.run()
    result = {"config": asdict(config), "history": [asdict(metric) for metric in history]}

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[sweep] {tag}: wrote {path}")
    return result


def _final_accuracy(result: dict, tail: int) -> float:
    history = result["history"]
    window = history[-tail:] if len(history) >= tail else history
    return sum(entry["accuracy"] for entry in window) / len(window)


def run_rank_sweep(args: argparse.Namespace) -> list[tuple[int, int, dict]]:
    """Fixed DP (or no DP) held constant; r_max varied homogeneously between runs."""
    runs: list[tuple[int, int, dict]] = []
    dp_epsilon = args.dp_epsilon
    eps_tag = "none" if dp_epsilon is None else _fmt(dp_epsilon)

    for rank in args.ranks:
        for seed in args.seeds:
            base = _base_config(args, seed)
            epsilons = None if dp_epsilon is None else [dp_epsilon] * base.num_clients
            config = replace(
                base,
                lora_rank=rank,
                client_ranks=None,
                client_epsilons=epsilons,
                method=f"ffalora-rank{rank}-eps{eps_tag}",
            )
            tag = f"rank_sweep_r{rank}_eps{eps_tag}_rounds{config.rounds}_nc{config.num_clients}_seed{seed}"
            result = _run_one(config, tag, Path(args.results_dir), args.force)
            runs.append((rank, seed, result))

    return runs


def run_dp_sweep(args: argparse.Namespace) -> list[tuple[float | None, int, dict]]:
    """Rank held constant (or, with --auto-rank, chosen per epsilon by ffalora.rank_rule); the
    shared per-client DP epsilon varied between runs."""
    runs: list[tuple[float | None, int, dict]] = []

    for eps in args.epsilons:
        eps_tag = "none" if eps is None else _fmt(eps)
        for seed in args.seeds:
            base = _base_config(args, seed)

            auto_rank = getattr(args, "auto_rank", False)
            if auto_rank:
                candidate_ranks = tuple(getattr(args, "candidate_ranks", [2, 4, 6, 8]))
                shard_size = GLUE_TRAIN_SIZES[base.dataset_task] / base.num_clients
                rank = recommended_rank(base, eps, shard_size, candidate_ranks=candidate_ranks)
                print(f"[sweep] auto-rank: eps={eps_tag} -> rank={rank}")
            else:
                rank = args.rank

            epsilons = None if eps is None else [eps] * base.num_clients
            config = replace(
                base,
                lora_rank=rank,
                client_ranks=None,
                client_epsilons=epsilons,
                method=f"ffalora-rank{rank}-eps{eps_tag}",
            )
            rank_tag = f"auto{rank}" if auto_rank else str(rank)
            tag = f"dp_sweep_rank{rank_tag}_eps{eps_tag}_rounds{config.rounds}_nc{config.num_clients}_seed{seed}"
            result = _run_one(config, tag, Path(args.results_dir), args.force)
            runs.append((eps, seed, result))

    return runs


def run_grid_sweep(args: argparse.Namespace) -> list[tuple[int, float | None, int, dict]]:
    """
    Full rank x epsilon grid -- every combination, not one axis at a time.
    Unlike `rank`/`dp` above, this can reveal an INTERACTION: whether the
    accuracy-maximizing rank shifts as epsilon changes, rather than just the
    two independent main effects.
    """
    runs: list[tuple[int, float | None, int, dict]] = []

    overrides: dict = {}
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.epochs is not None:
        overrides["local_epochs"] = args.epochs
    if args.max_physical_batch_size is not None:
        overrides["max_physical_batch_size"] = args.max_physical_batch_size

    for rank in args.ranks:
        for eps in args.epsilons:
            eps_tag = "none" if eps is None else _fmt(eps)
            for seed in args.seeds:
                base = _base_config(args, seed)
                if overrides:
                    base = replace(base, **overrides)

                epsilons = None if eps is None else [eps] * base.num_clients
                config = replace(
                    base,
                    lora_rank=rank,
                    client_ranks=None,
                    client_epsilons=epsilons,
                    method=f"ffalora-rank{rank}-eps{eps_tag}",
                )
                tag = (
                    f"grid_sweep_r{rank}_eps{eps_tag}"
                    f"_rounds{config.rounds}_nc{config.num_clients}"
                    f"_bs{config.batch_size}_ep{config.local_epochs}"
                    f"_seed{seed}"
                )
                result = _run_one(config, tag, Path(args.results_dir), args.force)
                runs.append((rank, eps, seed, result))

    return runs


def _aggregate_by_x(runs: list[tuple], tail: int) -> tuple[list, list[float], list[float]]:
    """Group (x, seed, result) triples by x; mean/std final accuracy across seeds."""
    by_x: dict = defaultdict(list)
    for x, _seed, result in runs:
        by_x[x].append(_final_accuracy(result, tail))

    xs = sorted(by_x, key=lambda v: (v is None, v))
    means = [float(np.mean(by_x[x])) for x in xs]
    stds = [float(np.std(by_x[x])) for x in xs]
    return xs, means, stds


def plot_rank_trend(runs: list[tuple[int, int, dict]], tail: int, output_path: Path) -> None:
    xs, means, stds = _aggregate_by_x(runs, tail)

    plt.figure(figsize=(7, 5))
    plt.errorbar(xs, means, yerr=stds, marker="o", capsize=4, linewidth=2, color="C0")
    plt.xlabel("LoRA rank (r_max)")
    plt.ylabel(f"Accuracy (mean of last {tail} rounds)")
    plt.title("Accuracy vs. rank (DP held constant)")
    plt.grid(True, alpha=0.3)
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[sweep] wrote {output_path}")


def plot_dp_trend(runs: list[tuple[float | None, int, dict]], tail: int, output_path: Path) -> None:
    xs, means, stds = _aggregate_by_x(runs, tail)

    # Split the no-DP baseline (x is None) out from the real epsilon values
    # so it can be drawn as a flat reference line instead of distorting a
    # log-scale x-axis.
    baseline = None
    plot_xs, plot_means, plot_stds = [], [], []
    for x, mean, std in zip(xs, means, stds):
        if x is None:
            baseline = mean
        else:
            plot_xs.append(x)
            plot_means.append(mean)
            plot_stds.append(std)

    plt.figure(figsize=(7, 5))
    if plot_xs:
        plt.errorbar(plot_xs, plot_means, yerr=plot_stds, marker="o", capsize=4,
                     linewidth=2, color="C0", label="DP-SGD")
        plt.xscale("log")
    if baseline is not None:
        plt.axhline(baseline, linestyle="--", color="gray", label=f"no-DP baseline ({baseline:.3f})")
    plt.xlabel("Target epsilon (log scale)")
    plt.ylabel(f"Accuracy (mean of last {tail} rounds)")
    plt.title("Accuracy vs. privacy budget (rank held constant)")
    plt.grid(True, alpha=0.3)
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[sweep] wrote {output_path}")


def plot_grid_trend(runs: list[tuple[int, float | None, int, dict]], tail: int, output_path: Path) -> None:
    """One accuracy-vs-rank line per epsilon value -- if optimal rank shifts
    with epsilon (the interaction this sweep exists to test), each line's
    peak sits at a different x position rather than all lines peaking at
    the same rank."""
    by_eps: dict = defaultdict(lambda: defaultdict(list))
    for rank, eps, _seed, result in runs:
        by_eps[eps][rank].append(_final_accuracy(result, tail))

    eps_values = sorted(by_eps, key=lambda v: (v is None, v))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    plt.figure(figsize=(8, 5))
    print("\n[grid] accuracy-maximizing rank per epsilon:")
    for i, eps in enumerate(eps_values):
        ranks = sorted(by_eps[eps])
        means = [float(np.mean(by_eps[eps][r])) for r in ranks]
        stds = [float(np.std(by_eps[eps][r])) for r in ranks]
        label = "no-DP" if eps is None else f"eps={_fmt(eps)}"

        plt.errorbar(ranks, means, yerr=stds, marker="o", capsize=3, linewidth=2,
                     label=label, color=colors[i % len(colors)])

        best_rank = ranks[int(np.argmax(means))]
        print(f"  {label:>10}: best rank = {best_rank:>3}  (accuracy={max(means):.4f})")

    plt.xlabel("LoRA rank (r_max)")
    plt.ylabel(f"Accuracy (mean of last {tail} rounds)")
    plt.title("Accuracy vs. rank, by privacy budget")
    plt.xscale("log", base=2)
    plt.grid(True, alpha=0.3)
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"\n[sweep] wrote {output_path}")


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--task", choices=list(GLUE_TASKS), default="sst2")
    parser.add_argument("--rounds", type=int, default=None, help="override Config.rounds for every run")
    parser.add_argument("--num-clients", type=int, default=None, help="override Config.num_clients for every run")
    parser.add_argument("--smoke", action="store_true",
                         help="shrink to rounds=2, num_clients=4, train_split='train[:1%%]' for a fast plumbing check")
    parser.add_argument("--results-dir", default="results/sweep")
    parser.add_argument("--figures-dir", default="figures/sweep")
    parser.add_argument("--tail", type=int, default=5, help="rounds averaged into each 'final accuracy' point")
    parser.add_argument("--force", action="store_true", help="re-run even if a cached result file already exists")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="sweep", required=True)

    rank_p = sub.add_parser("rank", help="fixed DP, vary rank")
    _add_common_args(rank_p)
    rank_p.add_argument("--ranks", type=int, nargs="+", default=[2, 4, 6, 8])
    rank_p.add_argument("--dp-epsilon", type=_parse_epsilon, default=None,
                         help="epsilon applied to every client at every rank point ('none' disables DP, the default)")

    dp_p = sub.add_parser("dp", help="fixed rank, vary DP epsilon")
    _add_common_args(dp_p)
    dp_p.add_argument("--rank", type=int, default=8,
                       help="ignored when --auto-rank is set")
    dp_p.add_argument("--epsilons", type=_parse_epsilon, nargs="+", default=[None, 1.0, 2.0, 4.0, 8.0])
    dp_p.add_argument("--auto-rank", action="store_true",
                       help="pick lora_rank per epsilon via ffalora.rank_rule.recommended_rank instead of "
                            "using a fixed --rank for every point")
    dp_p.add_argument("--candidate-ranks", type=int, nargs="+", default=[2, 4, 6, 8],
                       help="ranks recommended_rank chooses among when --auto-rank is set")

    both_p = sub.add_parser("both", help="run both sweeps with default sweep points")
    _add_common_args(both_p)
    both_p.add_argument("--ranks", type=int, nargs="+", default=[2, 4, 6, 8])
    both_p.add_argument("--dp-epsilon", type=_parse_epsilon, default=None)
    both_p.add_argument("--rank", type=int, default=8)
    both_p.add_argument("--epsilons", type=_parse_epsilon, nargs="+", default=[None, 1.0, 2.0, 4.0, 8.0])

    grid_p = sub.add_parser("grid", help="full rank x epsilon interaction grid")
    _add_common_args(grid_p)
    grid_p.add_argument("--ranks", type=int, nargs="+", default=[2, 4, 8, 16, 32, 64])
    grid_p.add_argument("--epsilons", type=_parse_epsilon, nargs="+", default=[1.0, 2.0, 4.0, 8.0])
    grid_p.add_argument("--batch-size", type=int, default=None,
                         help="override Config.batch_size for every run (larger batches improve DP privacy "
                              "amplification but need --max-physical-batch-size to fit in GPU memory)")
    grid_p.add_argument("--max-physical-batch-size", type=int, default=None,
                         help="Opacus BatchMemoryManager cap: split each DP logical batch into physical "
                              "sub-batches of at most this size; only takes effect under DP with --epochs set")
    grid_p.add_argument("--epochs", type=int, default=None,
                         help="full passes over each selected client's local shard per round "
                              "(sets Config.local_epochs, replacing the fixed local_steps count)")

    args = parser.parse_args()
    figures_dir = Path(args.figures_dir)

    if args.sweep in ("rank", "both"):
        runs = run_rank_sweep(args)
        plot_rank_trend(runs, args.tail, figures_dir / "rank_trend.png")
    if args.sweep in ("dp", "both"):
        runs = run_dp_sweep(args)
        plot_dp_trend(runs, args.tail, figures_dir / "dp_trend.png")
    if args.sweep == "grid":
        runs = run_grid_sweep(args)
        plot_grid_trend(runs, args.tail, figures_dir / "grid_trend.png")


if __name__ == "__main__":
    main()
