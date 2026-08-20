"""
This file contains functions for calibrating the rank of FFA-LoRA layers in a federated learning setting, particularly when differential privacy (DP) is applied.
The goal is to select a rank that maximizes model accuracy while accounting for the noise introduced by DP.
The calibration is based on empirical results from prior experiments, which provide a mapping from rank to expected accuracy without DP, and a method to estimate the noise multiplier for a given DP budget.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from federated_lora.config import Config
from federated_lora.rank_rule import noise_multiplier, shard_size
from rankrule.validate_rank_rule import _final_accuracy, load_grid

# Loads the final accuracies from the no-DP rank sweep (legacy, num_clients=20, GLUE-era).
def load_rank_sweep(results_dir: Path, tail: int) -> dict[int, list[float]]:
    """{rank: [final_accuracy per seed]} from the no-DP rank sweep."""
    by_rank: dict[int, list[float]] = defaultdict(list)
    for path in sorted(results_dir.glob("rank_sweep_r*_epsnone*_seed*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        rank = result["config"]["lora_rank"]
        by_rank[rank].append(_final_accuracy(result, tail))
    return by_rank


def _nc4_final_accuracy(result: dict, tail: int) -> float:
    """Same idea as _final_accuracy, but cifar100 result files use top1_acc (0..100), not accuracy (0..1)."""
    history = result['history']
    window = history[-tail:] if len(history) >= tail else history
    return sum(entry['top1_acc'] for entry in window) / len(window) / 100.0


def load_nc4_points(
    results_dir: Path, methods: list[str], tail: int, config: Config, size: float
) -> list[tuple[int, float, float]]:
    """
    (rank, sigma, accuracy) triples pooled across `methods` from the current
    num_clients=4 sweeps: dp_rank/rank<r>/ (fixed eps=RANK_SWEEP_EPS),
    dp_rank_grid/eps<e>_rank<r>/ (crossed eps x rank), and dp_rank_nodp/rank<r>/
    (no-DP anchor, once it exists). Pooling multiple GRID_METHODS at the same
    (rank, sigma) is deliberate, not just extra data: at homogeneous rank there's
    no heterogeneity for their aggregation to reconcile, so they should agree,
    but in practice they don't exactly (e.g. rblora vs flexlora can differ by a
    few points) -- pooling avoids baking one method's idiosyncrasies into the
    rank effect, and single-seed cells badly need the extra samples anyway.
    """
    points: list[tuple[int, float, float]] = []
    for method in methods:
        for pattern in (
            f'dp_rank/rank*/{method}_*_seed*.json',
            f'dp_rank_grid/eps*_rank*/{method}_*_seed*.json',
            f'dp_rank_nodp/rank*/{method}_*_seed*.json',
        ):
            for path in sorted(results_dir.glob(pattern)):
                result = json.loads(path.read_text(encoding='utf-8'))
                run_config = result['config']
                rank = run_config['lora_rank']
                eps = run_config['target_epsilon']
                sigma = noise_multiplier(config, eps, size)
                points.append((rank, sigma, _nc4_final_accuracy(result, tail)))
    return points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tail", type=int, default=5)
    parser.add_argument("--results-dir", default="results/sweep")
    parser.add_argument("--min-seeds", type=int, default=3,
                         help="minimum seeds for a (rank, epsilon) cell to be included in the fit (--source legacy only)")
    parser.add_argument("--source", choices=["legacy", "nc4"], default="legacy",
                         help="legacy: num_clients=20 GLUE-era rank_sweep/grid_sweep files. "
                              "nc4: current num_clients=4 dp_rank/dp_rank_grid/dp_rank_nodp files.")
    parser.add_argument("--methods", nargs="+", default=["rblora", "hetlora", "flora", "flexlora"],
                         help="--source nc4 only: which methods' runs to pool as the calibration signal "
                              "(GRID_METHODS by default -- they should all agree at homogeneous rank/eps)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    num_clients = 4 if args.source == "nc4" else 20
    config = Config(task='cifar100', method=args.methods[0] if args.source == "nc4" else "rblora", num_clients=num_clients)
    size = shard_size(config)

    # (rank, sigma, accuracy) triples from every source of evidence.
    points: list[tuple[int, float, float]]

    if args.source == "nc4":
        points = load_nc4_points(results_dir, args.methods, args.tail, config, size)
    else:
        points = []
        for rank, accs in load_rank_sweep(results_dir, args.tail).items():
            if len(accs) >= args.min_seeds:
                points.append((rank, 0.0, sum(accs) / len(accs)))

        for eps, by_rank in load_grid(results_dir, args.tail).items():
            if eps is None:
                continue  # the grid sweep doesn't include a no-DP point; the rank sweep covers that
            sigma = noise_multiplier(config, eps, size)
            for rank, accs in by_rank.items():
                if len(accs) >= args.min_seeds:
                    points.append((rank, sigma, sum(accs) / len(accs)))

    ranks = sorted({rank for rank, _sigma, _acc in points})
    if len(points) < len(ranks) + 1:
        print(f"only {len(points)} usable (rank, epsilon) cells for {len(ranks)} ranks -- "
              f"need at least {len(ranks) + 1} to fit one acc_nodp per rank plus a shared kappa. "
              f"Lower --min-seeds or wait for more of the grid sweep to land.")
        return

    # Fit a linear model to the (rank, sigma, accuracy) points. The model is:
    #   accuracy(rank, sigma) = acc_nodp(rank) - kappa * rank * sigma^2
    # where acc_nodp(rank) is the expected accuracy without DP for that rank, and kappa is a shared coefficient that captures the accuracy loss due to DP noise.
    X = np.zeros((len(points), len(ranks) + 1))
    y = np.zeros(len(points))
    for i, (rank, sigma, acc) in enumerate(points):
        X[i, ranks.index(rank)] = 1.0
        X[i, -1] = -(rank * sigma ** 2)
        y[i] = acc

    beta, _residuals, _rank_of_X, _singular_values = np.linalg.lstsq(X, y, rcond=None)
    acc_nodp = dict(zip(ranks, beta[:-1]))
    kappa = float(beta[-1])

    predicted = X @ beta
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    print(f"fit over {len(points)} (rank, epsilon) cells across {len(ranks)} ranks: R^2={r_squared:.3f}\n")
    print("ACC_NODP_BY_RANK = {")
    for rank in ranks:
        print(f"    {rank}: {acc_nodp[rank]:.4f},")
    print("}")
    print(f"kappa = {kappa:.6f}")

    print("\nper-point residuals (measured - predicted; large |residual| means the linear model fits that cell poorly):")
    for (rank, sigma, acc), pred in zip(points, predicted):
        print(f"  rank={rank:>3} sigma={sigma:.4f} measured={acc:.4f} predicted={pred:.4f} residual={acc - pred:+.4f}")


if __name__ == "__main__":
    main()
