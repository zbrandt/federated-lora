"""
This file contains functions for calibrating the rank of FFA-LoRA layers in a federated learning setting, particularly when differential privacy (DP) is applied.
The goal is to select a rank that maximizes model accuracy while accounting for the noise introduced by DP.
The calibration is based on empirical results from prior experiments, which provide a mapping from rank to expected accuracy without DP, and a method to estimate the noise multiplier for a given DP budget.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ffalora.config import Config
from ffalora.rank_rule import GLUE_TRAIN_SIZES, noise_multiplier
from validate_rank_rule import _final_accuracy, load_grid

# Loads the final accuracies from the no-DP rank sweep.
def load_rank_sweep(results_dir: Path, tail: int) -> dict[int, list[float]]:
    """{rank: [final_accuracy per seed]} from the no-DP rank sweep."""
    by_rank: dict[int, list[float]] = defaultdict(list)
    for path in sorted(results_dir.glob("rank_sweep_r*_epsnone*_seed*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        rank = result["config"]["lora_rank"]
        by_rank[rank].append(_final_accuracy(result, tail))
    return by_rank

# Loads the final accuracies from the DP grid sweep.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tail", type=int, default=5)
    parser.add_argument("--results-dir", default="results/sweep")
    parser.add_argument("--min-seeds", type=int, default=3,
                         help="minimum seeds for a (rank, epsilon) cell to be included in the fit")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    config = Config()
    shard_size = GLUE_TRAIN_SIZES[config.dataset_task] / config.num_clients

    # (rank, sigma, accuracy) triples from every source of evidence.
    points: list[tuple[int, float, float]] = []

    for rank, accs in load_rank_sweep(results_dir, args.tail).items():
        if len(accs) >= args.min_seeds:
            points.append((rank, 0.0, sum(accs) / len(accs)))

    for eps, by_rank in load_grid(results_dir, args.tail).items():
        if eps is None:
            continue  # the grid sweep doesn't include a no-DP point; the rank sweep covers that
        sigma = noise_multiplier(config, eps, shard_size)
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
