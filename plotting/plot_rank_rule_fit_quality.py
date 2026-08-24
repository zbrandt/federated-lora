#!/usr/bin/env python
"""
plot_rank_rule_fit_quality.py -- why rank_rule's model can't be salvaged by
refitting: a per-method parity plot (predicted vs measured accuracy) using
the SAME model form (accuracy = acc_nodp[rank] - kappa*rank*sigma^2), fit
separately per method on num_clients=4 data instead of pooled.

Points on the diagonal = the model's (rank, sigma)-only form explains that
method's accuracy well. Points scattered off the diagonal = it doesn't --
because that method's accuracy also depends on something the model was never
given, like aggregation error under rank heterogeneity (rblora/hetlora) or an
outright divergence cliff at low rank + high noise (flexlora). flora is the
one method whose real aggregation is exactly invariant to rank spread, so
it's the one case where (rank, sigma) really is the whole story.

Usage:
    python plotting/plot_rank_rule_fit_quality.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from federated_lora.config import Config
from federated_lora.rank_rule import shard_size
from rankrule.fit_rank_rule import load_nc4_points

OUTPUT_PATH = Path("figures/sweep/rank_rule_fit_quality.png")

INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
METHOD_COLORS = {
    "rblora": "#2a78d6",
    "hetlora": "#eb6834",
    "flora": "#1baf7a",
    "flexlora": "#eda100",
}
METHODS = ["rblora", "hetlora", "flora", "flexlora"]


def fit_per_method(points: list[tuple[int, float, float]]):
    """Same lstsq fit as rankrule/fit_rank_rule.py's pooled fit, applied to one method's points."""
    ranks = sorted({rank for rank, _sigma, _acc in points})
    if len(points) < len(ranks) + 1:
        return None

    X = np.zeros((len(points), len(ranks) + 1))
    y = np.zeros(len(points))
    for i, (rank, sigma, acc) in enumerate(points):
        X[i, ranks.index(rank)] = 1.0
        X[i, -1] = -(rank * sigma ** 2)
        y[i] = acc

    beta, _residuals, _rank_of_X, _sv = np.linalg.lstsq(X, y, rcond=None)
    predicted = X @ beta
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return predicted, y, r_squared


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tail", type=int, default=5)
    parser.add_argument("--results-dir", default="results/sweep")
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    config = Config(task="cifar100", method="rblora", num_clients=4)
    size = shard_size(config)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--", zorder=1, label="perfect fit")

    for method in METHODS:
        points = load_nc4_points(results_dir, [method], args.tail, config, size)
        fit = fit_per_method(points)
        if fit is None:
            print(f"skipping {method}: not enough points yet")
            continue
        predicted, measured, r2 = fit
        color = METHOD_COLORS[method]
        ax.scatter(predicted, measured, s=70, color=color, edgecolors=INK, linewidths=0.6,
                   zorder=3, label=f"{method}  (R²={r2:.2f}, n={len(points)})")

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("model-predicted accuracy", fontsize=10, color=SECONDARY_INK)
    ax.set_ylabel("measured accuracy", fontsize=10, color=SECONDARY_INK)
    ax.set_title("Why rank_rule can't be refit: per-method fit quality\n(num_clients=4, same model form fit separately per method)",
                 fontsize=12, color=INK, pad=10)
    ax.set_aspect("equal")
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.legend(frameon=False, labelcolor=SECONDARY_INK, fontsize=9, loc="upper left")

    fig.tight_layout()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
