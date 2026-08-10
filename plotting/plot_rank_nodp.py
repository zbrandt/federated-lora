from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fit_rank_rule import load_rank_sweep

OUTPUT_PATH = Path("figures/sweep/rank_nodp_trend.png")

INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
COLOR = "#2a78d6"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tail", type=int, default=5)
    parser.add_argument("--results-dir", default="results/sweep")
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args()

    by_rank = load_rank_sweep(Path(args.results_dir), args.tail)
    ranks = sorted(by_rank)
    means = np.array([np.mean(by_rank[r]) for r in ranks])
    stds = np.array([np.std(by_rank[r]) for r in ranks])
    n_seeds = [len(by_rank[r]) for r in ranks]

    print(f"{'rank':>5} | {'n seeds':>7} | {'mean acc':>8} | {'std':>6}")
    for r, m, s, n in zip(ranks, means, stds, n_seeds):
        print(f"{r:>5} | {n:>7} | {m:>8.4f} | {s:>6.4f}")
    best = ranks[int(np.argmax(means))]
    print(f"\nbest rank (no DP): {best} (accuracy={means[ranks.index(best)]:.4f})")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ranks, means, marker="o", markersize=8, linewidth=2, color=COLOR, solid_capstyle="round", zorder=3)
    ax.fill_between(ranks, means - stds, means + stds, alpha=0.15, color=COLOR, zorder=1)
    ax.scatter([best], [means[ranks.index(best)]], s=160, facecolors="none",
               edgecolors=COLOR, linewidths=2, zorder=4)

    ax.set_title("No-DP rank sweep -- accuracy vs rank", fontsize=13, color=INK, pad=10)
    ax.set_xlabel("LoRA rank", fontsize=10, color=SECONDARY_INK)
    ax.set_ylabel(f"Accuracy (mean of last {args.tail} rounds, shaded = ±1 std across seeds)",
                  fontsize=10, color=SECONDARY_INK)
    ax.set_xticks(ranks)
    ax.set_ylim(0.4, 1.0)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)

    fig.tight_layout()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {output_path}")


if __name__ == "__main__":
    main()
