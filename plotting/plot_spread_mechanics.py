"""
Methodology figure (no experiment data needed): what gamma (epsilon-spread)
and kappa (rank-spread) actually do to the 4 clients' values in the
spread-grid sweep (sweep.py spread_grid / make_spread). Shows each client's
assigned epsilon/rank at every spread level, 0..1.
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

from sweep import EPS_FLOOR, MEAN_EPS, MEAN_RANK, NUM_CLIENTS, RANK_FLOOR, SPREAD_LEVELS, make_spread

OUTPUT_PATH = Path("figures/sweep/spread_mechanics.png")

INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
CLIENT_COLORS = ["#2a78d6", "#2a78d6", "#eb6834", "#eb6834"]  # first 2 clients (above mean) vs last 2 (below)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args()

    fig, (ax_eps, ax_rank) = plt.subplots(1, 2, figsize=(11, 5))

    for ax, mean, floor, label, round_int, spread_name in (
        (ax_eps, MEAN_EPS, EPS_FLOOR, "client epsilon", False, "gamma"),
        (ax_rank, MEAN_RANK, RANK_FLOOR, "client LoRA rank", True, "kappa"),
    ):
        for level in SPREAD_LEVELS:
            values = make_spread(NUM_CLIENTS, mean, level, floor, round_int=round_int)
            xs = [level] * NUM_CLIENTS
            for x, v, color in zip(xs, values, CLIENT_COLORS):
                ax.scatter(x, v, s=90, color=color, edgecolors=INK, linewidths=0.7, zorder=3)
        ax.axhline(mean, color=MUTED, linestyle=":", linewidth=1, zorder=1)
        ax.annotate(f"mean={mean:g}", xy=(1.02, mean), xycoords=("axes fraction", "data"),
                    fontsize=8, color=MUTED, va="center")
        ax.set_xticks(SPREAD_LEVELS)
        ax.set_xlabel(f"{spread_name} (spread level)", fontsize=10, color=SECONDARY_INK)
        ax.set_ylabel(label, fontsize=10, color=SECONDARY_INK)
        ax.set_title(f"{label} vs {spread_name}", fontsize=12, color=INK, pad=10)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9)

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=CLIENT_COLORS[0], markeredgecolor=INK, label="2 clients above mean"),
        plt.Line2D([0], [0], marker="o", linestyle="", color=CLIENT_COLORS[2], markeredgecolor=INK, label="2 clients below mean"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, labelcolor=SECONDARY_INK, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("spread-grid mechanics: 0 = everyone at the mean, 1 = max dispersion to floor", fontsize=11, color=SECONDARY_INK)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
