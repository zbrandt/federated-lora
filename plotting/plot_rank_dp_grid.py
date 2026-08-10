from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ffalora.config import Config
from ffalora.rank_rule import GLUE_TRAIN_SIZES, recommended_rank
from validate_rank_rule import load_grid

OUTPUT_PATH = Path("figures/sweep/rank_dp_grid.png")

INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
EPS_COLORS = {1.0: "#2a78d6", 2.0: "#eb6834", 4.0: "#1baf7a", 8.0: "#eda100"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tail", type=int, default=5)
    parser.add_argument("--results-dir", default="results/sweep")
    parser.add_argument("--min-seeds", type=int, default=5)
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args()

    by_eps_rank = load_grid(Path(args.results_dir), args.tail)
    config = Config()
    shard_size = GLUE_TRAIN_SIZES[config.dataset_task] / config.num_clients

    fig, ax = plt.subplots(figsize=(7, 5))
    all_ranks: set[int] = set()

    for eps in sorted(eps for eps in by_eps_rank if eps is not None):
        by_rank = by_eps_rank[eps]
        ranks = sorted(rank for rank, accs in by_rank.items() if len(accs) >= args.min_seeds)
        if not ranks:
            continue
        all_ranks.update(ranks)

        means = np.array([np.mean(by_rank[rank]) for rank in ranks])
        stds = np.array([np.std(by_rank[rank]) for rank in ranks])
        color = EPS_COLORS.get(eps, MUTED)
        n_seeds = len(by_rank[ranks[0]])
        label = f"ε = {eps:g} (n={n_seeds})"

        ax.plot(ranks, means, marker="o", markersize=8, linewidth=2, color=color,
                 label=label, solid_capstyle="round", zorder=3)
        ax.fill_between(ranks, means - stds, means + stds, alpha=0.15, color=color, zorder=1)

        if len(ranks) >= 2:
            best_index = int(np.argmax(means))
            measured_best = ranks[best_index]
            ax.scatter([measured_best], [means[best_index]], s=160, facecolors="none",
                       edgecolors=color, linewidths=2, zorder=4)

            try:
                predicted = recommended_rank(config, eps, shard_size, candidate_ranks=tuple(ranks))
            except ValueError:
                predicted = None
            if predicted is not None and predicted != measured_best:
                ax.annotate(f"rule picked {predicted}", (measured_best, means[best_index]),
                            textcoords="offset points", xytext=(8, 8), fontsize=8, color=color)

    ax.set_title("Rank x DP grid sweep -- accuracy vs rank", fontsize=13, color=INK, pad=10)
    ax.set_xlabel("LoRA rank", fontsize=10, color=SECONDARY_INK)
    ax.set_ylabel(f"Accuracy (mean of last {args.tail} rounds, shaded = ±1 std across seeds)",
                  fontsize=10, color=SECONDARY_INK)
    if all_ranks:
        ax.set_xticks(sorted(all_ranks))
    ax.set_ylim(0.4, 1.0)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.legend(frameon=False, labelcolor=SECONDARY_INK)

    fig.tight_layout()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
