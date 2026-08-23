"""
Composite convergence map for dp_rank_grid: one panel per (eps, rank) cell
that has landed, arranged in a grid and sharing a y-axis so panels are
directly comparable. num_clients=4 throughout.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_PATH = Path("figures/sweep/dp_rank_grid_composite.png")

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


def load_cell(cell_dir: Path) -> dict[str, list[dict]]:
    histories = {}
    for path in sorted(cell_dir.glob("*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        method = result["config"]["method"]
        if result["history"]:
            histories[method] = result["history"]
    return histories


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default="results/sweep/dp_rank_grid")
    parser.add_argument("--metric", default="top1_acc", choices=["top1_acc", "top5_acc", "train_loss", "test_loss"])
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    def sort_key(p: Path) -> tuple[float, int]:
        eps_str, rank_str = p.name.split("_")
        return float(eps_str.replace("eps", "")), int(rank_str.replace("rank", ""))

    cell_dirs = sorted(results_dir.glob("eps*_rank*"), key=sort_key)
    if not cell_dirs:
        raise SystemExit(f"no cells found under {results_dir}")

    n = len(cell_dirs)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.2 * rows), sharey=True, squeeze=False)
    flat_axes = axes.flatten()

    metric_label = {"top1_acc": "top-1 accuracy (%)", "top5_acc": "top-5 accuracy (%)",
                     "train_loss": "train loss", "test_loss": "test loss"}[args.metric]

    for ax, cell_dir in zip(flat_axes, cell_dirs):
        eps_str, rank_str = cell_dir.name.split("_")
        eps = eps_str.replace("eps", "")
        rank = rank_str.replace("rank", "")
        histories = load_cell(cell_dir)
        for method, history in histories.items():
            rounds = [e["round_index"] for e in history]
            values = [e[args.metric] for e in history]
            ax.plot(rounds, values, linewidth=2, color=METHOD_COLORS.get(method, MUTED),
                     label=method, solid_capstyle="round", zorder=3)
        missing = set(METHOD_COLORS) - set(histories)
        subtitle = f"eps={eps}, rank={rank}"
        if missing:
            subtitle += f"\n(missing: {', '.join(sorted(missing))})"
        ax.set_title(subtitle, fontsize=11, color=INK, pad=8)
        ax.set_xlabel("round", fontsize=10, color=SECONDARY_INK)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9)

    for ax in flat_axes[:n]:
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel(metric_label, fontsize=10, color=SECONDARY_INK)
    for ax in flat_axes[n:]:
        ax.set_visible(False)

    handles = [plt.Line2D([0], [0], color=c, linewidth=2, label=m) for m, c in METHOD_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=len(METHOD_COLORS), frameon=False,
               labelcolor=SECONDARY_INK, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("dp_rank_grid (num_clients=4): accuracy convergence by (eps, rank) cell",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
