"""
Final accuracy per (eps, rank) cell x method, for the current num_clients=4
grid sweep (results/sweep/dp_rank_grid/eps<e>_rank<r>/) -- NOT the same data
as figures/sweep/rank_dp_grid.png, which reads the legacy num_clients=20
grid_sweep_r*_eps*_seed*.json files instead. Only cells that have landed are
plotted; this is meant to be safe to rerun as more of the 4x4 grid completes.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_PATH = Path("figures/sweep/dp_rank_grid_nc4.png")

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


def _final_accuracy(result: dict, tail: int) -> float:
    history = result["history"]
    window = history[-tail:] if len(history) >= tail else history
    return sum(entry["top1_acc"] for entry in window) / len(window) / 100.0


def load_cells(results_dir: Path, tail: int) -> dict[str, dict[str, float]]:
    """{cell_label: {method: final_accuracy}} sorted by (eps, rank)."""
    cells: dict[tuple[float, int], dict[str, float]] = defaultdict(dict)
    for cell_dir in sorted(results_dir.glob("eps*_rank*")):
        eps_str, rank_str = cell_dir.name.split("_")
        eps = float(eps_str.replace("eps", ""))
        rank = int(rank_str.replace("rank", ""))
        for path in sorted(cell_dir.glob("*.json")):
            result = json.loads(path.read_text(encoding="utf-8"))
            method = result["config"]["method"]
            cells[(eps, rank)][method] = _final_accuracy(result, tail)
    return {f"eps={eps:g}\nrank={rank}": methods for (eps, rank), methods in sorted(cells.items())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tail", type=int, default=5)
    parser.add_argument("--results-dir", default="results/sweep/dp_rank_grid")
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args()

    cells = load_cells(Path(args.results_dir), args.tail)
    if not cells:
        raise SystemExit(f"no cells found under {args.results_dir}")

    labels = list(cells)
    x = np.arange(len(labels))
    width = 0.8 / len(METHODS)

    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(labels) + 2), 5.5))

    for i, method in enumerate(METHODS):
        values = [cells[label].get(method) for label in labels]
        offsets = x + (i - (len(METHODS) - 1) / 2) * width
        present_x = [ox for ox, v in zip(offsets, values) if v is not None]
        present_v = [v for v in values if v is not None]
        ax.bar(present_x, present_v, width=width * 0.9, color=METHOD_COLORS[method],
               label=method, edgecolor=INK, linewidth=0.6)
        missing_x = [ox for ox, v in zip(offsets, values) if v is None]
        for mx in missing_x:
            ax.annotate("missing", xy=(mx, 0.02), rotation=90, fontsize=7, color=MUTED,
                        ha="center", va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, color=SECONDARY_INK)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel(f"final accuracy (mean of last {args.tail} rounds)", fontsize=10, color=SECONDARY_INK)
    ax.set_title("dp_rank_grid (num_clients=4): accuracy by (eps, rank) cell", fontsize=12, color=INK, pad=10)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, labelcolor=SECONDARY_INK, fontsize=9, loc="lower right", ncol=len(METHODS))

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
    print(f"plotted {len(labels)} cell(s): {', '.join(l.replace(chr(10), ' ') for l in labels)}")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
