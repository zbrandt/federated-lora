#!/usr/bin/env python
"""
plot_optimal_rank_curve.py -- accuracy vs rank, one line per epsilon, built
directly from measured results/sweep/dp_rank_grid/eps<e>_rank<r>/ cells
(num_clients=4, the domain the coupling sweep actually runs in).

This replaces the rank_rule model's fitted curve with the real thing: no
regression, no extrapolation, just what was actually measured. Point markers
are per-method values; the line/shaded band is the mean +/- std across
methods at that (eps, rank) cell. Only cells that have actually landed are
plotted -- safe to rerun as more of the grid completes.

Usage:
    python plotting/plot_optimal_rank_curve.py
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

OUTPUT_PATH = Path("figures/sweep/optimal_rank_curve.png")

INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
EPS_COLORS = {0.5: "#2a78d6", 1.0: "#8a5fd1", 3.0: "#eb6834", 8.0: "#1baf7a"}
METHOD_MARKERS = {"rblora": "o", "hetlora": "s", "flora": "^", "flexlora": "D"}


def _final_accuracy(result: dict, tail: int) -> float:
    history = result["history"]
    window = history[-tail:] if len(history) >= tail else history
    return sum(entry["top1_acc"] for entry in window) / len(window)


def load_cells(results_dir: Path, tail: int) -> dict[tuple[float, int], dict[str, float]]:
    cells: dict[tuple[float, int], dict[str, float]] = defaultdict(dict)
    for cell_dir in sorted(results_dir.glob("eps*_rank*")):
        eps_str, rank_str = cell_dir.name.split("_")
        eps = float(eps_str.replace("eps", ""))
        rank = int(rank_str.replace("rank", ""))
        for path in sorted(cell_dir.glob("*.json")):
            result = json.loads(path.read_text(encoding="utf-8"))
            method = result["config"]["method"]
            if not result["history"]:
                continue
            cells[(eps, rank)][method] = _final_accuracy(result, tail)
    return cells


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tail", type=int, default=5)
    parser.add_argument("--results-dir", default="results/sweep/dp_rank_grid")
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args()

    cells = load_cells(Path(args.results_dir), args.tail)
    if not cells:
        raise SystemExit(f"no cells found under {args.results_dir}")

    by_eps: dict[float, dict[int, dict[str, float]]] = defaultdict(dict)
    for (eps, rank), methods in cells.items():
        by_eps[eps][rank] = methods

    all_ranks = sorted({rank for methods_by_rank in by_eps.values() for rank in methods_by_rank})

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for eps in sorted(by_eps):
        ranks = sorted(by_eps[eps])
        means = np.array([np.mean(list(by_eps[eps][r].values())) for r in ranks])
        stds = np.array([np.std(list(by_eps[eps][r].values())) for r in ranks])
        color = EPS_COLORS.get(eps, MUTED)
        n_methods_label = "/".join(str(len(by_eps[eps][r])) for r in ranks)
        label = f"eps={eps:g}  (n methods per point: {n_methods_label})"

        if len(ranks) >= 2:
            ax.plot(ranks, means, linewidth=2, color=color, solid_capstyle="round", zorder=3, label=label)
            ax.fill_between(ranks, means - stds, means + stds, alpha=0.15, color=color, zorder=1)
        else:
            ax.scatter(ranks, means, s=10, color=color, zorder=3, label=label + "  [single point -- no line]")

        for r in ranks:
            for method, acc in by_eps[eps][r].items():
                ax.scatter([r], [acc], marker=METHOD_MARKERS.get(method, "x"), s=35,
                           facecolors="none", edgecolors=color, linewidths=1.2, zorder=4)

    method_legend = [plt.Line2D([0], [0], marker=m, color=MUTED, linestyle="", markerfacecolor="none",
                                 markeredgecolor=MUTED, label=name)
                      for name, m in METHOD_MARKERS.items()]

    ax.set_xscale("log", base=2)
    ax.set_xticks(all_ranks)
    ax.set_xticklabels([str(r) for r in all_ranks])
    ax.set_xlabel("LoRA rank", fontsize=10, color=SECONDARY_INK)
    ax.set_ylabel(f"top-1 accuracy (%), mean of last {args.tail} rounds", fontsize=10, color=SECONDARY_INK)
    ax.set_title("Measured accuracy vs rank across DP budgets (num_clients=4, real data -- no model)",
                 fontsize=12, color=INK, pad=10)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)

    eps_legend = ax.legend(frameon=False, labelcolor=SECONDARY_INK, fontsize=8.5, loc="lower right")
    ax.add_artist(eps_legend)
    ax.legend(handles=method_legend, frameon=False, labelcolor=SECONDARY_INK, fontsize=8,
              loc="upper left", title="markers = per-method points", title_fontsize=8)

    fig.tight_layout()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output_path}")
    print(f"cells plotted: {sorted(cells.keys())}")


if __name__ == "__main__":
    main()
