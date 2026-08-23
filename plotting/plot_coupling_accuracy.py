#!/usr/bin/env python
"""
plot_coupling_accuracy.py -- accuracy comparison of the DP<->rank coupling
strategies (linear vs threshold vs rank_rule).

Reads results/sweep/dp_coupling/{linear,threshold,rank_rule}/*.json (one file
per method) and draws one small-multiple subplot per method, each with one
accuracy-vs-round line per coupling strategy. Complements
plot_coupling_strategies.py (which shows what rank each strategy *assigns* at
a given epsilon) by showing what accuracy that assignment actually buys.

Usage:
    python plotting/plot_coupling_accuracy.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_ROOT = Path("results/sweep/dp_coupling")
OUTPUT_PATH = Path("figures/sweep/coupling_accuracy.png")

INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"

STRATEGIES = ["linear", "threshold", "rank_rule"]
STRATEGY_COLORS = {
    "linear": "#2a78d6",
    "threshold": "#eb6834",
    "rank_rule": "#1baf7a",
}
STRATEGY_STYLES = {
    "linear": "-",
    "threshold": "--",
    "rank_rule": ":",
}
METRIC_LABELS = {
    "top1_acc": "top-1 accuracy (%)",
    "top5_acc": "top-5 accuracy (%)",
}


def final_value(history: list[dict], metric: str, tail: int) -> float:
    window = history[-tail:] if len(history) >= tail else history
    return sum(entry[metric] for entry in window) / len(window)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-root", default=str(RESULTS_ROOT))
    parser.add_argument("--metric", default="top1_acc", choices=["top1_acc", "top5_acc"])
    parser.add_argument("--tail", type=int, default=5, help="mean of last N rounds, reported in the per-subplot legend")
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args()

    results_root = Path(args.results_root)

    # strategy -> method -> history
    data: dict[str, dict[str, list[dict]]] = {}
    for strategy in STRATEGIES:
        strategy_dir = results_root / strategy
        if not strategy_dir.exists():
            continue
        for json_path in sorted(strategy_dir.glob("*.json")):
            result = json.loads(json_path.read_text(encoding="utf-8"))
            method = result["config"]["method"]
            history = result["history"]
            if not history:
                continue
            data.setdefault(strategy, {})[method] = history

    if not data:
        raise SystemExit(f"no coupling results found under {results_root}")

    all_methods = sorted({m for methods in data.values() for m in methods})
    n = len(all_methods)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 4.2 * nrows), squeeze=False)

    for i, method in enumerate(all_methods):
        ax = axes[i // ncols][i % ncols]
        for strategy in STRATEGIES:
            history = data.get(strategy, {}).get(method)
            if not history:
                continue
            rounds = [e["round_index"] for e in history]
            values = [e[args.metric] for e in history]
            final = final_value(history, args.metric, args.tail)
            ax.plot(rounds, values, linewidth=2, color=STRATEGY_COLORS[strategy],
                     linestyle=STRATEGY_STYLES[strategy], solid_capstyle="round", zorder=3,
                     label=f"{strategy} (final {final:.1f})")

        ax.set_title(method, fontsize=11, color=INK, pad=8)
        ax.set_xlabel("round", fontsize=9.5, color=SECONDARY_INK)
        ax.set_ylabel(METRIC_LABELS[args.metric], fontsize=9.5, color=SECONDARY_INK)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8.5)
        ax.legend(frameon=False, labelcolor=SECONDARY_INK, fontsize=8)

    # hide unused axes if methods don't fill the grid
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle("DP<->rank coupling strategies: accuracy by method", fontsize=13, color=INK, y=1.0)
    fig.tight_layout()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
