from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_PATH = Path("figures/sweep/aggregation_compare.png")

INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
STRATEGY_COLORS = {"naive": "#eb6834", "rank_aware": "#2a78d6"}
STRATEGY_LABELS = {"naive": "naive FedAvg", "rank_aware": "rank-aware FedAvg"}


def _final_accuracy(result: dict, tail: int) -> float:
    history = result["history"]
    window = history[-tail:] if len(history) >= tail else history
    return sum(entry["accuracy"] for entry in window) / len(window)


def load_comparison(results_dir: Path, tail: int) -> dict[str, list[float]]:
    accuracies: dict[str, list[float]] = defaultdict(list)
    for path in sorted(results_dir.glob("agg_compare_*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        strategy = result["config"]["aggregation_strategy"]
        accuracies[strategy].append(_final_accuracy(result, tail))
    return accuracies


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tail", type=int, default=5)
    parser.add_argument("--results-dir", default="results/aggregation_compare")
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args()

    accuracies = load_comparison(Path(args.results_dir), args.tail)
    strategies = [s for s in ("naive", "rank_aware") if s in accuracies]

    print(f"{'strategy':>12} | {'n seeds':>7} | {'mean acc':>8} | {'std':>6}")
    for strategy in strategies:
        accs = accuracies[strategy]
        mean = float(np.mean(accs))
        std = float(np.std(accs))
        print(f"{strategy:>12} | {len(accs):>7} | {mean:>8.4f} | {std:>6.4f}")

    fig, ax = plt.subplots(figsize=(6, 5))
    means = [float(np.mean(accuracies[s])) for s in strategies]
    stds = [float(np.std(accuracies[s])) for s in strategies]
    colors = [STRATEGY_COLORS[s] for s in strategies]
    labels = [STRATEGY_LABELS[s] for s in strategies]

    x = np.arange(len(strategies))
    ax.bar(x, means, yerr=stds, capsize=6, color=colors, width=0.5, zorder=3)
    for xi, acc, n in zip(x, means, (len(accuracies[s]) for s in strategies)):
        ax.annotate(f"{acc:.4f}", (xi, acc), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=10, color=INK)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(f"Accuracy (mean of last {args.tail} rounds, error bars = ±1 std across seeds)",
                  fontsize=10, color=SECONDARY_INK)
    ax.set_title("Rank-heterogeneous aggregation: naive vs rank-aware FedAvg", fontsize=13, color=INK, pad=10)
    ax.set_ylim(0.4, 1.0)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
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
