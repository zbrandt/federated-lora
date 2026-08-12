from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_PATH = Path("figures/sweep/epsilon_spread.png")

INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
MODE_COLORS = {"uniform": "#eb6834", "personalized": "#2a78d6"}
MODE_LABELS = {"uniform": "uniform rank (from mean eps)", "personalized": "personalized rank (from own eps)"}

TAG_RE = re.compile(r"epsspread_(uniform|personalized)_spread([\d.]+)_meaneps([\d.]+)_seed(\d+)")


def _final_accuracy(result: dict, tail: int) -> float:
    history = result["history"]
    window = history[-tail:] if len(history) >= tail else history
    return sum(entry["accuracy"] for entry in window) / len(window)


def load_comparison(results_dir: Path, tail: int) -> dict[tuple[float, str], list[float]]:
    accuracies: dict[tuple[float, str], list[float]] = defaultdict(list)
    for path in sorted(results_dir.glob("epsspread_*.json")):
        m = TAG_RE.match(path.stem)
        if not m:
            continue
        mode, spread = m.group(1), float(m.group(2))
        result = json.loads(path.read_text(encoding="utf-8"))
        accuracies[(spread, mode)].append(_final_accuracy(result, tail))
    return accuracies


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tail", type=int, default=5)
    parser.add_argument("--results-dir", default="results/epsilon_spread")
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args()

    accuracies = load_comparison(Path(args.results_dir), args.tail)
    spreads = sorted({spread for spread, _mode in accuracies})

    print(f"{'spread':>7} | {'mode':>12} | {'n seeds':>7} | {'mean acc':>8} | {'std':>6}")
    for spread in spreads:
        for mode in ("uniform", "personalized"):
            accs = accuracies.get((spread, mode), [])
            if not accs:
                continue
            print(f"{spread:>7g} | {mode:>12} | {len(accs):>7} | {np.mean(accs):>8.4f} | {np.std(accs):>6.4f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    for mode in ("uniform", "personalized"):
        xs = [s for s in spreads if accuracies.get((s, mode))]
        means = np.array([np.mean(accuracies[(s, mode)]) for s in xs])
        stds = np.array([np.std(accuracies[(s, mode)]) for s in xs])
        color = MODE_COLORS[mode]
        ax.plot(xs, means, marker="o", markersize=8, linewidth=2, color=color,
                 label=MODE_LABELS[mode], solid_capstyle="round", zorder=3)
        ax.fill_between(xs, means - stds, means + stds, alpha=0.15, color=color, zorder=1)

    ax.set_title("Personalized vs uniform rank as epsilon spread widens (mean eps fixed)",
                  fontsize=12, color=INK, pad=10)
    ax.set_xlabel("epsilon spread (0 = every client identical)", fontsize=10, color=SECONDARY_INK)
    ax.set_ylabel(f"Accuracy (mean of last {args.tail} rounds, shaded = ±1 std across seeds)",
                  fontsize=10, color=SECONDARY_INK)
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
    print(f"\nwrote {output_path}")


if __name__ == "__main__":
    main()
