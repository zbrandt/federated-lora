#!/usr/bin/env python
"""
plot_aggregation_error.py -- aggregation-error-vs-round comparison across
methods, reading the CSVs written by analyze_aggregation_error.py
(columns: label, method, round, aggregation_error).

Under the spreadA scenario (client_ranks 2/8/16/32), rblora and flora share
one uploads run (results/aggregation_error/spreadA.csv) while hetlora needs
its own same-shape run (results/aggregation_error/spreadA_het.csv) -- see
analyze_aggregation_error.py's docstring. This just concatenates every *.csv
in --results-dir and plots one line per method, regardless of which file it
came from, since they're all measuring the same spreadA scenario.

Usage:
    python plotting/plot_aggregation_error.py
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_PATH = Path("figures/sweep/aggregation_error.png")

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


def load_series(results_dir: Path) -> dict[str, dict[str, list[tuple[int, float]]]]:
    # method -> label -> [(round, error), ...]
    series: dict[str, dict[str, list[tuple[int, float]]]] = defaultdict(lambda: defaultdict(list))
    csv_paths = sorted(results_dir.glob("*.csv"))
    if not csv_paths:
        raise SystemExit(f"no CSVs found in {results_dir}")
    for path in csv_paths:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                series[row["method"]][row["label"]].append((int(row["round"]), float(row["aggregation_error"])))
    return series


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default="results/aggregation_error")
    parser.add_argument("--title", default="Aggregation error under rank spread (client_ranks 2/8/16/32)")
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args()

    series = load_series(Path(args.results_dir))

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for method, by_label in series.items():
        color = METHOD_COLORS.get(method, MUTED)
        for label, points in by_label.items():
            points.sort()
            rounds = [p[0] for p in points]
            errors = [p[1] for p in points]
            line_label = method if len(by_label) == 1 else f"{method} ({label})"
            ax.plot(rounds, errors, linewidth=2, color=color, label=line_label,
                     solid_capstyle="round", zorder=3)

    ax.set_xlabel("round", fontsize=10, color=SECONDARY_INK)
    ax.set_ylabel("relative Frobenius aggregation error", fontsize=10, color=SECONDARY_INK)
    ax.set_title(args.title, fontsize=12, color=INK, pad=10)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.legend(frameon=False, labelcolor=SECONDARY_INK, fontsize=9)

    fig.tight_layout()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
