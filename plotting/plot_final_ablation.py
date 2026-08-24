#!/usr/bin/env python
"""
plot_final_ablation.py -- the "does heterogeneity help or hurt" ablation.

Compares final accuracy per method across three scenarios that all share the
same mean budget (eps=3.0, rank=16) but differ in *how* heterogeneity is
handled:
  - fixed_rank_het_dp   -- results/sweep/dp_spread_grid/gamma1.0_kappa0.0
                           (rank held fixed across clients, DP epsilon varies)
  - het_rank_fixed_dp   -- results/sweep/dp_spread_grid/gamma0.0_kappa1.0
                           (rank varies across clients, DP epsilon held fixed)
  - coupling_linear/threshold -- results/sweep/dp_coupling/<fn>
                           (both vary together, rank coupled to each client's
                           epsilon via the named strategy)

rank_rule is deliberately excluded here -- at num_clients=4 its fitted model
never prefers anything but rank 16 across the tested epsilon range, so its
"coupling" run is actually a constant-rank baseline, not a real coupling
strategy. See figures/sweep/optimal_rank_curve.png for the real measured
accuracy-vs-rank relationship instead.

Answers whether deliberately coupling rank to a client's privacy budget beats
leaving either axis fixed while the other is heterogeneous.

Usage:
    python plotting/plot_final_ablation.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_PATH = Path("figures/sweep/final_ablation.png")

INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
METHOD_COLORS = {
    "rblora": "#2a78d6",
    "hetlora": "#eb6834",
    "flora": "#1baf7a",
    "flexlora": "#eda100",
    "ffa_lora": "#8a5fd1",
}

SCENARIOS = [
    ("fixed rank\n+ het DP", "results/sweep/dp_spread_grid/gamma1.0_kappa0.0"),
    ("het rank\n+ fixed DP", "results/sweep/dp_spread_grid/gamma0.0_kappa1.0"),
    ("coupled\n(linear)", "results/sweep/dp_coupling/linear"),
    ("coupled\n(threshold)", "results/sweep/dp_coupling/threshold"),
]


def final_value(history: list[dict], metric: str, tail: int) -> float:
    window = history[-tail:] if len(history) >= tail else history
    return sum(entry[metric] for entry in window) / len(window)


def load_scenario(results_dir: Path, metric: str, tail: int) -> dict[str, float]:
    values: dict[str, float] = {}
    for json_path in sorted(results_dir.glob("*.json")):
        result = json.loads(json_path.read_text(encoding="utf-8"))
        method = result["config"]["method"]
        history = result["history"]
        if not history:
            continue
        values[method] = final_value(history, metric, tail)
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metric", default="top1_acc", choices=["top1_acc", "top5_acc"])
    parser.add_argument("--tail", type=int, default=5, help="average the last N rounds instead of just the final round")
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args()

    metric_label = {"top1_acc": "top-1 accuracy (%)", "top5_acc": "top-5 accuracy (%)"}[args.metric]

    scenario_data = []
    all_methods: list[str] = []
    for label, dir_str in SCENARIOS:
        results_dir = Path(dir_str)
        if not results_dir.exists():
            print(f"skipping {label!r}: {results_dir} not found")
            continue
        values = load_scenario(results_dir, args.metric, args.tail)
        if not values:
            print(f"skipping {label!r}: no finished runs in {results_dir}")
            continue
        scenario_data.append((label, values))
        for m in values:
            if m not in all_methods:
                all_methods.append(m)

    if not scenario_data:
        raise SystemExit("no scenarios had any finished runs")

    n_scenarios = len(scenario_data)
    n_methods = len(all_methods)
    width = 0.8 / n_methods
    x = np.arange(n_scenarios)

    fig, ax = plt.subplots(figsize=(2.4 * n_scenarios + 2, 5.5))

    for i, method in enumerate(all_methods):
        heights = [values.get(method, np.nan) for _, values in scenario_data]
        offset = (i - (n_methods - 1) / 2) * width
        color = METHOD_COLORS.get(method, MUTED)
        ax.bar(x + offset, heights, width=width * 0.9, color=color, label=method, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _ in scenario_data], fontsize=9.5)
    ax.set_ylabel(f"{metric_label} (mean of last {args.tail} rounds)", fontsize=10, color=SECONDARY_INK)
    ax.set_title("Does heterogeneity help or hurt? fixed-rank vs het-rank vs coupled DP↔rank",
                 fontsize=12, color=INK, pad=10)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.legend(frameon=False, labelcolor=SECONDARY_INK, fontsize=9, ncol=n_methods, loc="upper center",
              bbox_to_anchor=(0.5, -0.12))

    fig.tight_layout()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
