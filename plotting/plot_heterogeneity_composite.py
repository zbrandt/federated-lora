#!/usr/bin/env python
"""
plot_heterogeneity_composite.py -- "does heterogeneity help or hurt?" slide.

Three side-by-side convergence panels, sharing axes and a color scale keyed
to spread level, built from results/sweep/dp_spread_grid/gamma<g>_kappa<k>/:

  Panel 1: gamma varied (0.0, 0.33, 0.66, 1.0), kappa fixed at 0.0
           -- epsilon-spread only
  Panel 2: kappa varied (0.0, 0.33, 0.66, 1.0), gamma fixed at 0.0
           -- rank-spread only
  Panel 3: gamma == kappa varied together (0.0, 0.33, 0.66, 1.0)
           -- combined/compounding effect

Each line is top1_acc vs round, averaged across whatever methods have a
finished run in that cell (rblora/hetlora/flora/flexlora). The full grid is
16 (gamma, kappa) cells; as of writing only 3 are done (gamma1.0_kappa0.0,
gamma0.0_kappa1.0, gamma1.0_kappa1.0), so each panel currently plots at most
one line. Panels missing any of their 4 required cells are watermarked
"PARTIAL" and get a subtitle reporting how many levels are actually in.

Usage:
    python plotting/plot_heterogeneity_composite.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_ROOT = Path("results/sweep/dp_spread_grid")
OUTPUT_PATH = Path("figures/sweep/heterogeneity_composite.png")

INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"

LEVELS = [0.0, 0.33, 0.66, 1.0]
# sequential light->dark blue, keyed to spread level so the same level reads
# the same color in every panel
LEVEL_COLORS = {
    0.0: "#c3d2e8",
    0.33: "#7fa4d1",
    0.66: "#3d6fad",
    1.0: "#0f3b6e",
}

PANELS = [
    ("gamma varied, kappa=0.0\n(epsilon-spread only)", lambda lvl: (lvl, 0.0)),
    ("kappa varied, gamma=0.0\n(rank-spread only)", lambda lvl: (0.0, lvl)),
    ("gamma = kappa varied jointly\n(combined effect)", lambda lvl: (lvl, lvl)),
]


def load_cell(gamma: float, kappa: float) -> tuple[list[int], list[float], int] | None:
    cell_dir = RESULTS_ROOT / f"gamma{gamma}_kappa{kappa}"
    if not cell_dir.exists():
        return None
    per_round: dict[int, list[float]] = {}
    n_methods = 0
    for json_path in sorted(cell_dir.glob("*.json")):
        result = json.loads(json_path.read_text(encoding="utf-8"))
        history = result["history"]
        if not history:
            continue
        n_methods += 1
        for entry in history:
            per_round.setdefault(entry["round_index"], []).append(entry["top1_acc"])
    if n_methods == 0:
        return None
    rounds = sorted(per_round)
    means = [sum(per_round[r]) / len(per_round[r]) for r in rounds]
    return rounds, means, n_methods


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args()

    # gather all panel data first so we can share y-limits across panels
    panel_data = []  # list of dict: level -> (rounds, means, n_methods) or None
    all_means: list[float] = []
    for _, cell_fn in PANELS:
        levels_data = {}
        for lvl in LEVELS:
            gamma, kappa = cell_fn(lvl)
            cell = load_cell(gamma, kappa)
            levels_data[lvl] = cell
            if cell is not None:
                all_means.extend(cell[1])
        panel_data.append(levels_data)

    if not all_means:
        raise SystemExit(f"no dp_spread_grid results found under {RESULTS_ROOT}")

    y_lo, y_hi = min(all_means), max(all_means)
    y_pad = (y_hi - y_lo) * 0.08 or 1.0
    y_lo, y_hi = y_lo - y_pad, y_hi + y_pad
    x_hi = max((cell[0][-1] for levels_data in panel_data for cell in levels_data.values() if cell), default=100)

    fig, axes = plt.subplots(1, 3, figsize=(16, 6.2), sharey=True)

    for ax, (title, _), levels_data in zip(axes, PANELS, panel_data):
        completed = sum(1 for c in levels_data.values() if c is not None)
        for lvl in LEVELS:
            cell = levels_data[lvl]
            if cell is None:
                continue
            rounds, means, n_methods = cell
            ax.plot(rounds, means, linewidth=2.2, color=LEVEL_COLORS[lvl], solid_capstyle="round",
                     zorder=3, label=f"level {lvl:.2f} (n={n_methods} methods)")

        subtitle = f"{completed}/4 levels complete"
        if completed < len(LEVELS):
            subtitle = "PARTIAL -- " + subtitle
            ax.text(0.5, 0.5, "PARTIAL", transform=ax.transAxes, fontsize=42, color=MUTED,
                     alpha=0.18, ha="center", va="center", rotation=28, zorder=1, weight="bold")

        ax.set_title(f"{title}\n{subtitle}", fontsize=10.5, color=INK, pad=10)
        ax.set_xlabel("round", fontsize=9.5, color=SECONDARY_INK)
        ax.set_xlim(1, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8.5)
        ax.legend(frameon=False, labelcolor=SECONDARY_INK, fontsize=8, loc="lower right")

    axes[0].set_ylabel("top-1 accuracy (%), mean across methods", fontsize=9.5, color=SECONDARY_INK)

    fig.suptitle("Does heterogeneity help or hurt? epsilon-spread vs rank-spread vs combined",
                 fontsize=14, color=INK, y=1.02)
    fig.text(0.5, -0.02,
              "Baseline (gamma=0, kappa=0) not yet run -- panels currently show only their gamma/kappa=1.0 endpoint.",
              ha="center", fontsize=9, color=MUTED, style="italic")

    fig.tight_layout()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
