"""
Round-by-round accuracy curves for the `dp` sweep (rank=8, epsilon in
{none,1,2,4,8}, seeds 42/43/44) -- one panel per epsilon, one line per seed.

Unlike the summary trend plot (figures/sweep/dp_trend.png, mean of the last
5 rounds), this shows the actual training dynamics: DP-SGD starts pinned at
the majority-class baseline (LoRA's B is zero-initialized, so round 1 is
just the frozen backbone + a random head) and has to escape that flat region
before it can learn. Noise competes with the escape gradient, so tighter
epsilon delays the escape -- sometimes past round 100 entirely. That
escape-timing spread, not simple noise-around-a-mean, is what widens the
trend plot's error bars at low epsilon.

Reads directly from results/sweep/*.json -- no ffalora/torch import needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path("results/sweep")
OUTPUT_PATH = Path("figures/sweep/dp_escape_curves.png")

EPSILONS = [None, 1, 2, 4, 8]
SEEDS = [42, 43, 44]
BASELINE_ACCURACY = 0.5092

# Validated categorical triplet (dataviz skill reference palette) -- first
# three slots clear the all-pairs CVD/contrast floors, the right guarantee
# for a small-multiples chart that repeats the same 3 series across panels.
SEED_COLORS = {42: "#2a78d6", 43: "#eb6834", 44: "#1baf7a"}  # blue, orange, aqua

MUTED = "#898781"
GRID = "#e1e0d9"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"


def _load(eps: float | None, seed: int) -> dict:
    tag = "none" if eps is None else f"{eps:g}"
    path = RESULTS_DIR / f"dp_sweep_rank8_eps{tag}_seed{seed}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    fig, axes = plt.subplots(1, len(EPSILONS), figsize=(20, 4.2), sharey=True)

    for ax, eps in zip(axes, EPSILONS):
        for seed in SEEDS:
            result = _load(eps, seed)
            history = result["history"]
            rounds = [entry["round_index"] for entry in history]
            accuracy = [entry["accuracy"] for entry in history]
            ax.plot(rounds, accuracy, linewidth=2, color=SEED_COLORS[seed],
                     label=f"seed {seed}", solid_capstyle="round")

        ax.axhline(BASELINE_ACCURACY, linestyle="--", linewidth=1, color=MUTED, zorder=0)

        label = "No DP" if eps is None else f"ε = {eps:g}"
        ax.set_title(label, fontsize=12, color=INK, pad=8)
        ax.set_xlabel("Round", fontsize=10, color=SECONDARY_INK)
        ax.set_ylim(0.35, 1.0)
        ax.set_xlim(0, 100)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9)

    axes[0].set_ylabel("Accuracy", fontsize=10, color=SECONDARY_INK)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.06), fontsize=10, labelcolor=SECONDARY_INK)

    fig.suptitle(
        "DP-SGD's escape-from-initialization delay grows (and becomes seed-dependent) as ε shrinks",
        fontsize=13, color=INK, y=1.18,
    )

    fig.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
