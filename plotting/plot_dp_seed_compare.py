from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path("results/sweep")
OUTPUT_PATH = Path("figures/sweep/dp_seed_compare.png")

EPSILONS = [None, 1.0, 2.0, 4.0, 8.0]
SEEDS = [42, 43, 44]
ESCAPE_THRESHOLD = 0.6
BASELINE_ACCURACY = 0.5092

INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SEED_COLORS = {42: "#2a78d6", 43: "#eb6834", 44: "#1baf7a"}


def _load(eps: float | None, seed: int) -> dict:
    tag = "none" if eps is None else f"{eps:g}"
    path = RESULTS_DIR / f"dp_sweep_rank8_eps{tag}_seed{seed}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tail", type=int, default=5)
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args()

    x_positions = list(range(len(EPSILONS)))
    x_labels = ["no-DP" if eps is None else f"ε = {eps:g}" for eps in EPSILONS]

    print(f"{'epsilon':>8} {'seed':>5} {'final acc':>10} {'escape round (acc>0.6)':>24}")
    per_seed: dict[int, list[float]] = {seed: [] for seed in SEEDS}
    escaped: dict[int, list[bool]] = {seed: [] for seed in SEEDS}
    for eps in EPSILONS:
        for seed in SEEDS:
            result = _load(eps, seed)
            history = result["history"]
            window = history[-args.tail:]
            final = sum(entry["accuracy"] for entry in window) / len(window)
            escape_round = next((entry["round_index"] for entry in history
                                  if entry["accuracy"] > ESCAPE_THRESHOLD), None)
            per_seed[seed].append(final)
            escaped[seed].append(final > ESCAPE_THRESHOLD)
            print(f"{str(eps):>8} {seed:>5} {final:>10.4f} {str(escape_round):>24}")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.axhline(BASELINE_ACCURACY, linestyle="--", linewidth=1, color=MUTED, zorder=0,
               label=f"majority-class floor ({BASELINE_ACCURACY:.3f})")

    for seed in SEEDS:
        color = SEED_COLORS[seed]
        accs = per_seed[seed]
        ax.plot(x_positions, accs, linewidth=2, color=color, zorder=2, solid_capstyle="round",
                 label=f"seed {seed}")
        escaped_x = [x for x, e in zip(x_positions, escaped[seed]) if e]
        escaped_y = [a for a, e in zip(accs, escaped[seed]) if e]
        stuck_x = [x for x, e in zip(x_positions, escaped[seed]) if not e]
        stuck_y = [a for a, e in zip(accs, escaped[seed]) if not e]
        ax.scatter(escaped_x, escaped_y, s=90, color=color, zorder=3, marker="o")
        ax.scatter(stuck_x, stuck_y, s=90, facecolors="none", edgecolors=color, linewidths=2,
                    zorder=3, marker="X")

    ax.set_title("DP sweep (rank 8) -- per-seed final accuracy by privacy budget",
                  fontsize=13, color=INK, pad=10)
    ax.set_xlabel("Privacy budget", fontsize=10, color=SECONDARY_INK)
    ax.set_ylabel(f"Accuracy (mean of last {args.tail} rounds)", fontsize=10, color=SECONDARY_INK)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0.4, 1.0)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, frameon=False, labelcolor=SECONDARY_INK, loc="upper right",
              bbox_to_anchor=(0.98, 0.7))
    fig.text(0.5, -0.02,
              "● escaped the collapse within 100 rounds     ✕ stayed stuck near the majority-class floor",
              ha="center", fontsize=9, color=MUTED)

    fig.tight_layout()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {output_path}")


if __name__ == "__main__":
    main()
