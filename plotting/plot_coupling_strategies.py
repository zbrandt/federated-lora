from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from federated_lora.coupling import linear_coupling, threshold_coupling

OUTPUT_PATH = Path("figures/sweep/coupling_strategies.png")

INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
LINEAR_COLOR = "#2a78d6"
THRESHOLD_COLOR = "#eb6834"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eps-min", type=float, default=0.25)
    parser.add_argument("--eps-max", type=float, default=16.0)
    parser.add_argument("--candidate-ranks", type=int, nargs="+", default=[2, 8, 16, 32])
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args()

    candidate_ranks = tuple(args.candidate_ranks)
    eps_grid = np.geomspace(args.eps_min, args.eps_max, 400)

    linear_ranks = np.array([linear_coupling(float(e), candidate_ranks) for e in eps_grid])
    threshold_ranks = np.array([threshold_coupling(float(e), candidate_ranks) for e in eps_grid])
    disagree = linear_ranks != threshold_ranks

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.step(eps_grid, linear_ranks, where="post", linewidth=2, color=LINEAR_COLOR, label="linear_coupling")
    ax.step(eps_grid, threshold_ranks, where="post", linewidth=2, color=THRESHOLD_COLOR,
            linestyle="--", label="threshold_coupling")

    if disagree.any():
        ax.fill_between(eps_grid, min(candidate_ranks), max(candidate_ranks),
                         where=disagree, color=MUTED, alpha=0.12, step="post",
                         label="strategies disagree")

    ax.set_xscale("log")
    ax.set_yscale("log", base=2)
    ax.set_yticks(candidate_ranks)
    ax.set_yticklabels([str(r) for r in candidate_ranks])
    ax.set_xlabel("client epsilon", fontsize=10, color=SECONDARY_INK)
    ax.set_ylabel("assigned LoRA rank", fontsize=10, color=SECONDARY_INK)
    ax.set_title("rank<->DP coupling strategies: linear vs threshold", fontsize=12, color=INK, pad=10)
    ax.grid(True, which="both", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, labelcolor=SECONDARY_INK, fontsize=9, loc="upper left")

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

    disagreement_frac = disagree.mean()
    print(f"disagreement across eps in [{args.eps_min}, {args.eps_max}]: {disagreement_frac:.1%} of sampled points")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
