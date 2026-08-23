"""
Composite convergence map for dp_spread_grid: one panel per (gamma, kappa)
cell that has landed, sharing a y-axis so panels are directly comparable.
Gamma = epsilon-spread level, kappa = rank-spread level (see sweep.py's
make_spread); mean epsilon/rank held fixed across every panel, so any visual
difference between panels is attributable to which axis is heterogeneous,
not to a different average budget.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_PATH = Path("figures/sweep/dp_spread_grid_composite.png")

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


def load_cell(cell_dir: Path) -> dict[str, list[dict]]:
    histories = {}
    for path in sorted(cell_dir.glob("*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        method = result["config"]["method"]
        if result["history"]:
            histories[method] = result["history"]
    return histories


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default="results/sweep/dp_spread_grid")
    parser.add_argument("--metric", default="top1_acc", choices=["top1_acc", "top5_acc", "train_loss", "test_loss"])
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    cell_dirs = sorted(results_dir.glob("gamma*_kappa*"))
    if not cell_dirs:
        raise SystemExit(f"no cells found under {results_dir}")

    n = len(cell_dirs)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    metric_label = {"top1_acc": "top-1 accuracy (%)", "top5_acc": "top-5 accuracy (%)",
                     "train_loss": "train loss", "test_loss": "test loss"}[args.metric]

    for ax, cell_dir in zip(axes, cell_dirs):
        gamma_str, kappa_str = cell_dir.name.split("_")
        gamma = gamma_str.replace("gamma", "")
        kappa = kappa_str.replace("kappa", "")
        histories = load_cell(cell_dir)
        for method, history in histories.items():
            rounds = [e["round_index"] for e in history]
            values = [e[args.metric] for e in history]
            ax.plot(rounds, values, linewidth=2, color=METHOD_COLORS.get(method, MUTED),
                     label=method, solid_capstyle="round", zorder=3)
        missing = set(METHOD_COLORS) - set(histories)
        subtitle = f"gamma={gamma}, kappa={kappa}"
        if missing:
            subtitle += f"\n(missing: {', '.join(sorted(missing))})"
        ax.set_title(subtitle, fontsize=11, color=INK, pad=8)
        ax.set_xlabel("round", fontsize=10, color=SECONDARY_INK)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9)

    axes[0].set_ylabel(metric_label, fontsize=10, color=SECONDARY_INK)

    handles = [plt.Line2D([0], [0], color=c, linewidth=2, label=m) for m, c in METHOD_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=len(METHOD_COLORS), frameon=False,
               labelcolor=SECONDARY_INK, fontsize=9, bbox_to_anchor=(0.5, -0.05))

    fig.suptitle("dp_spread_grid: same mean budget (eps=3.0, rank=16), only heterogeneity varies",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
