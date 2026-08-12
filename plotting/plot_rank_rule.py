from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ffalora.config import Config
from ffalora.rank_rule import ACC_NODP_BY_RANK, GLUE_TRAIN_SIZES, kappa, noise_multiplier
from rankrule.validate_rank_rule import load_grid

OUTPUT_PATH = Path("figures/sweep/rank_rule_model.png")

INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
RANK_COLORS = {2: "#2a78d6", 4: "#eb6834", 6: "#1baf7a", 8: "#eda100", 16: "#8a5cd6", 32: "#c94f7c"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranks", type=int, nargs="+", default=[2, 4, 6, 8])
    parser.add_argument("--eps-min", type=float, default=0.5)
    parser.add_argument("--eps-max", type=float, default=16.0)
    parser.add_argument("--tail", type=int, default=5)
    parser.add_argument("--min-seeds", type=int, default=3)
    parser.add_argument("--results-dir", default="results/sweep")
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args()

    config = Config()
    shard_size = GLUE_TRAIN_SIZES[config.dataset_task] / config.num_clients
    ranks = [r for r in args.ranks if r in ACC_NODP_BY_RANK]

    eps_grid = np.geomspace(args.eps_min, args.eps_max, 40)
    sigma_grid = np.array([noise_multiplier(config, float(eps), shard_size) for eps in eps_grid])
    k = kappa(config, shard_size)

    print(f"kappa = {k:.6f} (anchor: rank 8, eps 1.0)")
    print(f"{'rank':>5} | {'acc_nodp':>8}")
    for r in ranks:
        print(f"{r:>5} | {ACC_NODP_BY_RANK[r]:>8.4f}")

    measured = load_grid(Path(args.results_dir), args.tail)

    fig, (ax_sigma, ax_acc) = plt.subplots(1, 2, figsize=(13, 5.5))

    ax_sigma.plot(eps_grid, sigma_grid, linewidth=2, color=INK)
    ax_sigma.set_xscale("log")
    ax_sigma.set_yscale("log")
    ax_sigma.set_title("DP noise multiplier sigma(eps)", fontsize=12, color=INK, pad=10)
    ax_sigma.set_xlabel("target epsilon", fontsize=10, color=SECONDARY_INK)
    ax_sigma.set_ylabel("sigma", fontsize=10, color=SECONDARY_INK)
    ax_sigma.grid(True, which="both", color=GRID, linewidth=0.8)
    ax_sigma.set_axisbelow(True)

    modeled_by_rank = {r: ACC_NODP_BY_RANK[r] - k * r * sigma_grid ** 2 for r in ranks}

    for r in ranks:
        color = RANK_COLORS.get(r, MUTED)
        ax_acc.plot(eps_grid, modeled_by_rank[r], linewidth=2, color=color, label=f"rank {r} (model)", zorder=2)

        eps_seen, acc_seen, n_seeds = [], [], []
        for eps, by_rank in measured.items():
            if eps is None or r not in by_rank:
                continue
            accs = by_rank[r]
            if len(accs) < args.min_seeds:
                continue
            eps_seen.append(eps)
            acc_seen.append(sum(accs) / len(accs))
            n_seeds.append(len(accs))
        if eps_seen:
            ax_acc.scatter(eps_seen, acc_seen, s=70, color=color, edgecolors=INK, linewidths=0.8,
                            zorder=3, label=f"rank {r} (measured, n>={min(n_seeds)})")

    stacked = np.stack([modeled_by_rank[r] for r in ranks])
    best_rank_by_eps = np.array(ranks)[np.argmax(stacked, axis=0)]
    switches = np.nonzero(best_rank_by_eps[:-1] != best_rank_by_eps[1:])[0]
    for idx in switches:
        eps_at_switch = float(eps_grid[idx])
        ax_acc.axvline(eps_at_switch, color=MUTED, linestyle=":", linewidth=1, zorder=1)
        ax_acc.annotate(f"rank {best_rank_by_eps[idx]}->{best_rank_by_eps[idx + 1]}\nat eps~{eps_at_switch:.2g}",
                         xy=(eps_at_switch, 0.42), fontsize=8, color=MUTED, ha="center")

    ax_acc.set_xscale("log")
    ax_acc.set_ylim(0.35, 1.0)
    ax_acc.set_title("modeled vs measured accuracy(r, eps)", fontsize=12, color=INK, pad=10)
    ax_acc.set_xlabel("target epsilon", fontsize=10, color=SECONDARY_INK)
    ax_acc.set_ylabel(f"accuracy (measured = mean of last {args.tail} rounds)", fontsize=10, color=SECONDARY_INK)
    ax_acc.grid(True, which="both", color=GRID, linewidth=0.8)
    ax_acc.set_axisbelow(True)
    ax_acc.legend(frameon=False, labelcolor=SECONDARY_INK, fontsize=8, loc="lower right")

    for ax in (ax_sigma, ax_acc):
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
