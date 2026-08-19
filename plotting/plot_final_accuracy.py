#!/usr/bin/env python
"""
plot_final_accuracy.py — final accuracy vs. a swept condition (epsilon or
rank), one line per method.

Different from plot_convergence.py: that shows the full trajectory across
rounds for ONE condition. This collapses each condition down to a single
number (mean of the last --tail rounds' accuracy, to smooth out per-round
noise) and plots that summary across MULTIPLE conditions -- e.g. the classic
privacy-utility tradeoff curve (accuracy vs. epsilon) or a rank sensitivity
curve (accuracy vs. rank).

Usage:
    python plotting/plot_final_accuracy.py --sweep-root results/sweep/dp_compare \
        --condition-prefix eps --xlabel epsilon --title "Privacy-utility tradeoff" \
        --output figures/final_accuracy/dp_compare_vs_eps.png

    python plotting/plot_final_accuracy.py --sweep-root results/sweep/dp_rank \
        --condition-prefix rank --xlabel "LoRA rank" --title "Rank sensitivity" \
        --output figures/final_accuracy/dp_rank_vs_rank.png
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

INK = '#0b0b0b'
SECONDARY_INK = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
METHOD_COLORS = {
    'rblora': '#2a78d6',
    'hetlora': '#eb6834',
    'flora': '#1baf7a',
    'flexlora': '#eda100',
    'ffa_lora': '#8a5fd1',
    'dp_lora': '#c0392b',
    'la_lora': '#16a085',
    'rolora': '#7f8c8d',
}
METRIC_LABELS = {
    'top1_acc': 'top-1 accuracy (%)',
    'top5_acc': 'top-5 accuracy (%)',
}


def final_value(history: list[dict], metric: str, tail: int) -> float:
    window = history[-tail:] if len(history) >= tail else history
    return sum(entry[metric] for entry in window) / len(window)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sweep-root', required=True, help='directory containing one subdir per condition, e.g. results/sweep/dp_compare')
    parser.add_argument('--condition-prefix', required=True, help='subdir name prefix to strip to get the numeric x-value, e.g. "eps" for "eps3.0" subdirs')
    parser.add_argument('--metric', default='top1_acc', choices=['top1_acc', 'top5_acc'])
    parser.add_argument('--tail', type=int, default=5, help='average the last N rounds instead of just the final round, to smooth noise')
    parser.add_argument('--xlabel', default='')
    parser.add_argument('--title', default='')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    sweep_root = Path(args.sweep_root)
    pattern = re.compile(rf'^{re.escape(args.condition_prefix)}([\d.]+)$')

    # method -> list of (x, y) sorted by x
    series: dict[str, list[tuple[float, float]]] = {}
    skipped = []

    for condition_dir in sorted(sweep_root.iterdir()):
        if not condition_dir.is_dir():
            continue
        match = pattern.match(condition_dir.name)
        if not match:
            continue
        x = float(match.group(1))

        for json_path in sorted(condition_dir.glob('*.json')):
            result = json.loads(json_path.read_text())
            method = result['config']['method']
            history = result['history']
            if not history:
                skipped.append(f'{condition_dir.name}/{json_path.name} (empty history)')
                continue
            y = final_value(history, args.metric, args.tail)
            series.setdefault(method, []).append((x, y))

    if not series:
        raise SystemExit(f'no matching condition dirs (prefix "{args.condition_prefix}") found under {sweep_root}')
    if skipped:
        print('skipped (not finished yet):', ', '.join(skipped))

    fig, ax = plt.subplots(figsize=(7, 5))
    for method, points in series.items():
        points.sort()
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        color = METHOD_COLORS.get(method, MUTED)
        ax.plot(xs, ys, marker='o', markersize=7, linewidth=2, color=color,
                 label=method, solid_capstyle='round', zorder=3)

    ax.set_title(args.title or f'Final accuracy vs {args.condition_prefix}', fontsize=13, color=INK, pad=10)
    ax.set_xlabel(args.xlabel or args.condition_prefix, fontsize=10, color=SECONDARY_INK)
    ax.set_ylabel(f'{METRIC_LABELS.get(args.metric, args.metric)} (mean of last {args.tail} rounds)',
                  fontsize=10, color=SECONDARY_INK)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    for spine in ('left', 'bottom'):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.legend(frameon=False, labelcolor=SECONDARY_INK)

    fig.tight_layout()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {output_path}')


if __name__ == '__main__':
    main()
