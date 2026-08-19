#!/usr/bin/env python
"""
plot_convergence.py — convergence map: a chosen metric vs. round, one line
per method, for a given sweep condition.

Reads every method's result JSON in a results directory (as written by
experiment.py -- each has a "history" list of per-round
{round_index, train_loss, test_loss, top1_acc, top5_acc}) and plots that
metric's trajectory across rounds. Shows not just final accuracy but how
each method gets there -- e.g. whether one method converges faster, or
oscillates, even if two methods land at similar final accuracy.

Usage:
    python plotting/plot_convergence.py --results-dir results/sweep/dp_compare/eps3.0 \
        --title "eps=3.0" --output figures/convergence/dp_compare_eps3.0.png
"""

from __future__ import annotations

import argparse
import json
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
    'train_loss': 'train loss',
    'test_loss': 'test loss',
}


def load_method_history(path: Path) -> tuple[str, list[dict]]:
    result = json.loads(path.read_text())
    return result['config']['method'], result['history']


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results-dir', required=True, help='directory containing one {method}_{task}_seed{seed}.json per method')
    parser.add_argument('--metric', default='top1_acc', choices=['top1_acc', 'top5_acc', 'train_loss', 'test_loss'])
    parser.add_argument('--title', default='')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    json_files = sorted(results_dir.glob('*.json'))
    if not json_files:
        raise SystemExit(f'no result json files found in {results_dir}')

    fig, ax = plt.subplots(figsize=(7, 5))
    plotted = 0

    for path in json_files:
        method, history = load_method_history(path)
        if not history:
            print(f'skipping {path.name}: empty history (run may still be in progress)')
            continue
        rounds = [entry['round_index'] for entry in history]
        values = [entry[args.metric] for entry in history]
        color = METHOD_COLORS.get(method, MUTED)
        ax.plot(rounds, values, linewidth=2, color=color, label=method, solid_capstyle='round', zorder=3)
        plotted += 1

    if plotted == 0:
        raise SystemExit(f'no non-empty histories found in {results_dir}')

    ax.set_title(args.title or f'Convergence — {results_dir.name}', fontsize=13, color=INK, pad=10)
    ax.set_xlabel('round', fontsize=10, color=SECONDARY_INK)
    ax.set_ylabel(METRIC_LABELS.get(args.metric, args.metric), fontsize=10, color=SECONDARY_INK)
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
