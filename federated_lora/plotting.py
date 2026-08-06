#!/usr/bin/env python3
"""
plotting.py — schema-tolerant plotter for federated-lora results.

Reads the per-run result JSONs you already have in results/ and writes
accuracy-by-round figures per task, without retraining and without touching the
harness. It is tolerant of the schema drift that broke `main.py plot`:

  * reads the server's `top1_acc` / `top5_acc` history fields (also accepts
    `accuracy` / `acc` and friends), and plots BOTH top-1 and top-5 when present
  * finds the round index under any common key (round_index / round / ...)
  * auto-detects a 0-100 accuracy scale and normalizes it to a 0-1 fraction
  * skips any unreadable file and prints why (so a stale/foreign JSON can't
    take the whole plot down)
  * prints a final-round mean+/-std table so you have exact numbers for the deck

Usage, from the repo root on the server:

    uv run python -m federated_lora plot                  # reads results/, writes figures/
    uv run python -m federated_lora plot results/ --output-dir figures
    uv run python -m federated_lora plot results/lalora_cifar100_seed42.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Metric label -> candidate history keys, in priority order.
METRICS = {
	'top1': ('top1_acc', 'accuracy', 'acc', 'test_accuracy', 'eval_accuracy',
	         'test_acc', 'acc@1', 'top_1_acc'),
	'top5': ('top5_acc', 'top_5_acc', 'acc@5'),
}
ROUND_KEYS = ('round_index', 'round', 'round_idx', 'step', 'r')


def _first(d: dict, keys, default=None):
	for k in keys:
		if k in d:
			return d[k]
	return default


def load_run(path: Path) -> dict:
	payload = json.loads(path.read_text(encoding='utf-8'))
	config = payload.get('config', {}) or {}
	history = payload.get('history', []) or []

	rounds: list[int] = []
	series: dict[str, list[float]] = {name: [] for name in METRICS}
	for i, entry in enumerate(history):
		# top-1 is required; skip entries that have no top-1-like field
		top1 = _first(entry, METRICS['top1'])
		if top1 is None:
			continue
		rounds.append(int(_first(entry, ROUND_KEYS, i + 1)))
		for name, keys in METRICS.items():
			val = _first(entry, keys)
			series[name].append(float(val) if val is not None else float('nan'))

	if not rounds:
		keys = sorted(history[0].keys()) if history else 'empty history'
		raise ValueError(f'no top-1 accuracy field found (entry keys: {keys})')

	# drop metrics that are entirely absent (all NaN), e.g. top5 not logged
	series = {
		name: vals
		for name, vals in series.items()
		if any(v == v for v in vals)  # v==v is False only for NaN
	}
	return {
		'method': str(config.get('method', path.stem)),
		'task': str(config.get('task', 'unknown')),
		'seed': int(config.get('seed', 0)),
		'rounds': rounds,
		'series': series,
	}


def gather_paths(inputs: list[str]) -> list[Path]:
	paths: list[Path] = []
	for raw in inputs:
		p = Path(raw)
		if p.is_dir():
			paths.extend(sorted(p.glob('*.json')))
		elif p.suffix == '.json':
			paths.append(p)
	return paths


def plot(
	inputs: list[str] | str = 'results', output_dir: str = 'figures'
) -> list[Path]:
	if isinstance(inputs, (str, Path)):
		inputs = [inputs]
	paths = gather_paths(inputs)
	if not paths:
		raise SystemExit(f'No JSON files found in: {inputs}')

	runs, skipped = [], []
	for p in paths:
		try:
			runs.append(load_run(p))
		except Exception as ex:  # noqa: BLE001 - report and continue
			skipped.append((p.name, str(ex)))

	for name, why in skipped:
		print(f'  skip {name}: {why}')
	if not runs:
		sys.exit('No usable runs — nothing to plot.')

	# Normalize a 0-100 accuracy scale down to a 0-1 fraction if needed.
	all_vals = [v for r in runs for s in r['series'].values() for v in s]
	max_acc = max(v for v in all_vals if v == v)
	if max_acc > 1.5:
		print(
			f'  detected 0-100 accuracy scale (max={max_acc:.2f}); '
			f'normalizing to a 0-1 fraction'
		)
		for r in runs:
			r['series'] = {
				name: [v / 100.0 for v in vals]
				for name, vals in r['series'].items()
			}

	methods = sorted({r['method'] for r in runs})
	cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
	colors = {m: cycle[i % len(cycle)] for i, m in enumerate(methods)}

	grouped: dict[str, dict[str, list[dict]]] = defaultdict(
		lambda: defaultdict(list)
	)
	for r in runs:
		grouped[r['task']][r['method']].append(r)

	metric_labels = {'top1': 'Top-1 accuracy', 'top5': 'Top-5 accuracy'}
	out_dir = Path(output_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	written = []
	for task, method_runs in sorted(grouped.items()):
		metrics_here = sorted(
			{m for rs in method_runs.values() for r in rs for m in r['series']},
			key=lambda m: list(METRICS).index(m),
		)
		for metric in metrics_here:
			plt.figure(figsize=(8, 5))
			for method in sorted(method_runs):
				rs = [r for r in method_runs[method] if metric in r['series']]
				if not rs:
					continue
				n = min(len(r['series'][metric]) for r in rs)
				xs = np.asarray(rs[0]['rounds'][:n])
				stacked = np.asarray([r['series'][metric][:n] for r in rs])
				mean, std = stacked.mean(axis=0), stacked.std(axis=0)
				plt.plot(
					xs, mean, marker='o', markersize=3, linewidth=2,
					color=colors[method],
					label=f'{method} (n={len(rs)} seed'
					      f'{"s" if len(rs) != 1 else ""})',
				)
				if np.any(std > 0):
					plt.fill_between(
						xs, mean - std, mean + std, alpha=0.15,
						color=colors[method],
					)
			label = metric_labels.get(metric, metric)
			plt.title(f'{task.upper()} — {label.lower()} by round')
			plt.xlabel('Round')
			plt.ylabel(label)
			plt.ylim(0.0, 1.0)
			plt.grid(True, alpha=0.3)
			plt.legend()
			plt.tight_layout()
			fig_path = out_dir / f'{task}_{metric}_by_round.png'
			plt.savefig(fig_path, dpi=200)
			plt.close()
			written.append(fig_path)
			print(f'  wrote {fig_path}')

	# Final-round mean+/-std table (exact numbers for the deck).
	print('\nFinal-round accuracy (mean +/- std across seeds), %:')
	for task, method_runs in sorted(grouped.items()):
		print(f'  [{task}]')
		for method in sorted(method_runs):
			rs = method_runs[method]
			parts = []
			for metric in ('top1', 'top5'):
				finals = [
					r['series'][metric][-1] * 100.0
					for r in rs
					if metric in r['series']
				]
				if finals:
					arr = np.asarray(finals)
					parts.append(
						f'{metric}={arr.mean():.2f}+/-{arr.std():.2f}'
					)
			print(f'    {method:<9} n={len(rs)}  ' + '  '.join(parts))

	print(f'\nWrote {len(written)} figure(s) to {out_dir.resolve()}')
	return written


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__)
	ap.add_argument('inputs', nargs='*', default=['results'])
	ap.add_argument('--output-dir', default='figures')
	args = ap.parse_args()
	plot(args.inputs, args.output_dir)


if __name__ == '__main__':
	main()
