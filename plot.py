from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

METRICS: dict[str, str] = {
	'top1_acc': 'Top-1 accuracy (%)',
	'top5_acc': 'Top-5 accuracy (%)',
	'train_loss': 'Train loss',
	'test_loss': 'Test loss',
}

METHODS: dict[str, str] = {
	'dp_lora': 'DP-LoRA',
	'ffa_lora': 'FFA-LoRA',
	'rolora': 'RoLoRA',
	'la_lora': 'LA-LoRA',
}


@dataclass
class Run:
	"""One finished experiment, loaded from a single result JSON file."""

	method: str
	seed: int
	config: dict
	rounds: np.ndarray
	metrics: dict[str, np.ndarray]


def load_run(path: Path) -> Run:
	"""
	Load a single result file into a Run.

	Parameters
	----------
	path : Path
		Path to a result JSON written by experiment.py.

	Returns
	-------
	Run
		The method, seed, config, round indices, and per-metric curves.
	"""
	payload = json.loads(path.read_text(encoding='utf-8'))
	config = payload['config']
	history = payload['history']

	rounds = np.array([entry['round_index'] for entry in history])
	metrics = {
		key: np.array([entry[key] for entry in history]) for key in METRICS
	}
	return Run(
		method=config['method'],
		seed=config['seed'],
		config=config,
		rounds=rounds,
		metrics=metrics,
	)


def load_runs(results_dir: Path) -> list[Run]:
	"""
	Load every result JSON found under ``results_dir`` (searched recursively).

	Parameters
	----------
	results_dir : Path
		Directory holding the ``<method>_<task>_seed<seed>.json`` files.

	Returns
	-------
	list[Run]
		One Run per result file.
	"""
	paths = sorted(results_dir.glob('**/*.json'))
	if not paths:
		raise SystemExit(f'no result files found under {results_dir}')
	return [load_run(path) for path in paths]


def group_by_method(runs: list[Run]) -> dict[str, list[Run]]:
	"""
	Group runs by method, ordered as in METHODS and sorted by seed.

	Parameters
	----------
	runs : list[Run]
		All loaded runs.

	Returns
	-------
	dict[str, list[Run]]
		Method id -> its runs (one per seed). Methods with no runs are
		omitted; unknown methods are ignored.
	"""
	grouped: dict[str, list[Run]] = {}
	for method in METHODS:
		seeds = sorted(
			(run for run in runs if run.method == method),
			key=lambda run: run.seed,
		)
		if seeds:
			grouped[method] = seeds
	return grouped


def seed_mean_std(
	runs: list[Run], metric: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""
	Average a metric across seeds, per round.

	Parameters
	----------
	runs : list[Run]
		The runs for a single method, one per seed.
	metric : str
		A key of METRICS.

	Returns
	-------
	tuple[np.ndarray, np.ndarray, np.ndarray]
		Round indices, the per-round mean over seeds, and the per-round
		standard deviation. Curves are aligned to the shortest run so
		unequal-length histories still stack cleanly.
	"""
	length = min(len(run.rounds) for run in runs)
	rounds = runs[0].rounds[:length]
	stacked = np.stack([run.metrics[metric][:length] for run in runs])
	return rounds, stacked.mean(axis=0), stacked.std(axis=0)


def plot_metrics(grouped: dict[str, list[Run]], subtitle: str) -> plt.Figure:
	"""
	Draw each metric against communication round, one line per method.

	Each line is the mean over seeds; the shaded band is +/- one standard
	deviation.

	Parameters
	----------
	grouped : dict[str, list[Run]]
		Method id -> its runs, as returned by group_by_method.
	subtitle : str
		A one-line description of the shared experimental setup.

	Returns
	-------
	plt.Figure
		The 2x2 figure, ready to save.
	"""
	fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
	for axis, (metric, label) in zip(axes.flat, METRICS.items()):
		for method, runs in grouped.items():
			rounds, mean, std = seed_mean_std(runs, metric)
			axis.plot(rounds, mean, linewidth=2, label=METHODS[method])
			axis.fill_between(rounds, mean - std, mean + std, alpha=0.15)
		axis.set_ylabel(label)
		axis.grid(True, alpha=0.3)

	for axis in axes[-1]:
		axis.set_xlabel('Communication round')
	axes.flat[0].legend()

	fig.suptitle(subtitle)
	fig.tight_layout()
	return fig


def describe_setup(config: dict) -> str:
	"""Build a one-line summary of the setup shared by all runs."""
	return (
		f'CIFAR-100 | {config["num_clients"]} clients | '
		f'Dirichlet $\\alpha$={config["dirichlet_alpha"]} | '
		f'$\\epsilon$={config["target_epsilon"]} | '
		f'LoRA rank {config["lora_rank"]}'
	)


def print_summary(grouped: dict[str, list[Run]]) -> None:
	"""Print final-round accuracy as mean +/- std across seeds, per method."""
	print('Final-round accuracy across seeds (mean +/- std):')
	for method, runs in grouped.items():
		top1 = np.array([run.metrics['top1_acc'][-1] for run in runs])
		top5 = np.array([run.metrics['top5_acc'][-1] for run in runs])
		print(
			f'  {METHODS[method]:<9} '
			f'top1={top1.mean():5.2f} +/- {top1.std():4.2f}   '
			f'top5={top5.mean():5.2f} +/- {top5.std():4.2f}   '
			f'(n={len(runs)})'
		)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description='Plot federated LoRA benchmark results by method.'
	)
	parser.add_argument('--results-dir', type=str, default='results')
	parser.add_argument('--output', type=str, default='figures/benchmark.png')

	return parser.parse_args()


def main(args: argparse.Namespace) -> None:
	""" """
	runs = load_runs(Path(args.results_dir))
	grouped = group_by_method(runs)

	fig = plot_metrics(grouped, subtitle=describe_setup(runs[0].config))

	output = Path(args.output)
	output.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output, dpi=200, bbox_inches='tight')

	print_summary(grouped)
	print(f'\nwrote {output}')


if __name__ == '__main__':
	args = parse_args()
	main(args)
