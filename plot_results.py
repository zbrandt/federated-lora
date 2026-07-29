from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use(
	'Agg'
)  # sets rendering engine to a non-interactive, headless backend to prevent GUI pop-ups
import matplotlib.pyplot as plt


@dataclass(slots=True)
class Run:
	method: str
	task: str
	seed: int
	rounds: list[int]
	accuracy: list[float]


def _iter_json_paths(inputs: list[str] | str) -> list[Path]:
	"""
	Expand files and directories into a list of JSON paths.

	Parameters
	----------
	inputs : list[str] | str
		One or more file paths and/or directory paths as strings or Path
		objects. Only JSON files are kept.

	Returns
	-------
	list[Path]
		Flat list of JSON Path objects
	"""
	if isinstance(inputs, (str, Path)):
		inputs = [inputs]
	paths: list[Path] = []
	for raw in inputs:
		p = Path(raw)
		if p.is_dir():
			paths.extend(sorted(p.glob('*.json')))
		elif p.suffix == '.json':
			paths.append(p)
	return paths


def load_run(path: Path) -> Run:
	"""
	Loads a run from a given path and returns data as a Run object.

	Parameters
	----------
	path : Path
		A Path object to a run's JSON output.

	Returns
	-------
	Run
		A Run object with method, task, seed, rounds, and accuracy information.
	"""
	payload = json.loads(path.read_text(encoding='utf-8'))
	config = payload.get('config', {})
	history = payload.get('history', [])
	return Run(
		method=str(config.get('method', path.stem)),
		task=str(config.get('task', 'unknown')),
		seed=int(config.get('seed', 0)),
		rounds=[int(entry['round_index']) for entry in history],
		accuracy=[float(entry['accuracy']) for entry in history],
	)


def _aggregate_over_seeds(
	runs: list[Run],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""
	Mean and standard deviation of accuracy per round across given runs (seeds)
	for the same method and task.

	Runs are aligned to the shortest history so a truncated run can't misalign
	the average.

	Parameters
	----------
	runs : list[Run]
		A list of Run objects with method, task, seed, rounds, and accuracy
		information.

	Returns
	-------
	tuple[np.ndarray, np.ndarray, np.ndarray]
		Tuple of round indices, per-round mean accuracy across runs, and the
		per-round standard deviation.
	"""
	n_rounds = min(len(run.accuracy) for run in runs)
	rounds = np.asarray(runs[0].rounds[:n_rounds])
	stacked = np.asarray([run.accuracy[:n_rounds] for run in runs])
	return rounds, stacked.mean(axis=0), stacked.std(axis=0)


def _plot_task(
	task: str,
	method_runs: dict[str, list[Run]],
	colors: dict[str, str],
	output_path: Path,
) -> None:
	"""
	Draws one figure overlaying every method for a single task, with a mean line
	and a shaded standard deviation band per method, using the shared ``colors``
	map. Saves to ``output_path``.

	Parameters
	----------
	task : str
		A name of the task the method was ran on.
	method_runs : dict[str, list[Run]]
		A dictionary mapping the name of each method to its associated runs.
	colors : dict[str, str]
		A dictionary mapping method names to shared colors.
	output_path : Path
		A Path object for where the figure should be saved to.

	Returns
	-------
	None
		Saves output figure and does not return anything.
	"""
	plt.figure(figsize=(8, 5))
	for method in sorted(method_runs):
		runs = method_runs[method]
		rounds, mean, std = _aggregate_over_seeds(runs)
		color = colors[method]
		label = f'{method} (n={len(runs)} seed{"s" if len(runs) != 1 else ""})'
		plt.plot(
			rounds,
			mean,
			marker='o',
			markersize=3,
			linewidth=2,
			label=label,
			color=color,
		)
		if np.any(std > 0):
			plt.fill_between(
				rounds, mean - std, mean + std, alpha=0.15, color=color
			)

	plt.title(f'{task.upper()} — accuracy by round')
	plt.xlabel('Round')
	plt.ylabel('Accuracy')
	plt.ylim(0.0, 1.0)
	plt.grid(True, alpha=0.3)
	plt.legend()
	plt.tight_layout()
	plt.savefig(output_path, dpi=200)
	plt.close()


def plot(
	inputs: list[str] | str = 'results', output_dir: str = 'figures'
) -> list[Path]:
	"""
	Load runs from inputs and write one accuracy against round figure per task.

	Parameters
	----------
	inputs : list[str] | str
		One or more file paths and/or directory paths as strings or Path
		objects. Only JSON files are kept.
	output_dir : str
		Directory to create and write per-task figures into.

	Returns
	-------
	list[Path]
		List of written figure paths, one per task.
	"""
	paths = _iter_json_paths(inputs)
	if not paths:
		raise SystemExit(f'No result JSON files found in: {inputs}')

	runs = [load_run(p) for p in paths]

	# One stable color per method, shared across every task figure.
	methods = sorted({run.method for run in runs})
	cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
	colors = {
		method: cycle[i % len(cycle)] for i, method in enumerate(methods)
	}

	# grouped indexed by task, then method, with runs across seeds as values
	grouped: dict[str, dict[str, list[Run]]] = defaultdict(
		lambda: defaultdict(list)
	)  # lambda runs and automatically creates a new inner defaultdict as the value
	for run in runs:
		grouped[run.task][run.method].append(run)

	out_dir = Path(output_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	written: list[Path] = []
	for task, method_runs in sorted(grouped.items()):
		output_path = out_dir / f'{task}_accuracy_by_round.png'
		_plot_task(task, method_runs, colors, output_path)
		written.append(output_path)

	print(f'Wrote {len(written)} figure(s) to {out_dir.resolve()}')
	return written


def main() -> None:
	import argparse

	parser = argparse.ArgumentParser(
		description='Plot accuracy against round from result data'
	)
	parser.add_argument(
		'inputs',
		nargs='*',
		default=['results'],
		help='result files or a directory of them (defaults to results/)',
	)
	parser.add_argument(
		'--output-dir', default='figures', help='directory for the figures'
	)
	args = parser.parse_args()
	plot(args.inputs, args.output_dir)


if __name__ == '__main__':
	main()
