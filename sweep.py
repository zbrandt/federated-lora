from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
PYTHON = sys.executable

TASKS = ['cifar100', 'tinyimagenet']
MODELS = ['vit', 'swin']
METHODS = ['dp_lora', 'ffa_lora', 'rolora', 'la_lora']
EPSILONS = [3, 2, 1]
SEEDS = [42]

LR = {
	3: {'dp_lora': 0.2, 'ffa_lora': 0.2, 'rolora': 0.2, 'la_lora': 0.2},
	2: {'dp_lora': 0.2, 'ffa_lora': 0.2, 'rolora': 0.2, 'la_lora': 0.2},
	1: {'dp_lora': 0.1, 'ffa_lora': 0.2, 'rolora': 0.1, 'la_lora': 0.2},
}


def run_experiment(
	task: str,
	model: str,
	method: str,
	epsilon: int,
	lr: float,
	seed: int,
) -> bool:
	"""
	Launch a single experiment as a subprocess.

	Parameters
	----------
	task : str
		The dataset (``'cifar100'`` or ``'tinyimagenet'``).
	model : str
		The backbone (``'vit'`` or ``'swin'``).
	method : str
		The federated fine-tuning method.
	epsilon : int
		The target privacy loss budget.
	lr : float
		The learning rate used for matrix A, matrix B, and the head.
	seed : int
		The starting number used to initialize a random seed generator.

	Returns
	-------
	bool
		Whether the run exited successfully.
	"""
	output_dir = f'results/sweep/{task}/{model}'
	log_path = (
		REPO
		/ 'logs'
		/ 'sweep'
		/ task
		/ model
		/ f'{method}_eps{epsilon}_lr{lr}_seed{seed}.log'
	)
	log_path.parent.mkdir(parents=True, exist_ok=True)

	cmd = [
		PYTHON,
		str(REPO / 'experiment.py'),
		'--task',
		task,
		'--model',
		model,
		'--method',
		method,
		'--epsilon',
		str(epsilon),
		'--lr_head',
		str(lr),
		'--lr_a',
		str(lr),
		'--lr_b',
		str(lr),
		'--seed',
		str(seed),
		'--output_dir',
		output_dir,
	]

	print(
		f'{task}/{model} {method} eps={epsilon} lr={lr} seed={seed} '
		f'-> {log_path}',
		flush=True,
	)

	with open(log_path, 'w') as log_file:
		result = subprocess.run(
			cmd,
			stdout=log_file,
			stderr=subprocess.STDOUT,
			cwd=REPO,
			check=False,
		)

	ok = result.returncode == 0
	print(f'{"done" if ok else f"FAILED — see {log_path}"}', flush=True)
	return ok


def sweep(
	task: str,
	model: str,
	methods: list[str],
	epsilons: list[int],
	seeds: list[int],
) -> None:
	"""
	Sweep every (seed, method, epsilon) combination for one task/model track.
	"""
	failures = []
	for seed in seeds:
		for method in methods:
			for epsilon in epsilons:
				lr = LR[epsilon][method]
				if not run_experiment(task, model, method, epsilon, lr, seed):
					failures.append(f'{method}_eps{epsilon}_seed{seed}')

	total = len(seeds) * len(methods) * len(epsilons)
	print(
		f'\n{task}/{model}: {total - len(failures)}/{total} runs succeeded',
		flush=True,
	)
	if failures:
		print(f'failed: {", ".join(failures)}', flush=True)


def main():
	parser = argparse.ArgumentParser()

	parser.add_argument('--task', type=str, required=True, choices=TASKS)
	parser.add_argument('--model', type=str, required=True, choices=MODELS)
	parser.add_argument(
		'--methods', type=str, nargs='+', default=METHODS, choices=METHODS
	)
	parser.add_argument(
		'--epsilons', type=int, nargs='+', default=EPSILONS, choices=EPSILONS
	)
	parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS)

	args = parser.parse_args()

	sweep(
		task=args.task,
		model=args.model,
		methods=args.methods,
		epsilons=args.epsilons,
		seeds=args.seeds,
	)


if __name__ == '__main__':
	main()
