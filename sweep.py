from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
PYTHON = sys.executable

BENCHMARK = 'cifar100'
METHODS = ['dp_lora', 'ffa_lora', 'rolora', 'la_lora']
LR_GRID = [0.01, 0.02, 0.1, 0.2]
COMPARE_LR = 0.02
SEEDS = [42, 43, 44]

CLIPPINGS = ['median', 'flat']
CLIP_ROUNDS = 100

BEST_LR = {
	3: {'dp_lora': 0.02, 'ffa_lora': 0.02, 'rolora': 0.02, 'la_lora': 0.02},
	2: {'dp_lora': 0.02, 'ffa_lora': 0.02, 'rolora': 0.02, 'la_lora': 0.02},
	1: {'dp_lora': 0.01, 'ffa_lora': 0.02, 'rolora': 0.01, 'la_lora': 0.02},
}


def run_experiment(
	task: str,
	method: str,
	epsilon: int,
	clipping: str,
	lr: float,
	seed: int,
	output_dir: str,
	log_path: str,
):
	""" """
	cmd = [
		PYTHON,
		str(REPO / 'experiment.py'),
		'--task',
		task,
		'--method',
		method,
		'--epsilon',
		str(epsilon),
		'--clipping',
		clipping,
		'--lr-a',
		str(lr),
		'--lr-b',
		str(lr),
		'--lr-head',
		str(lr),
		'--seed',
		str(seed),
		'--output-dir',
		output_dir,
	]

	log_path.parent.mkdir(parents=True, exist_ok=True)

	print(
		f'{method} eps={epsilon} lr={lr} seed={seed} clip={clipping} -> {log_path}',
		flush=True,
	)

	with open(log_path, 'w') as log:
		result = subprocess.run(
			cmd, stdout=log, stderr=subprocess.STDOUT, cwd=REPO, check=False
		)
		ok = result.returncode == 0
		print(
			f'{"done" if ok else "FAILED — see " + str(log_path)}', flush=True
		)
	return ok


def method_sweep(epsilon: int, seed: int):
	"""Sweep over methods. Keep seed and epsilon fixed"""
	for method in METHODS:
		lr = BEST_LR[epsilon][method]
		out = 'results/sweep/swin/seed/'
		log = (
			REPO
			/ 'logs'
			/ 'sweep'
			/ 'swin'  # TODO
			/ seed
			/ f'{method}_cifar100_eps{epsilon}_lr{lr}_seed{seed}.log'
		)

		# TODO
		run_experiment(
			'cifar100', method, epsilon, 'median', lr, seed, out, log
		)


def main():
	parser = argparse.ArgumentParser()

	parser.add_argument('--mode', type=str, choices=['method'])
	parser.add_argument('--epsilon', type=int, choices=[1, 2, 3])
	parser.add_argument('--seed', type=int, choices=[42, 43, 44])

	args = parser.parse_args()

	if args.mode == 'method':
		method_sweep(args.seed, args.epsilon)


if __name__ == '__main__':
	main()
