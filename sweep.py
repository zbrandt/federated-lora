#!/usr/bin/env python
"""
sweep.py — drive DP federated-LoRA experiments on the GPU nodes.

Every experiment is its own subprocess (GPU memory freed between runs) and the
lr is always passed explicitly. experiment.py's --task is the benchmark name
({cifar100,sst2}) and its output filename is {method}_{task}_seed{seed}.json,
so eps/lr are encoded in the --output-dir path (results/sweep/<mode>/eps<e>/...)
to keep runs from overwriting each other. Logs (tagged with eps/lr) live under
logs/sweep/<mode>/; pick-lr parses those.

Run ONE epsilon per node (sim12=3, sim13=2, sim14=1) inside tmux:

    tmux new-session -d -s run -c ~/federated-lora '.venv/bin/python sweep.py compare    --eps 3; exec bash'
    tmux new-session -d -s run -c ~/federated-lora '.venv/bin/python sweep.py lr-search  --eps 3; exec bash'
    tmux new-session -d -s run -c ~/federated-lora '.venv/bin/python sweep.py seed-sweep --eps 3; exec bash'
    .venv/bin/python sweep.py pick-lr
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
PYTHON = sys.executable  # use .venv/bin/python on the nodes

BENCHMARK = 'cifar100'
METHODS = ['dp_lora', 'ffa_lora', 'rolora', 'la_lora']
LR_GRID = [0.01, 0.02, 0.1, 0.2]  
COMPARE_LR = 0.02
SEEDS = [42, 43, 44]
SEARCH_ROUNDS = 50

CLIPPINGS = ['median', 'flat']
CLIP_ROUNDS = 100

# Best lr per epsilon per method. FILL IN from `pick-lr` before seed-sweep.
BEST_LR = {
	3: {'dp_lora': 0.02, 'ffa_lora': 0.02, 'rolora': 0.02, 'la_lora': 0.02},
	2: {'dp_lora': 0.02, 'ffa_lora': 0.02, 'rolora': 0.02, 'la_lora': 0.02},
	1: {'dp_lora': 0.02, 'ffa_lora': 0.02, 'rolora': 0.02, 'la_lora': 0.02},
}


def run_experiment(
	method, epsilon, lr, seed, output_dir, log_path, rounds=None, clipping=None
):
	"""Launch one experiment.py run in its own process; stream output to log_path."""
	cmd = [
		PYTHON,
		str(REPO / 'experiment.py'),
		'--task',
		BENCHMARK,
		'--method',
		method,
		'--epsilon',
		str(epsilon),
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
	if rounds is not None:
		cmd += ['--rounds', str(rounds)]
	if clipping is not None:
		cmd += ['--clipping', clipping]

	log_path.parent.mkdir(parents=True, exist_ok=True)
	print(
		f'>>> {method:9} eps={epsilon} lr={lr} seed={seed} -> {log_path}',
		flush=True,
	)
	with open(log_path, 'w') as log:
		result = subprocess.run(
			cmd, stdout=log, stderr=subprocess.STDOUT, cwd=REPO, check=False
		)
	ok = result.returncode == 0
	print(
		f'    {"done" if ok else "FAILED — see " + str(log_path)}', flush=True
	)
	return ok

def compare(eps):
	"""4 methods at COMPARE_LR, seed 42, full rounds -> results/sweep/compare/eps<e>/."""
	out = f'results/sweep/compare/eps{eps}/'
	for method in METHODS:
		log = (
			REPO
			/ 'logs'
			/ 'sweep'
			/ 'compare'
			/ f'{method}_cifar100_eps{eps}_seed42.log'
		)
		run_experiment(method, eps, COMPARE_LR, 42, out, log)


def lr_search(eps):
	"""4 methods x LR_GRID, seed 42, SEARCH_ROUNDS -> results/sweep/lrsearch/eps<e>/lr<lr>/."""
	for method in METHODS:
		for lr in LR_GRID:
			out = f'results/sweep/lrsearch/eps{eps}/lr{lr}/'
			log = (
				REPO
				/ 'logs'
				/ 'sweep'
				/ 'lrsearch'
				/ f'{method}_cifar100_eps{eps}_lr{lr}_seed42.log'
			)
			run_experiment(method, eps, lr, 42, out, log, rounds=SEARCH_ROUNDS)


def seed_sweep(eps):
	"""Each method at BEST_LR[eps], across SEEDS, full rounds -> results/sweep/final/eps<e>/."""
	out = f'results/sweep/final/eps{eps}/'
	for method in METHODS:
		lr = BEST_LR[eps][method]
		for seed in SEEDS:
			log = (
				REPO
				/ 'logs'
				/ 'sweep'
				/ 'final'
				/ f'{method}_cifar100_eps{eps}_lr{lr}_seed{seed}.log'
			)
			run_experiment(method, eps, lr, seed, out, log)

def clip_sweep(eps):
	"""4 methods x LR_GRID x {median, flat} at eps, seed 42, CLIP_ROUNDS.

	-> results/sweep/clip/eps<e>/<clipping>/lr<lr>/. Same (method, eps, lr)
	run under each clipping strategy so the only difference is the clipping;
	per-round clip-threshold metrics are logged for the collapse analysis.
	"""
	for method in METHODS:
		for clipping in CLIPPINGS:
			for lr in LR_GRID:
				out = f'results/sweep/clip/eps{eps}/{clipping}/lr{lr}/'
				log = (
					REPO
					/ 'logs'
					/ 'sweep'
					/ 'clip'
					/ f'{method}_cifar100_eps{eps}_{clipping}_lr{lr}_seed42.log'
				)
				run_experiment(
					method, eps, lr, 42, out, log, rounds=CLIP_ROUNDS, clipping=clipping
				)

def pick_lr(log_dir='logs/sweep/lrsearch'):
	"""Read lr-search logs and print the best lr per (method, epsilon)."""
	pat = re.compile(
		r'(?P<method>.+?)_cifar100_eps(?P<eps>\d+)_lr(?P<lr>[\d.]+)_seed\d+\.log'
	)
	scores = {}  # (method, eps) -> list of (last5_top1, lr)
	for f in sorted(Path(log_dir).glob('*.log')):
		m = pat.match(f.name)
		if not m:
			continue
		top1 = [
			float(x) for x in re.findall(r'top1_acc=([\d.]+)', f.read_text())
		]
		if not top1:
			continue
		tail = top1[-5:]
		key = (m['method'], m['eps'])
		scores.setdefault(key, []).append(
			(sum(tail) / len(tail), float(m['lr']))
		)

	for eps in ['3', '2', '1']:
		print(f'\n=== epsilon = {eps} ===')
		best_pairs = []
		for method in METHODS:
			cand = scores.get((method, eps))
			if not cand:
				print(f'  {method:9} (no runs found)')
				continue
			cand.sort(reverse=True)
			best_score, best_lr = cand[0]
			by_lr = ', '.join(
				f'{lr}:{s:.1f}' for s, lr in sorted(cand, key=lambda x: x[1])
			)
			print(
				f'  {method:9} best lr={best_lr:<6} last5-top1={best_score:5.2f}   ({by_lr})'
			)
			best_pairs.append(f"'{method}': {best_lr}")
		if best_pairs:
			print(f'  -> BEST_LR[{eps}] = {{{", ".join(best_pairs)}}}')


def main():
	parser = argparse.ArgumentParser(
		description='Drive DP federated-LoRA experiments.'
	)
	parser.add_argument(
		'mode',
		choices=['compare', 'lr-search', 'seed-sweep', 'clip-sweep', 'pick-lr'],
	)
	parser.add_argument(
		'--eps',
		type=int,
		choices=[1, 2, 3],
		help='privacy budget (required for every mode except pick-lr)',
	)
	args = parser.parse_args()

	if args.mode == 'pick-lr':
		pick_lr()
		return
	if args.eps is None:
		parser.error(f'--eps is required for mode "{args.mode}"')
	{
		'compare': compare,
		'lr-search': lr_search,
		'seed-sweep': seed_sweep,
		'clip-sweep': clip_sweep,
	}[args.mode](args.eps)


if __name__ == '__main__':
	main()
