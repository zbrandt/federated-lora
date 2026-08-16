#!/usr/bin/env python
"""
sweep.py — drive federated-LoRA experiments on the GPU nodes.

Replaces the old shell scripts (sweep.sh / lr_search.sh / seed_sweep.sh) and
pick_best_lr.py with one readable file. Every experiment is launched as its OWN
process (a subprocess call to experiment.py) so GPU memory is released between
runs. Learning rate is ALWAYS passed explicitly, so runs never fall back on the
argparse default (0.20) — which disagrees with config.py's 0.02 and silently
determined the lr of earlier sweeps.

Run ONE epsilon per node (sim12=3, sim13=2, sim14=1) inside tmux. Launch with
the venv's python so experiment.py inherits it:

    tmux new-session -d -s run '.venv/bin/python sweep.py compare    --eps 3; exec bash'
    tmux new-session -d -s run '.venv/bin/python sweep.py lr-search  --eps 3; exec bash'
    tmux new-session -d -s run '.venv/bin/python sweep.py seed-sweep --eps 3; exec bash'
    .venv/bin/python sweep.py pick-lr        # after lr-search finishes on all nodes

Modes:
    compare     the 4 methods at COMPARE_LR, seed 42, full rounds  -> outputs/
    lr-search   the 4 methods x LR_GRID, seed 42, SEARCH_ROUNDS    -> outputs_lrsearch/
    pick-lr     read logs_lrsearch/, print the best lr per (method, eps)
    seed-sweep  each method at BEST_LR[eps], SEEDS, full rounds    -> outputs_final/
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
PYTHON = sys.executable  # interpreter running this script; use .venv/bin/python on the nodes

# The methods to compare (FLoRA is intentionally excluded).
METHODS = ['dp_lora', 'ffa_lora', 'rolora', 'la_lora']

# lr grid for `lr-search`: the paper's {0.01, 0.02, 0.1, 0.2} (Section 6.1) plus
# 0.005 (for fragile methods at strict eps) and 0.05 (a mid point).
LR_GRID = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]

# lr for the simple `compare` mode (a single-lr, 4-method run per epsilon).
COMPARE_LR = 0.02

# seeds for the final `seed-sweep`.
SEEDS = [42, 43, 44]

# shorter runs for the lr search (divergence + lr ranking are clear before 100).
SEARCH_ROUNDS = 50

# Best lr per epsilon per method. FILL IN from `pick-lr` output before seed-sweep.
BEST_LR = {
	3: {'dp_lora': 0.02, 'ffa_lora': 0.02, 'rolora': 0.02, 'la_lora': 0.02},
	2: {'dp_lora': 0.02, 'ffa_lora': 0.02, 'rolora': 0.02, 'la_lora': 0.02},
	1: {'dp_lora': 0.02, 'ffa_lora': 0.02, 'rolora': 0.02, 'la_lora': 0.02},
}


def run_experiment(method, epsilon, lr, seed, task, output_dir, log_path, rounds=None):
	"""Launch one experiment.py run in its own process; stream its output to log_path."""
	cmd = [
		PYTHON, str(REPO / 'experiment.py'),
		'--task', task,
		'--method', method,
		'--epsilon', str(epsilon),
		'--lr-a', str(lr), '--lr-b', str(lr), '--lr-head', str(lr),
		'--seed', str(seed),
		'--output-dir', output_dir,
	]
	if rounds is not None:
		cmd += ['--global-rounds', str(rounds)]

	log_path.parent.mkdir(parents=True, exist_ok=True)
	print(f'>>> {method:9} eps={epsilon} lr={lr} seed={seed} -> {log_path}', flush=True)
	with open(log_path, 'w') as log:
		result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=REPO)
	ok = result.returncode == 0
	print(f'    {"done" if ok else "FAILED — see " + str(log_path)}', flush=True)
	return ok


def compare(eps):
	"""The 4 methods at COMPARE_LR, seed 42, full rounds -> outputs/."""
	task = f'cifar100_eps{eps}'
	for method in METHODS:
		log = REPO / 'logs' / f'{method}_{task}_seed42.log'
		run_experiment(method, eps, COMPARE_LR, 42, task, 'outputs/', log)


def lr_search(eps):
	"""The 4 methods x LR_GRID, seed 42, SEARCH_ROUNDS -> outputs_lrsearch/."""
	for method in METHODS:
		for lr in LR_GRID:
			task = f'cifar100_eps{eps}_lr{lr}'
			log = REPO / 'logs_lrsearch' / f'{method}_{task}_seed42.log'
			run_experiment(method, eps, lr, 42, task, 'outputs_lrsearch/', log,
			               rounds=SEARCH_ROUNDS)


def seed_sweep(eps):
	"""Each method at BEST_LR[eps], across SEEDS, full rounds -> outputs_final/."""
	# task omits lr so plot.py groups the seeds of each method together.
	task = f'cifar100_eps{eps}'
	for method in METHODS:
		lr = BEST_LR[eps][method]
		for seed in SEEDS:
			log = REPO / 'logs_final' / f'{method}_{task}_lr{lr}_seed{seed}.log'
			run_experiment(method, eps, lr, seed, task, 'outputs_final/', log)


def pick_lr(log_dir='logs_lrsearch'):
	"""Read lr-search logs and print the best lr per (method, epsilon)."""
	pat = re.compile(
		r'(?P<method>.+?)_cifar100_eps(?P<eps>\d+)_lr(?P<lr>[\d.]+)_seed\d+\.log'
	)
	scores = {}  # (method, eps) -> list of (last5_top1, lr)
	for f in sorted(Path(log_dir).glob('*.log')):
		m = pat.match(f.name)
		if not m:
			continue
		top1 = [float(x) for x in re.findall(r'top1_acc=([\d.]+)', f.read_text())]
		if not top1:
			continue
		tail = top1[-5:]
		key = (m['method'], m['eps'])
		scores.setdefault(key, []).append((sum(tail) / len(tail), float(m['lr'])))

	for eps in ['3', '2', '1']:
		print(f'\n=== epsilon = {eps} ===')
		best_pairs = []
		for method in METHODS:
			cand = scores.get((method, eps))
			if not cand:
				print(f'  {method:9} (no runs found)')
				continue
			cand.sort(reverse=True)  # highest last5-top1 first
			best_score, best_lr = cand[0]
			by_lr = ', '.join(f'{lr}:{s:.1f}' for s, lr in sorted(cand, key=lambda x: x[1]))
			print(f'  {method:9} best lr={best_lr:<6} last5-top1={best_score:5.2f}   ({by_lr})')
			best_pairs.append(f"'{method}': {best_lr}")
		if best_pairs:
			print(f'  -> BEST_LR[{eps}] = {{{", ".join(best_pairs)}}}')


def main():
	parser = argparse.ArgumentParser(description='Drive federated-LoRA experiments.')
	parser.add_argument('mode', choices=['compare', 'lr-search', 'seed-sweep', 'pick-lr'])
	parser.add_argument('--eps', type=int, choices=[1, 2, 3],
	                    help='privacy budget (required for every mode except pick-lr)')
	args = parser.parse_args()

	if args.mode == 'pick-lr':
		pick_lr()
		return
	if args.eps is None:
		parser.error(f'--eps is required for mode "{args.mode}"')
	{'compare': compare, 'lr-search': lr_search, 'seed-sweep': seed_sweep}[args.mode](args.eps)


if __name__ == '__main__':
	main()
