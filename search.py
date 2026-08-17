#!/usr/bin/env python
"""
search.py — find the best learning rate per method WITHOUT differential privacy.

DP-free companion to sweep.py: runs each method with --epsilon 0 (no noise, no
clipping) to see at what lr each method learns best on its own. experiment.py's
--task is the benchmark name, so lr is encoded in the --output-dir
(results/search/<benchmark>/lr<lr>/); logs (tagged with lr) live under
logs/search/ and are parsed by `report`.

    tmux new-session -d -s search -c ~/federated-lora '.venv/bin/python search.py run; exec bash'
    .venv/bin/python search.py report
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
PYTHON = sys.executable  # use .venv/bin/python on the nodes

METHODS = ['dp_lora', 'ffa_lora', 'rolora', 'la_lora']
LR_GRID = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
SEED = 42
SEARCH_ROUNDS = 50

LOG_DIR = REPO / 'logs' / 'search'
RESULTS_ROOT = 'results/search'


def run_experiment(
	method, lr, benchmark, seed, output_dir, log_path, rounds=None
):
	"""Launch one no-DP (--epsilon 0) experiment.py run; log to log_path."""
	cmd = [
		PYTHON,
		str(REPO / 'experiment.py'),
		'--task',
		benchmark,
		'--method',
		method,
		'--epsilon',
		'0',
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

	log_path.parent.mkdir(parents=True, exist_ok=True)
	print(
		f'>>> {method:9} no-dp lr={lr} seed={seed} -> {log_path}', flush=True
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


def run(benchmark):
	"""4 methods x LR_GRID at --epsilon 0, seed 42, SEARCH_ROUNDS."""
	for method in METHODS:
		for lr in LR_GRID:
			out = f'{RESULTS_ROOT}/{benchmark}/lr{lr}/'
			log = LOG_DIR / f'{method}_{benchmark}_nodp_lr{lr}_seed{SEED}.log'
			run_experiment(
				method, lr, benchmark, SEED, out, log, rounds=SEARCH_ROUNDS
			)
	report()


def report(log_dir: Path = LOG_DIR):
	"""Read search logs and print (and save) the best lr per method."""
	pat = re.compile(
		r'(?P<method>.+?)_(?P<benchmark>[a-z0-9]+)_nodp_lr(?P<lr>[\d.]+)_seed\d+\.log'
	)
	scores: dict[str, list[tuple[float, float]]] = {}
	if not log_dir.is_dir():
		print(f'no logs found under {log_dir}')
		return

	for f in sorted(log_dir.glob('*.log')):
		m = pat.match(f.name)
		if not m:
			continue
		top1 = [
			float(x) for x in re.findall(r'top1_acc=([\d.]+)', f.read_text())
		]
		if not top1:
			continue
		tail = top1[-5:]
		scores.setdefault(m['method'], []).append(
			(sum(tail) / len(tail), float(m['lr']))
		)

	best: dict[str, float] = {}
	print('\n=== best LR per method (no DP) ===')
	for method in METHODS:
		cand = scores.get(method)
		if not cand:
			print(f'  {method:9} (no runs found)')
			continue
		cand.sort(reverse=True)
		best_score, best_lr = cand[0]
		by_lr = ', '.join(
			f'{lr}:{s:.1f}' for s, lr in sorted(cand, key=lambda x: x[1])
		)
		best[method] = best_lr
		print(
			f'  {method:9} best lr={best_lr:<6} last5-top1={best_score:5.2f}   ({by_lr})'
		)

	if best:
		out = Path(RESULTS_ROOT) / 'best_lr.json'
		out.parent.mkdir(parents=True, exist_ok=True)
		out.write_text(json.dumps(best, indent=2), encoding='utf-8')
		print(f'\n  -> BEST_LR (no DP) = {best}')
		print(f'  wrote {out}')


def main():
	parser = argparse.ArgumentParser(
		description='Find the best lr per method without DP.'
	)
	parser.add_argument('mode', choices=['run', 'report'])
	parser.add_argument(
		'--benchmark',
		type=str,
		default='cifar100',
		choices=['cifar100', 'sst2'],
		help='benchmark to search on (passed to experiment.py --task)',
	)
	args = parser.parse_args()

	if args.mode == 'report':
		report()
	else:
		run(args.benchmark)


if __name__ == '__main__':
	main()
