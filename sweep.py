#!/usr/bin/env python
"""
sweep.py — drive DP federated-LoRA experiments across GPU nodes.

Every experiment is its own subprocess (GPU memory freed between runs).

Two modes:

  compare  — homogeneous DP: every client shares one epsilon. One epsilon
             per node, 4 nodes total. Eps is encoded in --output-dir
             (results/sweep/dp_compare/eps<e>/) since experiment.py's own
             output filename ({method}_{task}_seed{seed}.json) doesn't
             include it, which would otherwise let different-epsilon runs of
             the same method/seed overwrite each other. Logs live under
             logs/sweep/dp_compare/.

  hetero   — heterogeneous DP: each of the 4 clients gets its own epsilon
             (CLIENT_EPSILONS below), so every client has a different
             privacy guarantee within the same run. Only one cell per
             method (no --eps axis), so this needs just one node. Results
             -> results/sweep/dp_hetero/, logs -> logs/sweep/dp_hetero/.

    tmux new-session -d -s run -c ~/federated-lora '.venv/bin/python sweep.py compare --eps 0.5; exec bash'
    tmux new-session -d -s run -c ~/federated-lora '.venv/bin/python sweep.py compare --eps 1;   exec bash'
    tmux new-session -d -s run -c ~/federated-lora '.venv/bin/python sweep.py compare --eps 3;   exec bash'
    tmux new-session -d -s run -c ~/federated-lora '.venv/bin/python sweep.py compare --eps 8;   exec bash'
    tmux new-session -d -s run -c ~/federated-lora '.venv/bin/python sweep.py hetero; exec bash'

Pass --rounds for a fast smoke test across all methods before committing a
node to the full run, e.g. `sweep.py compare --eps 3 --rounds 5` or
`sweep.py hetero --rounds 5`.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
PYTHON = sys.executable  # use .venv/bin/python (or .venv/Scripts/python.exe) on the nodes

BENCHMARK = 'cifar100'
METHODS = ['rblora', 'hetlora', 'flora', 'flexlora', 'ffa_lora']
NUM_CLIENTS = 4
SEED = 42

# One epsilon per client (client i gets CLIENT_EPSILONS[i]) -- same four
# values as the homogeneous --eps sweep, just distributed heterogeneously
# instead of shared, so the two sweeps are directly comparable.
CLIENT_EPSILONS = [0.5, 1.0, 3.0, 8.0]

# Config defaults (lr_a=0.25, lr_b=0.25, lr_head=0.15) are used for every
# method/epsilon cell -- no per-method LR search for this comparison.
LR_A = 0.25
LR_B = 0.25
LR_HEAD = 0.15


def run_experiment(
    method, seed, output_dir, log_path, epsilon=None, client_epsilons=None, rounds=None
):
    """Launch one experiment.py run in its own process; stream output to log_path."""
    cmd = [
        PYTHON,
        str(REPO / 'experiment.py'),
        '--task', BENCHMARK,
        '--method', method,
        '--lr-a', str(LR_A),
        '--lr-b', str(LR_B),
        '--lr-head', str(LR_HEAD),
        '--seed', str(seed),
        '--num-clients', str(NUM_CLIENTS),
        '--output-dir', output_dir,
    ]
    if client_epsilons is not None:
        cmd += ['--client-epsilons'] + [str(e) for e in client_epsilons]
    else:
        cmd += ['--epsilon', str(epsilon)]
    if rounds is not None:
        cmd += ['--rounds', str(rounds)]

    log_path.parent.mkdir(parents=True, exist_ok=True)
    tag = f'eps={epsilon}' if client_epsilons is None else f'client_eps={client_epsilons}'
    print(f'>>> {method:9} {tag} -> {log_path}', flush=True)
    with open(log_path, 'w') as log:
        result = subprocess.run(
            cmd, stdout=log, stderr=subprocess.STDOUT, cwd=REPO, check=False
        )
    ok = result.returncode == 0
    print(
        f'    {"done" if ok else "FAILED — see " + str(log_path)}', flush=True
    )
    return ok


def compare(eps, rounds=None):
    """5 methods at fixed lr, seed 42, 4 clients sharing one eps -> results/sweep/dp_compare/eps<e>/."""
    out = f'results/sweep/dp_compare/eps{eps}/'
    for method in METHODS:
        log = (
            REPO
            / 'logs'
            / 'sweep'
            / 'dp_compare'
            / f'{method}_{BENCHMARK}_eps{eps}_seed{SEED}.log'
        )
        run_experiment(method, SEED, out, log, epsilon=eps, rounds=rounds)


def hetero(rounds=None):
    """5 methods, one client_epsilons=[0.5,1,3,8] assignment -> results/sweep/dp_hetero/."""
    out = 'results/sweep/dp_hetero/'
    for method in METHODS:
        log = (
            REPO
            / 'logs'
            / 'sweep'
            / 'dp_hetero'
            / f'{method}_{BENCHMARK}_seed{SEED}.log'
        )
        run_experiment(
            method, SEED, out, log, client_epsilons=CLIENT_EPSILONS, rounds=rounds
        )


def main():
    parser = argparse.ArgumentParser(
        description='Drive DP federated-LoRA experiments.'
    )
    parser.add_argument('mode', choices=['compare', 'hetero'])
    parser.add_argument(
        '--eps',
        type=float,
        choices=[0.5, 1.0, 3.0, 8.0],
        help='privacy budget for this node (required for "compare", unused for "hetero")',
    )
    parser.add_argument(
        '--rounds',
        type=int,
        default=None,
        help='override global_rounds for a fast smoke test (default: full 100 rounds)',
    )
    args = parser.parse_args()

    if args.mode == 'compare':
        if args.eps is None:
            parser.error('--eps is required for mode "compare"')
        compare(args.eps, rounds=args.rounds)
    else:
        hetero(rounds=args.rounds)


if __name__ == '__main__':
    main()
