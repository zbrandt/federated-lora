#!/usr/bin/env python
"""
sweep.py — drive DP federated-LoRA experiments across GPU nodes.

Every experiment is its own subprocess (GPU memory freed between runs).

Three modes:

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

  rank     — homogeneous rank sweep: every client uses the same LoRA rank
             (no client_ranks heterogeneity), epsilon held fixed at
             RANK_SWEEP_EPS. One rank per node, 4 nodes total. Results
             -> results/sweep/dp_rank/rank<r>/, logs -> logs/sweep/dp_rank/.

  grid     — homogeneous DP x homogeneous rank, crossed: every (eps, rank)
             pair from the 4x4 grid, GRID_METHODS only (no ffa_lora -- it's
             just along for the ride in the other sweeps, not a focus of the
             rank axis). One (eps, rank) cell per node, 16 nodes for full
             parallelism. Results -> results/sweep/dp_rank_grid/eps<e>_rank<r>/,
             logs -> logs/sweep/dp_rank_grid/.

  spread-grid — epsilon-spread x rank-spread, crossed: instead of every
             client sharing one epsilon/rank, each spread level splits the 4
             clients into 2 above / 2 below a fixed mean (MEAN_EPS/MEAN_RANK),
             with the gap controlled by gamma (epsilon spread level, 0..1) and
             kappa (rank spread level, 0..1) -- 0 means everyone at the mean
             (identical to the "grid" cell at that mean), 1 means maximum
             dispersion down to EPS_FLOOR/RANK_FLOOR. This isolates the effect
             of *heterogeneity itself* from the effect of the population's
             average privacy/rank budget, since the mean is held fixed at
             every spread level. GRID_METHODS only (needs rank-aware
             aggregation, same reasoning as "grid"). One (gamma, kappa) cell
             per node, 16 nodes for full parallelism. The (gamma=0, kappa=0)
             cell is identical to "grid"'s (eps=MEAN_EPS, rank=MEAN_RANK)
             cell -- no need to re-run it if you already have that data.
             Results -> results/sweep/dp_spread_grid/gamma<g>_kappa<k>/,
             logs -> logs/sweep/dp_spread_grid/.

    tmux new-session -d -s run -c ~/federated-lora '.venv/bin/python sweep.py compare --eps 0.5; exec bash'
    tmux new-session -d -s run -c ~/federated-lora '.venv/bin/python sweep.py compare --eps 1;   exec bash'
    tmux new-session -d -s run -c ~/federated-lora '.venv/bin/python sweep.py compare --eps 3;   exec bash'
    tmux new-session -d -s run -c ~/federated-lora '.venv/bin/python sweep.py compare --eps 8;   exec bash'
    tmux new-session -d -s run -c ~/federated-lora '.venv/bin/python sweep.py hetero; exec bash'
    tmux new-session -d -s run -c ~/federated-lora '.venv/bin/python sweep.py rank --rank 2;  exec bash'
    tmux new-session -d -s run -c ~/federated-lora '.venv/bin/python sweep.py rank --rank 8;  exec bash'
    tmux new-session -d -s run -c ~/federated-lora '.venv/bin/python sweep.py rank --rank 16; exec bash'
    tmux new-session -d -s run -c ~/federated-lora '.venv/bin/python sweep.py rank --rank 32; exec bash'

    # grid: one node per (eps, rank) pair, 16 total. Run
    #   python sweep.py grid --list
    # to print all 16 ready-to-paste tmux commands instead of typing them out.

    # spread-grid: one node per (gamma, kappa) pair, 16 total. Run
    #   python sweep.py spread-grid --list
    # to print all 16 ready-to-paste tmux commands instead of typing them out.

Pass --smoke for a correctness+timing check before committing a node to a
full run, e.g. `sweep.py compare --eps 3 --smoke` or `sweep.py rank --rank 8
--smoke`. --smoke shrinks both rounds AND local_steps (2 rounds x 2 local
steps x 4 clients = 16 sequential steps per method, instead of the full
run's 100 x 20 x 4 = 8000) so a real failure surfaces in seconds, not
after hours of a mostly-pointless deep run. It also caps each method at
SMOKE_TIMEOUT_S wall-clock time and kills+moves on if that's exceeded,
so a hung/thrashing node can't silently burn the rest of your smoke test
budget -- you get a TIMEOUT verdict on that one method instead of nothing.
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
# grid mode drops ffa_lora: it doesn't do anything rank-aware (plain
# Server.fedavg(), same behavior at every rank), so it's not informative on
# the rank axis and would just be 16 extra no-signal runs.
GRID_METHODS = ['rblora', 'hetlora', 'flora', 'flexlora']
EPS_VALUES = [0.5, 1.0, 3.0, 8.0]
RANK_VALUES = [2, 8, 16, 32]
NUM_CLIENTS = 4
SEED = 42

# One epsilon per client (client i gets CLIENT_EPSILONS[i]) -- same four
# values as the homogeneous --eps sweep, just distributed heterogeneously
# instead of shared, so the two sweeps are directly comparable.
CLIENT_EPSILONS = [0.5, 1.0, 3.0, 8.0]

# Epsilon held fixed for the rank sweep -- Config's own default, and the
# middle value of the --eps grid used elsewhere.
RANK_SWEEP_EPS = 3.0

# spread-grid: mean epsilon/rank held fixed across every spread level (only
# dispersion changes), and the floor each axis can't cross even at max
# spread. Chosen to match an already-run "grid" cell (eps=3.0, rank=16) so
# spread=0 on both axes reuses that data instead of re-running it.
MEAN_EPS = 3.0
EPS_FLOOR = 0.5
MEAN_RANK = 16
RANK_FLOOR = 2
SPREAD_LEVELS = [0.0, 0.33, 0.66, 1.0]

# Config defaults (lr_a=0.25, lr_b=0.25, lr_head=0.15) are used for every
# method/epsilon/rank cell -- no per-method LR search for this comparison.
LR_A = 0.25
LR_B = 0.25
LR_HEAD = 0.15

SMOKE_ROUNDS = 2
SMOKE_LOCAL_STEPS = 2
SMOKE_TIMEOUT_S = 600  # kill and move on if one method's smoke run exceeds this


def run_experiment(
    method,
    seed,
    output_dir,
    log_path,
    epsilon=None,
    client_epsilons=None,
    lora_rank=None,
    client_ranks=None,
    rounds=None,
    local_steps=None,
    timeout=None,
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
    if client_ranks is not None:
        cmd += ['--client-ranks'] + [str(r) for r in client_ranks]
    elif lora_rank is not None:
        cmd += ['--lora-rank', str(lora_rank)]
    if rounds is not None:
        cmd += ['--rounds', str(rounds)]
    if local_steps is not None:
        cmd += ['--local-steps', str(local_steps)]

    log_path.parent.mkdir(parents=True, exist_ok=True)
    tag = f'eps={epsilon}' if client_epsilons is None else f'client_eps={client_epsilons}'
    if client_ranks is not None:
        tag += f' client_ranks={client_ranks}'
    elif lora_rank is not None:
        tag += f' rank={lora_rank}'
    print(f'>>> {method:9} {tag} -> {log_path}', flush=True)
    with open(log_path, 'w') as log:
        try:
            result = subprocess.run(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=REPO,
                check=False,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            log.write(f'\n[sweep.py] killed after exceeding {timeout}s timeout\n')
            print(f'    TIMEOUT after {timeout}s — see {log_path}', flush=True)
            return False
    ok = result.returncode == 0
    print(
        f'    {"done" if ok else "FAILED — see " + str(log_path)}', flush=True
    )
    return ok


def compare(eps, smoke=False):
    """5 methods at fixed lr, seed 42, 4 clients sharing one eps -> results/sweep/dp_compare/eps<e>/."""
    out = f'results/sweep/dp_compare/eps{eps}/'
    rounds = SMOKE_ROUNDS if smoke else None
    local_steps = SMOKE_LOCAL_STEPS if smoke else None
    timeout = SMOKE_TIMEOUT_S if smoke else None
    for method in METHODS:
        log = (
            REPO
            / 'logs'
            / 'sweep'
            / 'dp_compare'
            / f'{method}_{BENCHMARK}_eps{eps}_seed{SEED}.log'
        )
        run_experiment(
            method, SEED, out, log, epsilon=eps,
            rounds=rounds, local_steps=local_steps, timeout=timeout,
        )


def hetero(smoke=False):
    """5 methods, one client_epsilons=[0.5,1,3,8] assignment -> results/sweep/dp_hetero/."""
    out = 'results/sweep/dp_hetero/'
    rounds = SMOKE_ROUNDS if smoke else None
    local_steps = SMOKE_LOCAL_STEPS if smoke else None
    timeout = SMOKE_TIMEOUT_S if smoke else None
    for method in METHODS:
        log = (
            REPO
            / 'logs'
            / 'sweep'
            / 'dp_hetero'
            / f'{method}_{BENCHMARK}_seed{SEED}.log'
        )
        run_experiment(
            method, SEED, out, log, client_epsilons=CLIENT_EPSILONS,
            rounds=rounds, local_steps=local_steps, timeout=timeout,
        )


def rank(r, smoke=False):
    """5 methods at fixed lora_rank=r, homogeneous (no client_ranks), eps=RANK_SWEEP_EPS -> results/sweep/dp_rank/rank<r>/."""
    out = f'results/sweep/dp_rank/rank{r}/'
    rounds = SMOKE_ROUNDS if smoke else None
    local_steps = SMOKE_LOCAL_STEPS if smoke else None
    timeout = SMOKE_TIMEOUT_S if smoke else None
    for method in METHODS:
        log = (
            REPO
            / 'logs'
            / 'sweep'
            / 'dp_rank'
            / f'{method}_{BENCHMARK}_rank{r}_seed{SEED}.log'
        )
        run_experiment(
            method, SEED, out, log, epsilon=RANK_SWEEP_EPS, lora_rank=r,
            rounds=rounds, local_steps=local_steps, timeout=timeout,
        )


def grid(eps, r, smoke=False):
    """GRID_METHODS at fixed (eps, rank), homogeneous on both axes -> results/sweep/dp_rank_grid/eps<e>_rank<r>/."""
    out = f'results/sweep/dp_rank_grid/eps{eps}_rank{r}/'
    rounds = SMOKE_ROUNDS if smoke else None
    local_steps = SMOKE_LOCAL_STEPS if smoke else None
    timeout = SMOKE_TIMEOUT_S if smoke else None
    for method in GRID_METHODS:
        log = (
            REPO
            / 'logs'
            / 'sweep'
            / 'dp_rank_grid'
            / f'{method}_{BENCHMARK}_eps{eps}_rank{r}_seed{SEED}.log'
        )
        run_experiment(
            method, SEED, out, log, epsilon=eps, lora_rank=r,
            rounds=rounds, local_steps=local_steps, timeout=timeout,
        )


def print_grid_commands():
    """Print all 16 ready-to-paste tmux launch lines for the full (eps, rank) grid."""
    for eps in EPS_VALUES:
        for r in RANK_VALUES:
            print(
                "tmux new-session -d -s run -c ~/federated-lora "
                f"'.venv/bin/python sweep.py grid --eps {eps} --rank {r}; exec bash'"
            )


def grid_all(smoke=False):
    """All 16 (eps, rank) cells sequentially on one node -- 16 cells x 4 methods x 100 rounds, one after another."""
    for eps in EPS_VALUES:
        for r in RANK_VALUES:
            grid(eps, r, smoke=smoke)


def make_spread(num_clients, mean, spread, floor, round_int=False):
    """2 clients at mean+delta, 2 at mean-delta; mean stays fixed, delta scales with spread (0=identical, 1=max dispersion down to floor)."""
    if spread <= 0:
        values = [mean] * num_clients
    else:
        delta = min(spread * mean, mean - floor)
        half = num_clients // 2
        values = [mean + delta] * half + [mean - delta] * (num_clients - half)
    if round_int:
        values = [max(floor, round(v)) for v in values]
    return values


def spread_grid(gamma, kappa, smoke=False):
    """GRID_METHODS at fixed (gamma, kappa) spread levels around MEAN_EPS/MEAN_RANK -> results/sweep/dp_spread_grid/gamma<g>_kappa<k>/."""
    client_epsilons = make_spread(NUM_CLIENTS, MEAN_EPS, gamma, EPS_FLOOR)
    client_ranks = make_spread(NUM_CLIENTS, MEAN_RANK, kappa, RANK_FLOOR, round_int=True)

    out = f'results/sweep/dp_spread_grid/gamma{gamma}_kappa{kappa}/'
    rounds = SMOKE_ROUNDS if smoke else None
    local_steps = SMOKE_LOCAL_STEPS if smoke else None
    timeout = SMOKE_TIMEOUT_S if smoke else None
    for method in GRID_METHODS:
        log = (
            REPO
            / 'logs'
            / 'sweep'
            / 'dp_spread_grid'
            / f'{method}_{BENCHMARK}_gamma{gamma}_kappa{kappa}_seed{SEED}.log'
        )
        run_experiment(
            method, SEED, out, log,
            client_epsilons=client_epsilons, client_ranks=client_ranks,
            rounds=rounds, local_steps=local_steps, timeout=timeout,
        )


def print_spread_grid_commands():
    """Print all 16 ready-to-paste tmux launch lines for the full (gamma, kappa) spread grid."""
    for gamma in SPREAD_LEVELS:
        for kappa in SPREAD_LEVELS:
            print(
                "tmux new-session -d -s run -c ~/federated-lora "
                f"'.venv/bin/python sweep.py spread-grid --gamma {gamma} --kappa {kappa}; exec bash'"
            )


def spread_grid_all(smoke=False):
    """All 16 (gamma, kappa) cells sequentially on one node -- 16 cells x 4 methods x 100 rounds, one after another."""
    for gamma in SPREAD_LEVELS:
        for kappa in SPREAD_LEVELS:
            spread_grid(gamma, kappa, smoke=smoke)


def main():
    parser = argparse.ArgumentParser(
        description='Drive DP federated-LoRA experiments.'
    )
    parser.add_argument('mode', choices=['compare', 'hetero', 'rank', 'grid', 'spread-grid'])
    parser.add_argument(
        '--eps',
        type=float,
        choices=EPS_VALUES,
        help='privacy budget for this node (required for "compare"/"grid", unused otherwise)',
    )
    parser.add_argument(
        '--rank',
        type=int,
        choices=RANK_VALUES,
        help='LoRA rank for this node (required for "rank"/"grid", unused otherwise)',
    )
    parser.add_argument(
        '--gamma',
        type=float,
        choices=SPREAD_LEVELS,
        help='epsilon spread level for this node, 0..1 (required for "spread-grid", unused otherwise)',
    )
    parser.add_argument(
        '--kappa',
        type=float,
        choices=SPREAD_LEVELS,
        help='rank spread level for this node, 0..1 (required for "spread-grid", unused otherwise)',
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='"grid"/"spread-grid" modes only: print all 16 ready-to-paste tmux launch commands and exit, without running anything',
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='"grid"/"spread-grid" modes only: run all 16 cells sequentially on this one node instead of just one',
    )
    parser.add_argument(
        '--smoke',
        action='store_true',
        help=(
            f'shrink to {SMOKE_ROUNDS} rounds x {SMOKE_LOCAL_STEPS} local steps '
            f'and cap each method at {SMOKE_TIMEOUT_S}s; use before committing a '
            'node to the full run'
        ),
    )
    args = parser.parse_args()

    if args.mode == 'grid' and args.list:
        print_grid_commands()
        return
    if args.mode == 'spread-grid' and args.list:
        print_spread_grid_commands()
        return

    if args.mode == 'compare':
        if args.eps is None:
            parser.error('--eps is required for mode "compare"')
        compare(args.eps, smoke=args.smoke)
    elif args.mode == 'rank':
        if args.rank is None:
            parser.error('--rank is required for mode "rank"')
        rank(args.rank, smoke=args.smoke)
    elif args.mode == 'grid':
        if args.all:
            grid_all(smoke=args.smoke)
        elif args.eps is None or args.rank is None:
            parser.error('--eps and --rank are both required for mode "grid" (or pass --all to run every cell on this node, or --list to print all commands)')
        else:
            grid(args.eps, args.rank, smoke=args.smoke)
    elif args.mode == 'spread-grid':
        if args.all:
            spread_grid_all(smoke=args.smoke)
        elif args.gamma is None or args.kappa is None:
            parser.error('--gamma and --kappa are both required for mode "spread-grid" (or pass --all to run every cell on this node, or --list to print all commands)')
        else:
            spread_grid(args.gamma, args.kappa, smoke=args.smoke)
    else:
        hetero(smoke=args.smoke)


if __name__ == '__main__':
    main()
