#!/usr/bin/env python
"""Rank x epsilon x seed sweep -- the Phase 1 Step 3 experiment.

This produces the motivating figure for the whole project: does the optimal
rank shift as the privacy budget tightens? Existing work reports this only at
a single global rank/epsilon, and disagrees on the direction:
  - FedSA-LoRA-DP: lower rank is more DP-robust (noise spread over a smaller
    subspace)
  - LA-LoRA appendix: higher rank helps (more capacity to absorb noise)
  - FedASK: the optimum shifts to INTERMEDIATE ranks once DP is on
Resolving this on a controlled sweep is the point.

Resumable: rows already present in results.csv are skipped, so a job killed by
a wall-clock limit can simply be relaunched.

Examples
--------
python -m scripts.run_sweep --ranks 4 8 16 32 --epsilons 1 3 inf --seeds 0 1 2
python -m scripts.run_sweep --ranks 4 8 --epsilons 3 --seeds 0 --n_train 5000 --epochs 1
"""

import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config
from src.engine import run_single
from src.utils import (
    already_done,
    append_result,
    free_memory,
    get_logger,
    load_results,
    save_json,
    setup_logging,
)


def parse_epsilon(tok: str):
    """'inf' / 'none' -> None (non-private), else float."""
    if tok.lower() in ("inf", "none", "np", "nonprivate"):
        return None
    return float(tok)


def build_parser():
    p = argparse.ArgumentParser(description="rank x epsilon x seed sweep")
    p.add_argument("--ranks", type=int, nargs="+", default=[4, 8, 16, 32])
    p.add_argument("--epsilons", type=str, nargs="+", default=["1", "3", "inf"],
                   help="use 'inf' for the non-private baseline")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])

    p.add_argument("--dataset_name", default="sst2")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--max_physical_batch_size", type=int, default=16)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--delta", type=float, default=1e-5)
    p.add_argument("--n_train", type=int, default=None)
    p.add_argument("--no_simple_head", action="store_true")

    p.add_argument("--out_dir", default="results")
    p.add_argument("--results_file", default="results.csv")
    p.add_argument("--log_level", default="INFO")
    p.add_argument("--force", action="store_true", help="re-run rows already in the CSV")
    return p


def main():
    args = build_parser().parse_args()
    log = setup_logging(args.log_level, log_file=f"{args.out_dir}/logs/sweep.log")

    epsilons = [parse_epsilon(e) for e in args.epsilons]
    grid = list(itertools.product(args.ranks, epsilons, args.seeds))
    log.info("Sweep: %d configurations", len(grid))

    done = load_results(args.out_dir, args.results_file)

    for i, (rank, eps, seed) in enumerate(grid, 1):
        key = {"rank": rank, "epsilon": eps, "seed": seed}
        if not args.force and already_done(done, key):
            log.info("[%d/%d] skip (already done) %s", i, len(grid), key)
            continue

        log.info("[%d/%d] rank=%s eps=%s seed=%s", i, len(grid), rank, eps, seed)

        cfg = Config(
            rank=rank,
            epsilon=eps,
            seed=seed,
            dataset_name=args.dataset_name,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_physical_batch_size=args.max_physical_batch_size,
            max_grad_norm=args.max_grad_norm,
            delta=args.delta,
            n_train=args.n_train,
            simple_head=not args.no_simple_head,
            out_dir=args.out_dir,
        )

        try:
            out = run_single(cfg)
            free_memory(out.pop("model"))
            append_result(out["row"], args.out_dir, args.results_file)
            save_json(out["history"], f"{args.out_dir}/history/r{rank}_eps{eps}_s{seed}.json")
            log.info("   -> accuracy %.4f", out["row"]["accuracy"])
        except RuntimeError as e:
            # Usually CUDA OOM. Record it and keep the sweep alive rather than
            # losing every remaining configuration.
            log.error("   FAILED: %s", e)
            append_result({**cfg.to_flat_dict(), "accuracy": None, "error": str(e)[:200]},
                          args.out_dir, args.results_file)
            free_memory()

        done = load_results(args.out_dir, args.results_file)

    log.info("Sweep complete -> %s/%s", args.out_dir, args.results_file)


if __name__ == "__main__":
    main()
