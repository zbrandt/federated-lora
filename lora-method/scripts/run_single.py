#!/usr/bin/env python
"""Run one centralised training job.

Examples
--------
# non-private baseline
python -m scripts.run_single --rank 8

# DP at epsilon=3
python -m scripts.run_single --rank 8 --epsilon 3 --batch_size 512

# SNR DIAGNOSTIC: full DP pipeline, zero noise. Answers the binary question
# "is the pipeline wired correctly, or is the SNR just too low?".
# Keep it cheap -- a subset and one epoch is enough to see it learn.
python -m scripts.run_single --noise_multiplier 0.0 --n_train 5000 --epochs 1
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config
from src.engine import run_single
from src.utils import append_result, free_memory, save_json, setup_logging


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Single-client DP-LoRA run")

    p.add_argument("--model_name", default="roberta-base")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=int, default=None, help="defaults to rank")
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--no_simple_head", action="store_true",
                   help="keep RoBERTa's full 592k-param head (dominates the DP noise budget)")

    p.add_argument("--dataset_name", default="sst2", choices=["sst2", "qnli", "qqp", "mnli"])
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--n_train", type=int, default=None)
    p.add_argument("--n_val", type=int, default=None)

    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=512, help="logical batch")
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--max_physical_batch_size", type=int, default=16)

    p.add_argument("--epsilon", type=float, default=None)
    p.add_argument("--delta", type=float, default=1e-5)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--noise_multiplier", type=float, default=None,
                   help="set 0.0 for the SNR diagnostic")

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--out_dir", default="results")
    p.add_argument("--run_name", default="")
    p.add_argument("--log_level", default="INFO")
    return p


def config_from_args(args) -> Config:
    return Config(
        model_name=args.model_name,
        rank=args.rank,
        alpha=args.alpha,
        lora_dropout=args.lora_dropout,
        simple_head=not args.no_simple_head,
        dataset_name=args.dataset_name,
        max_length=args.max_length,
        n_train=args.n_train,
        n_val=args.n_val,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        max_physical_batch_size=args.max_physical_batch_size,
        epsilon=args.epsilon,
        delta=args.delta,
        max_grad_norm=args.max_grad_norm,
        noise_multiplier=args.noise_multiplier,
        seed=args.seed,
        device=args.device,
        out_dir=args.out_dir,
        run_name=args.run_name,
    )


def main():
    args = build_parser().parse_args()
    cfg = config_from_args(args)

    name = cfg.run_name or f"r{cfg.rank}_eps{cfg.epsilon}_s{cfg.seed}"
    log = setup_logging(args.log_level, log_file=f"{cfg.out_dir}/logs/{name}.log")
    log.info("Config: %s", cfg)

    out = run_single(cfg)
    free_memory(out.pop("model"))

    path = append_result(out["row"], cfg.out_dir)
    save_json(out["history"], f"{cfg.out_dir}/history/{name}.json")

    log.info("accuracy=%.4f  eps_spent=%s", out["row"]["accuracy"], out["row"]["epsilon_spent"])
    log.info("appended -> %s", path)


if __name__ == "__main__":
    main()
