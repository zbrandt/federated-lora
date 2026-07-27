#!/usr/bin/env python
"""Federated experiments (Phase 2 onward).

Examples
--------
# homogeneous baseline: 5 clients, same rank, no DP
python -m scripts.run_federated --num_clients 5 --rounds 10 --rank 8

# non-IID data split
python -m scripts.run_federated --split dirichlet --dirichlet_beta 0.5

# heterogeneous ranks (needs a rank-aware aggregator -- see src/aggregators.py)
python -m scripts.run_federated --client_ranks 4 4 8 8 16 --aggregator stack

# coupled rank + privacy heterogeneity: the actual research setting
python -m scripts.run_federated \
    --client_ranks 4 4 8 8 16 \
    --client_epsilons 0.5 0.5 3 3 8 \
    --aggregator stack_svd
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import FLConfig
from src.federated import federated_train
from src.utils import append_result, free_memory, save_json, setup_logging


def build_parser():
    p = argparse.ArgumentParser(description="Federated LoRA experiment")

    # federated
    p.add_argument("--num_clients", type=int, default=5)
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--local_epochs", type=int, default=1)
    p.add_argument("--clients_per_round", type=int, default=None)
    p.add_argument("--aggregator", default="fedavg")
    p.add_argument("--split", default="iid", choices=["iid", "dirichlet"])
    p.add_argument("--dirichlet_beta", type=float, default=0.5)

    # heterogeneity (the contribution)
    p.add_argument("--client_ranks", type=int, nargs="+", default=None,
                   help="one rank per client, e.g. --client_ranks 4 4 8 8 16")
    p.add_argument("--client_epsilons", type=float, nargs="+", default=None,
                   help="one epsilon per client, e.g. --client_epsilons 0.5 0.5 3 3 8")

    # shared
    p.add_argument("--model_name", default="roberta-base")
    p.add_argument("--rank", type=int, default=8, help="used when client_ranks is unset")
    p.add_argument("--dataset_name", default="sst2")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--epsilon", type=float, default=None)
    p.add_argument("--delta", type=float, default=1e-5)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--n_train", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--out_dir", default="results")
    p.add_argument("--run_name", default="")
    p.add_argument("--log_level", default="INFO")
    return p


def main():
    args = build_parser().parse_args()

    cfg = FLConfig(
        model_name=args.model_name,
        rank=args.rank,
        dataset_name=args.dataset_name,
        lr=args.lr,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        epsilon=args.epsilon,
        delta=args.delta,
        max_grad_norm=args.max_grad_norm,
        n_train=args.n_train,
        seed=args.seed,
        device=args.device,
        out_dir=args.out_dir,
        run_name=args.run_name,
        num_clients=args.num_clients,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        clients_per_round=args.clients_per_round,
        aggregator=args.aggregator,
        split=args.split,
        dirichlet_beta=args.dirichlet_beta,
        client_ranks=tuple(args.client_ranks) if args.client_ranks else None,
        client_epsilons=tuple(args.client_epsilons) if args.client_epsilons else None,
    )

    name = cfg.run_name or f"fl_{cfg.aggregator}_n{cfg.num_clients}_s{cfg.seed}"
    log = setup_logging(args.log_level, log_file=f"{cfg.out_dir}/logs/{name}.log")
    log.info("Config: %s", cfg)

    out = federated_train(cfg)
    free_memory(out.pop("model"))
    history = out.pop("history")

    row = {**cfg.to_flat_dict(), **out}
    row["client_ranks"] = "+".join(map(str, cfg.client_ranks)) if cfg.client_ranks else None
    row["client_epsilons"] = (
        "+".join(map(str, cfg.client_epsilons)) if cfg.client_epsilons else None
    )

    path = append_result(row, cfg.out_dir, "federated_results.csv")
    save_json(history, f"{cfg.out_dir}/history/{name}.json")

    log.info("final accuracy=%.4f", out["accuracy"])
    log.info("appended -> %s", path)


if __name__ == "__main__":
    main()
