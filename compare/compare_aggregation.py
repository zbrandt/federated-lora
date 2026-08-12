"""
Compare the two B-aggregation strategies (see ffalora/server/aggregate.py) under a
rank-heterogeneous client population: "naive" plain FedAvg, which dilutes a LoRA
column toward zero whenever this round's selected clients don't all cover it, vs
"rank_aware", which only averages a column over the clients whose rank covers it and
otherwise carries the previous value forward.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from ffalora.config import Config
from sweep import _run_one


# Assign the first `num_clients * frac_low` clients `low_rank`, the rest `high_rank`.
def make_heterogeneous_ranks(num_clients: int, low_rank: int, high_rank: int, frac_low: float) -> list[int]:
    n_low = round(num_clients * frac_low)
    return [low_rank] * n_low + [high_rank] * (num_clients - n_low)


# Final accuracy of a run, averaged over its last `tail` rounds.
def _final_accuracy(result: dict, tail: int) -> float:
    history = result["history"]
    window = history[-tail:] if len(history) >= tail else history
    return sum(entry["accuracy"] for entry in window) / len(window)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-clients", type=int, default=20)
    parser.add_argument("--client-sample-rate", type=float, default=0.2)
    parser.add_argument("--low-rank", type=int, default=2)
    parser.add_argument("--high-rank", type=int, default=8)
    parser.add_argument("--frac-low", type=float, default=0.5, help="fraction of clients assigned --low-rank")
    parser.add_argument("--epsilon", type=float, default=None, help="DP epsilon shared by every client; omit for no DP")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--tail", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="tiny config for a fast correctness check")
    parser.add_argument("--results-dir", default="results/aggregation_compare")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    num_clients = 4 if args.smoke else args.num_clients
    ranks = make_heterogeneous_ranks(num_clients, args.low_rank, args.high_rank, args.frac_low)
    eps_tag = "none" if args.epsilon is None else f"{args.epsilon:g}"

    accuracies: dict[str, list[float]] = defaultdict(list)

    for strategy in ("naive", "rank_aware"):
        for seed in args.seeds:
            config = Config(
                seed=seed,
                num_clients=num_clients,
                client_sample_rate=args.client_sample_rate,
                lora_rank=args.high_rank,
                client_ranks=ranks,
                client_epsilons=[args.epsilon] * num_clients if args.epsilon is not None else None,
                aggregation_strategy=strategy,
                rounds=2 if args.smoke else args.rounds,
                local_epochs=1 if args.smoke else None,
                train_split="train[:1%]" if args.smoke else "train",
            )

            tag = f"agg_compare_{strategy}_low{args.low_rank}_high{args.high_rank}_eps{eps_tag}_seed{seed}"
            result = _run_one(config, tag, results_dir, args.force)
            accuracies[strategy].append(_final_accuracy(result, args.tail))

    print(f"\nheterogeneous ranks: {ranks}")
    print(f"{'strategy':>12} | {'n seeds':>7} | {'mean acc':>8} | {'std':>6}")
    for strategy in ("naive", "rank_aware"):
        accs = accuracies[strategy]
        mean = sum(accs) / len(accs)
        std = (sum((a - mean) ** 2 for a in accs) / len(accs)) ** 0.5
        print(f"{strategy:>12} | {len(accs):>7} | {mean:>8.4f} | {std:>6.4f}")



if __name__ == "__main__":
    main()
