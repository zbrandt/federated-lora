from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from config import Config
from server import Server

def parse_args() -> argparse.Namespace:
    """Parse arguments from command line interface for the harness."""

    # container for argument specifications
    parser = argparse.ArgumentParser(description="Run the minimal federated LoRA harness")

    # attache individual argument specifications to the parser
    parser.add_argument("--method", choices=["fedit", "ffa"], default="fedit")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--clients", type=int, default=4)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-split", default="train[:1%]")
    parser.add_argument("--eval-split", default="validation[:1%]")
    parser.add_argument("--output", type=str, default=None)

    # run the parser and returns extracted data in an argparse.Namespace object
    return parser.parse_args()


def build(args: argparse.Namespace) -> Config:
    """Build internal config class from client arguments for the harness."""
    return Config(
        method=args.method,
        rounds=args.rounds,
        num_clients=args.clients,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        lora_rank=args.rank,
        seed=args.seed,
        train_split=args.train_split,
        eval_split=args.eval_split,
    )


def main():
    # Wire the config into the server, run the rounds, and print JSON results.
    args = parse_args()
    config = build(args)
    server = Server(config)
    history = [asdict(metric) for metric in server.run()]

    result = {"config": asdict(config), "history": history}
    print(json.dumps(result, indent=2))

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
