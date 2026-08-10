"""
Entry point for the program. Takes care of building and running the FLoRA server and clients, as well as saving the results to a specified or default directory.. 
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from flora.config import Config
from flora.client import Client
from flora.data import load_datasets
from flora.lora_layer import inject_heterogeneous_lora
from flora.server import Server

# verifies that each client rank is in the correct range and exists
def _resolve_client_ranks(config: Config) -> list[int]:
    if config.client_ranks is None:
        return [config.lora_rank] * config.num_clients

    if len(config.client_ranks) != config.num_clients:
        raise ValueError(
            f"config.client_ranks has {len(config.client_ranks)} entries, expected {config.num_clients}"
        )
    for rank in config.client_ranks:
        if not (1 <= rank <= config.lora_rank):
            raise ValueError(f"client rank {rank} must be in [1, {config.lora_rank}] (lora_rank is r_max)")
    return list(config.client_ranks)

# verifies the same thing but for epsilon
def _resolve_client_epsilons(config: Config) -> list[float | None]:
    if config.client_epsilons is None:
        return [None] * config.num_clients

    if len(config.client_epsilons) != config.num_clients:
        raise ValueError(
            f"config.client_epsilons has {len(config.client_epsilons)} entries, expected {config.num_clients}"
        )
    return list(config.client_epsilons)

# builds the FloRA server with device, random number generator, tokenizer, train sharding,
# evaluation dataset, base model, and FFA-LoRA adapter injection
# Instantiates the clients with their local datasets and, per client, its resolved LoRA rank and DP target epsilon.
# parameters: config: Config - The hyperparameter configuration from config.py.
# returns: Server - An instance of the Server class from server.py. 
def build(config: Config) -> Server:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = torch.Generator().manual_seed(config.seed)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    train_shards, eval_dataset = load_datasets(config, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(config.model_name, num_labels=config.num_labels).to(device)
    inject_heterogeneous_lora(model, config)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    client_ranks = _resolve_client_ranks(config)
    client_epsilons = _resolve_client_epsilons(config)

    clients = [
        Client(i, shard, config, device, collator, rank=client_ranks[i], target_epsilon=client_epsilons[i])
        for i, shard in enumerate(train_shards)
    ]

    return Server(config, model, clients, eval_dataset, tokenizer, device, generator)

# runs the FLoRA method, gets confirmation of the hyperparameter configuration from argument parsers, instantiates and runs the server from the configuration, saves results to specified or default result directory indexed by method name (label only), GLUE benchmark task, and seed.
# saves the results to specific or default result directory
# parameters: argv: list[str] | None - List of arguments from the user.
# returns: dict - result dictionary of configured hyperparameters as well as training and evaluation history. 
def run(argv: list[str] | None = None) -> dict:
    config = Config.from_argv(argv)
    server = build(config)
    history = [asdict(metric) for metric in server.run()]
    result = {"config": asdict(config), "history": history}

    if config.output:
        path = Path(config.output)
    else:
        path = Path(config.results_dir) / f"{config.method}_{config.dataset_task}_seed{config.seed}.json"

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[flora] wrote {path}", flush=True)
    return result
