from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from flora.config import Config
from flora.client import Client
from flora.data import load_datasets
from flora.method import FLoRA
from flora.server import Server


def _resolve_client_epsilons(config: Config) -> list[float | None]:
    if config.client_epsilons is None:
        return [None] * config.num_clients

    if len(config.client_epsilons) != config.num_clients:
        raise ValueError(
            f"config.client_epsilons has {len(config.client_epsilons)} entries, expected {config.num_clients}"
        )
    return list(config.client_epsilons)


def build(config: Config) -> Server:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = torch.Generator().manual_seed(config.seed)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    train_shards, eval_dataset = load_datasets(config, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(config.model_name, num_labels=config.num_labels).to(device)
    method = FLoRA(config)
    method.set_target_modules(model)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    client_epsilons = _resolve_client_epsilons(config)

    clients = [
        Client(i, shard, config, device, collator, method=method, target_epsilon=client_epsilons[i])
        for i, shard in enumerate(train_shards)
    ]

    return Server(config, model, clients, eval_dataset, tokenizer, device, generator, method=method)


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
