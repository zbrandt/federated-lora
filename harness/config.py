from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


FederatedMethod = Literal["fedit", "ffa"]
PartitionStrategy = Literal["iid", "noniid"]


@dataclass(slots=True)
class Config:
    # Model and dataset defaults mirror the current single-script prototype.
    model_name: str = "gpt2"
    dataset_name: str = "Salesforce/wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    train_split: str = "train[:1%]"
    eval_split: str = "validation[:1%]"
    text_field: str = "text"

    # Federated training shape.
    num_clients: int = 4
    partition_strategy: PartitionStrategy = "iid"
    dirichlet_alpha: float = 0.5
    rounds: int = 2
    local_epochs: int = 1
    batch_size: int = 2
    learning_rate: float = 5e-4
    max_length: int = 256
    seed: int = 42

    # LoRA adapter settings.
    lora_rank: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: tuple[str, ...] = ("c_attn",)

    # "fedit" averages A and B; "ffa" trains only B on the client.
    method: FederatedMethod = "fedit"
