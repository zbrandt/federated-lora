from __future__ import annotations

import torch
from flora.client import Client

def select_clients(clients: list[Client], sample_rate: float, generator: torch.Generator) -> list[Client]:
    k = max(1, round(sample_rate * len(clients)))
    picks = torch.randperm(len(clients), generator=generator)[:k]
    return [clients[i] for i in picks.tolist()]
