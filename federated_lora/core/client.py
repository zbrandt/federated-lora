from __future__ import annotations

from dataclasses import dataclass

import torch
from datasets import Dataset
from torch.utils.data import DataLoader


@dataclass(slots=True)
class ClientResult:
	state_dict: dict[str, torch.Tensor]
	n_examples: int
	average_loss: float


class Client:
	def __init__(
		self,
		id: int,
		train_shard: Dataset,
		train_dataloader: DataLoader,
	) -> None:
		self.id = id
		self.train_shard = train_shard
		self.train_dataloader = train_dataloader
