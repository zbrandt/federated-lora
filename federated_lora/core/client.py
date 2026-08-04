from __future__ import annotations

from dataclasses import dataclass

import torch
from datasets import Dataset
from torch.optim import Optimizer
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
		dataset: Dataset,
		steps: int,
		dataloader: DataLoader,
		optimizer: Optimizer,
		noise_multiplier: float,
	) -> None:
		self.id = id
		self.dataset = dataset
		self.steps = steps
		self.dataloader = dataloader
		self.optimizer = optimizer
		self.noise_multiplier = noise_multiplier
