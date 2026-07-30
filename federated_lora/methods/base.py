from __future__ import annotations

from typing import Protocol

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader


class Method(Protocol):
	name: str

	def local_update(
		self,
		model: Module,
		optimizer: Optimizer,
		train_dataloader: DataLoader,
		noise_multiplier: float,
		max_grad_norm: float,
		local_steps: int,
		device: str,
	) -> tuple[Module, float]: ...

	def aggregate(
		self, client_uploads: list[dict[str, torch.Tensor]]
	) -> dict[str, torch.Tensor]: ...
