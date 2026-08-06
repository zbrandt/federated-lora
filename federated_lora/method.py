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
		round_index: int = 0,
	) -> tuple[Module, float]: 
		""" """
		...


	def aggregate(
		self,
		client_uploads: list[dict[str, torch.Tensor]],
		weights: list[float],
	) -> dict[str, torch.Tensor]:
		"""
		Aggregate client uploads into the update model, with each method 
		defining its own aggregation rule.
		 
		Parameters
		----------
		client_uploads : list[dict[str, torch.Tensor]]
			TODO 
		weights : list[float] 
			The per-client training-set sizes aligned with ``client_uploads`` 
			for weighted updates.

		Returns
		-------
		dict[str, torch.Tensor]
			The updated model.
		"""
		...
