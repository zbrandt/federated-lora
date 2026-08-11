from __future__ import annotations

from dataclasses import dataclass

import torch
from opacus import PrivacyEngine
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from transformers import PreTrainedModel

from federated_lora.data import cycle
from federated_lora.method import Method
from federated_lora.model import (
	clear_gradients,
	get_trainable_state,
)


class Client:
	def __init__(
		self,
		id: int,
		model: PreTrainedModel,
		dataloader: DataLoader,
		optimizer: Optimizer,
		privacy_engine: PrivacyEngine,
		steps: int,
		device: torch.device,
	) -> None:
		self.id = id
		self.model = model
		self.dataloader = dataloader
		self.optimizer = optimizer
		self.privacy_engine = privacy_engine
		self.steps = steps
		self.device = device

	def local_update(
		self,
		round: int,
		method: Method,
	) -> None:
		""" """
		self.model = self.model.to(self.device)
		self.model.train()

		batches = cycle(self.dataloader)

		total_loss = 0.0
		for _ in range(self.steps):
			batch = next(batches)
			batch = {
				key: value.to(self.device) for key, value in batch.items()
			}

			self.optimizer.zero_grad(set_to_none=True)
			out = self.model(**batch)
			loss = out.loss
			loss.backward()

			method.local_update(model=self.model, round=round)

			self.optimizer.step()
			clear_gradients(self.model)

			total_loss += float(loss.item())

		upload = get_trainable_state(self.model)

		self.model = self.model.to('cpu')

		return upload, total_loss / self.steps
