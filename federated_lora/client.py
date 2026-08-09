from __future__ import annotations

from dataclasses import dataclass

import torch
from datasets import Dataset
from opacus import PrivacyEngine
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from transformers import PreTrainedModel

from federated_lora.data import cycle
from federated_lora.method import Method


@dataclass(slots=True)
class ClientResult:
	state_dict: dict[str, torch.Tensor]
	n_examples: int
	average_loss: float


class Client:
	def __init__(
		self,
		id: int,
		model: PreTrainedModel,
		dataset: Dataset,
		steps: int,
		dataloader: DataLoader,
		optimizer: Optimizer,
		noise_multiplier: float,
		privacy_engine: PrivacyEngine,
		device: torch.device
	) -> None:
		self.id = id
		self.model = model
		self.dataset = dataset
		self.steps = steps
		self.dataloader = dataloader
		self.optimizer = optimizer
		self.noise_multiplier = noise_multiplier
		self.privacy_engine = privacy_engine
		self.device = device

	def update(
		self,
		round: int,
		method: Method,
	) -> None:
		""" """
		self.model = self.model.to(self.device)
		self.model.train()

		batches = cycle(self.dataloader)

		total_loss = 0.0
		for _i in range(self.steps):
			batch = next(batches)
			batch = {
				key: value.to(self.device) for key, value in batch.items()
			}

			self.optimizer.zero_grad(set_to_none=True)
			out = self.model(**batch)
			loss = out.loss
			loss.backward()

			# TODO: mask LoRA gradients according to selected method, rename
			# method and / or define return value?
			method.local_update(model=self.model, round=round)

			self.optimizer.step()

			# TODO: clear grad samples, move to model.py?
			for p in self.model.parameters():
				if p.grad is not None:
					p.grad = None
				if hasattr(p, 'grad_sample'):
					p.grad_sample = None

			total_loss += float(loss.item())

		# TODO: move to model.py?
		trainable_parameters = {
			name
			for name, param in self.model.named_parameters()
			if param.requires_grad
		}
		upload = {
			key: value.detach().cpu().clone()
			for key, value in self.model.state_dict().items()
			if key in trainable_parameters
		}

		self.model = self.model.to('cpu')

		return upload, total_loss / self.steps
