from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from opacus import PrivacyEngine
from torch.optim import SGD
from torch.utils.data import DataLoader
from transformers import PreTrainedModel

from federated_lora.config import Config
from federated_lora.data import cycle
from federated_lora.method import Method
from federated_lora.model import (
	clear_gradients,
	get_trainable_state,
	group_trainable_parameters,
)
from federated_lora.privacy import compute_noise_level


@dataclass(slots=True)
class ClientResult:
	state_dict: dict[str, torch.Tensor]
	n_examples: int
	average_loss: float

# TODO: rewrite docstring
class Client:
	def __init__(
		self,
		id: int,
		model: PreTrainedModel,
		dataloader: DataLoader,
		config: Config,
		device: torch.device,
	) -> None:
		"""
		Build a federated client from the global model and its data loader.

		The global model (with its per-method ``requires_grad`` flags already set)
		is deep-copied, its trainable parameters are grouped into per-factor SGD
		groups, its DP noise multiplier is computed for the shard size, and the
		model/optimizer/loader are wrapped by Opacus' ``make_private``.

		Parameters
		----------
		id : int
			The client index.
		model : PreTrainedModel
			The global PEFT model to copy (trainable flags already applied).
		dataloader : DataLoader
			The client's Poisson-sampled train loader (pre ``make_private``).
		config : Config
			The run configuration (learning rates, batch size, DP budget, ...).
		device : torch.device
			The device used for local training.
		"""
		self.id = id
		self.device = device
		self.steps = config.local_steps

		client_model = copy.deepcopy(model).to('cpu')

		params_A, params_B, params_head = group_trainable_parameters(client_model)
		optimizer = SGD(
			[
				{'params': params_A, 'lr': config.lr_a},
				{'params': params_B, 'lr': config.lr_b},
				{'params': params_head, 'lr': config.lr_head},
			],
			weight_decay=0.0,
		)

		self.noise_multiplier = compute_noise_level(
			num_examples=len(dataloader.dataset),
			batch_size=config.batch_size,
			global_rounds=config.global_rounds,
			local_steps=config.local_steps,
			target_epsilon=config.target_epsilon,
			target_delta=config.target_delta,
		)

		self.privacy_engine = PrivacyEngine()
		self.model, self.optimizer, self.dataloader = (
			self.privacy_engine.make_private(
				module=client_model,
				optimizer=optimizer,
				data_loader=dataloader,
				noise_multiplier=self.noise_multiplier,
				max_grad_norm=config.clip_norm,
			)
		)

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
