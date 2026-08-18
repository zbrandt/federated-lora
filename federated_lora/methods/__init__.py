from __future__ import annotations

from typing import Protocol

import torch
import torch.nn as nn
from opacus.optimizers.optimizer import DPOptimizer

from federated_lora.methods.dp_lora import DPLoRA
from federated_lora.methods.ffa_lora import FFALoRA
from federated_lora.methods.flora import FLoRA
from federated_lora.methods.la_lora import LALoRA
from federated_lora.methods.rolora import RoLoRA

METHODS: dict[str, type[Method]] = {
	'dp_lora': DPLoRA,
	'ffa_lora': FFALoRA,
	'rolora': RoLoRA,
	'la_lora': LALoRA,
	'flora': FLoRA,
}


class Method(Protocol):
	"""
	A federated fine-tuning method.
	"""

	name: str

	def set_target_modules(self, model: nn.Module) -> None:
		"""
		Configure which modules of the model to adapt.

		Parameters
		----------
		model : nn.Module
			The PEFT model.
		"""
		...

	def step(
		self,
		model: nn.Module,
		optimizer: DPOptimizer,
		round: int,
		step: int,
		batch_size: int,
	) -> None:
		"""
		Perform a single optimization step to update trainable parameters.

		Parameters
		----------
		model : nn.Module
			The local PEFT model.
		optimizer : DPOptimizer
			The wrapper that adds additional functionality to clip per sample
			gradients and add Gaussian noise.
		round : int
			The current global communication round.
		step : int
			The current local step.
		batch_size : int
			The number of training examples processed before parameters are
			updated.
		"""
		...

	def aggregate(
		self,
		uploads: list[dict[str, torch.Tensor]],
		num_examples: list[int],
	) -> dict[str, torch.Tensor]:
		"""
		Aggregate client trainable parameter state-dict updates to the model.

		Parameters
		----------
		uploads : list[dict[str, torch.Tensor]]
			The client trainable parameter state-dict updates to the model.
		num_examples : list[int]
			The total number of examples of each client.

		Returns
		-------
		dict[str, torch.Tensor]
			The aggregated client updates to the model.
		"""
		...


__all__ = ['Method']
