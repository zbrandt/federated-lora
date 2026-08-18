from __future__ import annotations

import torch
import torch.nn as nn
from opacus.optimizers.optimizer import DPOptimizer

from federated_lora.privacy import privatize, zero_grad
from federated_lora.server import Server


class FFALoRA:
	name = 'ffa_lora'

	def set_target_modules(self, model: nn.Module) -> None:
		# fix the randomly initialized non-zero matrices (matrix A) and only
		# fine-tune the zero-initialized matrices (matrix B)
		for name, parameter in model.named_parameters():
			if 'lora_B' in name or 'classifier' in name:
				parameter.requires_grad = True
			else:
				parameter.requires_grad = False

	def step(
		self,
		model: nn.Module,
		optimizer: DPOptimizer,
		round: int,
		step: int,
		batch_size: int,
	) -> None:
		params = privatize(model, optimizer, batch_size)
		if params is None:
			return

		optimizer.original_optimizer.step()

		zero_grad(params)
		optimizer.zero_grad()

	def aggregate(
		self,
		uploads: list[dict[str, torch.Tensor]],
		num_examples: list[int],
	) -> dict[str, torch.Tensor]:
		return Server.fedavg(uploads, num_examples)
