from __future__ import annotations

import torch
import torch.nn as nn
from opacus.optimizers.optimizer import DPOptimizer

from federated_lora.model import dewindow_grad_samples
from federated_lora.privacy import privatize, zero_grad
from federated_lora.server import Server


class DPLoRA:
	name = 'dp_lora'

	def set_target_modules(self, model: nn.Module) -> None:
		for name, parameter in model.named_parameters():
			if 'lora_' in name or 'classifier' in name:
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
		dewindow_grad_samples(model, batch_size)

		if optimizer.clipping_strategy == 'flat':
			if optimizer.pre_step():
				optimizer.original_optimizer.step()
		else:
			if optimizer.step_hook:
				optimizer.step_hook(optimizer)

			params = privatize(model, optimizer)

			optimizer.original_optimizer.step()
			zero_grad(params)

		optimizer.zero_grad()

	def aggregate(
		self,
		uploads: list[dict[str, torch.Tensor]],
		num_examples: list[int],
	) -> dict[str, torch.Tensor]:
		# DP-LoRA weights clients by data size (paper Eq 4)
		return Server.fedavg(uploads, num_examples)
