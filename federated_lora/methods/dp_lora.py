from __future__ import annotations

import torch
import torch.nn as nn
from opacus.optimizers.optimizer import DPOptimizer

from federated_lora.server import Server
from federated_lora.privacy import (
	get_trainable_with_grad_sample,
	clip_and_accumulate,
	add_noise,
	scale_grad,
	zero_grad
)


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
		noise_multiplier: float
	) -> None:
		params = get_trainable_with_grad_sample(model)
		thresholds = clip_and_accumulate(params)

		add_noise(thresholds, noise_multiplier)
		scale_grad(params, optimizer.expected_batch_size, 1)
		zero_grad(params)

		optimizer.original_optimizer.step()
		optimizer.zero_grad()

	def aggregate(
		self,
		uploads: list[dict[str, torch.Tensor]],
		num_examples: list[int],
	) -> dict[str, torch.Tensor]:
		return Server.fedavg(uploads, num_examples)
