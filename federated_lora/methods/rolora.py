from __future__ import annotations

import torch
import torch.nn as nn
from opacus.optimizers.optimizer import DPOptimizer

from federated_lora.privacy import privatize, zero_grad
from federated_lora.model import dewindow_grad_samples
from federated_lora.server import Server


class RoLoRA:
	name = 'rolora'

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
		# freeze A and update B in an odd communication round,
		# freeze B and update A in an even communication round
		frozen = ('lora_A',) if round % 2 == 1 else ('lora_B',)
		dewindow_grad_samples(model, batch_size)
		
		if optimizer.clipping_strategy == 'flat':
			for name, p in model.named_parameters():
				if p.requires_grad and any(f in name for f in frozen):
					p.grad_sample = torch.zeros_like(p.grad_sample)
	
			if optimizer.pre_step():
				for name, p in model.named_parameters():
					if not p.requires_grad or p.grad is None:
						continue
					if any(f in name for f in frozen):
						p.grad = None
	
				optimizer.original_optimizer.step()
		else:
			if optimizer.step_hook:
				optimizer.step_hook(optimizer)

			for name, p in model.named_parameters():
				if p.requires_grad and any(f in name for f in frozen):
					p.grad_sample = None
					p.grad = None
	
			params = privatize(model, optimizer)

			optimizer.original_optimizer.step()
			zero_grad(params)
	
		optimizer.zero_grad()

	def aggregate(
		self,
		uploads: list[dict[str, torch.Tensor]],
		num_examples: list[int],
	) -> dict[str, torch.Tensor]:
		# RoLoRA aggregates uniformly (1/|C|), per Algorithm 1
		return Server.fedavg(uploads, [1] * len(uploads))
