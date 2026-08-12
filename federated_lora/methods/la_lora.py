from __future__ import annotations

import torch
import torch.nn as nn
from opacus.optimizers.optimizer import DPOptimizer
from torch.nn import functional as F

from federated_lora.server import Server


class LALoRA:
	name = 'la_lora'

	def set_target_modules(self, model: nn.Module) -> None:
		for name, parameter in model.named_parameters():
			if 'lora_' in name or 'classifier' in name:
				parameter.requires_grad = True
			else:
				parameter.requires_grad = False

	def smooth(self, grad: torch.Tensor, mode: str) -> torch.Tensor:
		""" """
		kernel = (
			torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], device=grad.device) / 16.0
		)
		kernel = kernel.view(1, 1, 5)

		if mode == 'row':  # Matrix A: shape (r, n)
			x = grad.unsqueeze(1)  # Shape: (r, 1, n)
			x_padded = F.pad(x, (2, 2), mode='reflect')
			filtered = F.conv1d(x_padded, kernel).squeeze(1)
			return filtered
		elif mode == 'col':  # Matrix B: shape (m, r)
			x = grad.T.unsqueeze(1)  # Transpose to (r, 1, m)
			x_padded = F.pad(x, (2, 2), mode='reflect')
			filtered = F.conv1d(x_padded, kernel).squeeze(1).T
			return filtered

	def step(
		self, model: nn.Module, optimizer: DPOptimizer, round: int, step: int
	) -> None:
		""" """
		if optimizer.pre_step():
			for name, parameter in model.named_parameters():
				if parameter.grad is None:
					continue
				if 'lora_A' in name:
					if step % 2 == 1:
						parameter.grad = self.smooth(
							parameter.grad, mode='row'
						)
					else:
						parameter.grad.zero_()
				elif 'lora_B' in name:
					if step % 2 == 0:
						parameter.grad = self.smooth(
							parameter.grad, mode='col'
						)
					else:
						parameter.grad.zero_()

			optimizer.original_optimizer.step()
			optimizer.zero_grad()

	def aggregate(
		self, uploads: list[dict[str, torch.Tensor]], num_examples: list[int]
	) -> dict[str, torch.Tensor]:
		agg = Server.fedavg(uploads, num_examples)
		return {cid: agg for cid in uploads}, agg
