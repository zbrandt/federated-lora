from __future__ import annotations

import torch
import torch.nn as nn
from opacus.optimizers.optimizer import DPOptimizer
from torch.nn import functional as F

from federated_lora.model import dewindow_grad_samples
from federated_lora.privacy import privatize, zero_grad
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

	def smooth_grad(self, name: str, grad: torch.Tensor) -> torch.Tensor:
		if 'lora_A' in name:
			return self.smooth(grad, mode='row')
		if 'lora_B' in name:
			return self.smooth(grad, mode='col')
		return grad

	def step(
		self,
		model: nn.Module,
		optimizer: DPOptimizer,
		round: int,
		step: int,
		batch_size: int,
	) -> None:
		frozen = ('lora_A',) if step % 2 == 0 else ('lora_B',)
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
					else:
						p.grad = self.smooth_grad(name, p.grad)

				optimizer.original_optimizer.step()
		else:
			if optimizer.step_hook:
				optimizer.step_hook(optimizer)

			for name, p in model.named_parameters():
				if p.requires_grad and any(f in name for f in frozen):
					p.grad_sample = None
					p.grad = None

			params = privatize(model, optimizer)

			for name, p in model.named_parameters():
				if p.requires_grad and p.grad is not None:
					p.grad = self.smooth_grad(name, p.grad)

			optimizer.original_optimizer.step()
			zero_grad(params)

		optimizer.zero_grad()

	def aggregate(
		self,
		uploads: list[dict[str, torch.Tensor]],
		num_examples: list[int],
	) -> dict[str, torch.Tensor]:
		return Server.fedavg(uploads, num_examples)
