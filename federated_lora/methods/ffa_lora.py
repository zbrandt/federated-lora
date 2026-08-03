from __future__ import annotations

from itertools import cycle

import torch
from opacus import PrivacyEngine
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader


class FFALoRA:
	name = 'ffa_lora'

	def local_update(
		self,
		model: Module,
		optimizer: Optimizer,
		train_dataloader: DataLoader,
		noise_multiplier: float,
		max_grad_norm: float,
		local_steps: int,
		device: str,
		round_index: int = 0,
	):
		model.train()

		privacy_engine = PrivacyEngine()
		model, optimizer, train_dataloader = privacy_engine.make_private(
			module=model,
			optimizer=optimizer,
			data_loader=train_dataloader,
			noise_multiplier=noise_multiplier,
			max_grad_norm=max_grad_norm,
			poisson_sampling=True,
		)

		batches = cycle(train_dataloader)

		total_loss = 0.0
		for _ in range(local_steps):
			batch = next(batches)
			batch = {key: value.to(device) for key, value in batch.items()}

			optimizer.zero_grad()
			loss = model(**batch).loss
			loss.backward()

			# Freeze A at its (shared) init: zero its per-sample gradient
			# before clipping so the DP clip norm and injected noise bound
			# only B, avoiding the joint-norm amplification of A/B updates.
			for name, parameter in model.named_parameters():
				grad_sample = getattr(parameter, 'grad_sample', None)
				if grad_sample is None:
					continue
				if 'lora_A' in name:
					parameter.grad_sample = torch.zeros_like(grad_sample)

			if optimizer.pre_step():
				for name, parameter in model.named_parameters():
					if parameter.grad is None:
						continue
					if 'lora_A' in name:
						parameter.grad = None
				optimizer.original_optimizer.step()
				optimizer.zero_grad()

			total_loss += float(loss.item())

		return model.to_standard_module(), total_loss / max(1, local_steps)

	def aggregate(
		self, client_uploads: list[dict[str, torch.Tensor]]
	) -> dict[str, torch.Tensor]:
		return {
			key: torch.stack(
				[upload[key].float() for upload in client_uploads], dim=0
			).mean(dim=0)
			for key in client_uploads[0]
		}