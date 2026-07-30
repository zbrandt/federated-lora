from __future__ import annotations

from itertools import cycle

import torch
from opacus import PrivacyEngine
from torch.nn import Module
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader


class LaLoRA:
	name = 'lalora'

	def smooth(self, grad: torch.Tensor, mode: str) -> torch.Tensor:
		"""
		TODO

		Parameters
		----------
		TODO

		Returns
		-------
		TODO
		"""
		if grad is None:
			return None

		# Fixed 1D Gaussian kernel G_s
		kernel = (
			torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], device=grad.device) / 16.0
		)
		kernel = kernel.view(1, 1, 5)

		if mode == 'row':  # Matrix A: shape (r, n)
			r, _n = grad.shape
			x = grad.unsqueeze(1)  # Shape: (r, 1, n)
			x_padded = F.pad(x, (2, 2), mode='reflect')
			filtered = F.conv1d(x_padded, kernel).squeeze(1)
			return filtered

		else:  # Matrix B: shape (m, r)
			_m, _r = grad.shape
			x = grad.T.unsqueeze(1)  # Transpose to (r, 1, m)
			x_padded = F.pad(x, (2, 2), mode='reflect')
			filtered = F.conv1d(x_padded, kernel).squeeze(1).T
			return filtered

	def local_update(
		self,
		model: Module,
		optimizer: AdamW,
		train_dataloader: DataLoader,
		noise_multiplier: float,
		max_grad_norm: float,
		local_steps: int,
		device: str,
	):
		"""
		TODO

		Parameters
		----------
		TODO

		Returns
		-------
		TODO
		"""
		model.train()

		privacy_engine = PrivacyEngine()
		model, optimizer, train_dataloader = privacy_engine.make_private(
			module=model,
			optimizer=optimizer,
			data_loader=train_dataloader,
			noise_multiplier=noise_multiplier,
			max_grad_norm=max_grad_norm,
			poisson_sampling=False,  # TODO
		)

		batches = cycle(train_dataloader)

		total_loss = 0.0
		for k in range(1, local_steps + 1):
			batch = next(batches)
			batch = {key: value.to(device) for key, value in batch.items()}

			optimizer.zero_grad()
			loss = model(**batch).loss
			loss.backward()

			if optimizer.pre_step():
				is_odd = k % 2 != 0
				for name, parameter in model.named_parameters():
					if parameter.grad is None:
						continue
					if 'lora_A' in name:
						parameter.grad = (
							self.smooth(parameter.grad, mode='row')
							if not is_odd
							else None
						)
					elif 'lora_B' in name:
						parameter.grad = (
							self.smooth(parameter.grad, mode='col')
							if is_odd
							else None
						)

				optimizer.original_optimizer.step()
				optimizer.zero_grad()

			total_loss += float(loss.item())

		# Unwrap the Opacus GradSampleModule so the server sees a plain module
		# whose parameters were updated in place; the server snapshots the
		# trainable tensors it needs from it.
		return model.to_standard_module(), total_loss / max(1, local_steps)

	def aggregate(
		self, client_uploads: list[dict[str, torch.Tensor]]
	) -> dict[str, torch.Tensor]:
		"""
		Average each trainable tensor across the client uploads (FedAvg).

		Parameters
		----------
		client_uploads : list[dict[str, torch.Tensor]]
			One state-dict snapshot per client, each restricted to the
			federated (trainable) keys: the LoRA A/B factors and the
			classification head.

		Returns
		-------
		dict[str, torch.Tensor]
			The new global state, i.e. the per-key mean over clients.
		"""
		return {
			key: torch.stack(
				[upload[key].float() for upload in client_uploads], dim=0
			).mean(dim=0)
			for key in client_uploads[0]
		}
