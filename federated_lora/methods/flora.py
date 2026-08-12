from __future__ import annotations

import torch
import torch.nn as nn
from opacus.optimizers.optimizer import DPOptimizer

from federated_lora.server import Server


class FLoRA:
	name = 'flora'

	def set_target_modules(self, model: nn.Module) -> None:
		for name, parameter in model.named_parameters():
			if 'lora_' in name or 'classifier' in name:
				parameter.requires_grad = True
			else:
				parameter.requires_grad = False

	def step(
		self, model: nn.Module, optimizer: DPOptimizer, round: int, step: int
	) -> None:
		optimizer.step()
		optimizer.zero_grad()

	def aggregate(
		self,
		uploads: list[dict[str, torch.Tensor]],
		num_examples: list[int],
	) -> dict[str, torch.Tensor]:
		# TODO: implement FLoRA aggregation
		return Server.fedavg(uploads, num_examples)
		# total = float(sum(num_examples))
		# weights = [n / total for n in num_examples]
		# keys = uploads[0].keys()

		# a_suffix, b_suffix = ".A.weight", ".B.weight"
		# prefixes = {key[: -len(a_suffix)] for key in keys if key.endswith(a_suffix)}
		# aggregated: dict[str, torch.Tensor] = {}

		# for prefix in prefixes:
		#     a_key, b_key = prefix + a_suffix, prefix + b_suffix
		#     rank = uploads[0][a_key].shape[0]

		#     a_stack = torch.cat([upload[a_key].to(torch.float32) for upload in uploads], dim=0)
		#     b_stack = torch.cat(
		#         [upload[b_key].to(torch.float32) * weight for upload, weight in zip(uploads, weights)], dim=1
		#     )
		#     delta_w = b_stack @ a_stack

		#     U, S, Vt = torch.linalg.svd(delta_w, full_matrices=False)
		#     sqrt_s = S[:rank].clamp_min(0).sqrt()
		#     new_b = U[:, :rank] * sqrt_s.unsqueeze(0)
		#     new_a = sqrt_s.unsqueeze(1) * Vt[:rank, :]

		#     ref_a, ref_b = uploads[0][a_key], uploads[0][b_key]
		#     aggregated[a_key] = new_a.to(dtype=ref_a.dtype, device=ref_a.device)
		#     aggregated[b_key] = new_b.to(dtype=ref_b.dtype, device=ref_b.device)

		# for key in keys:
		#     if key.endswith(a_suffix) or key.endswith(b_suffix):
		#         continue
		#     reference = uploads[0][key]
		#     accumulator = torch.zeros_like(reference, dtype=torch.float32)
		#     for upload, weight in zip(uploads, weights):
		#         accumulator += upload[key].to(torch.float32) * weight
		#     aggregated[key] = accumulator.to(dtype=reference.dtype, device=reference.device)

		# return aggregated
