from __future__ import annotations

import torch
import torch.nn as nn
from opacus.optimizers.optimizer import DPOptimizer
from peft.tuners.lora import LoraLayer

from federated_lora.model import unwrap

class HetLoRA:

	name = "hetlora"
	# clients are built at a shared max rank and self-prune down to their
	# own target rank during local training
	prunes = True

	def set_target_modules(self, model: nn.Module) -> None:
		"""
		Configure which modules of the model to adapt.

		Parameters
		----------
		model : nn.Module
			The PEFT model.
		"""
		for name, parameter in model.named_parameters():
			if "lora_" in name or "classifier" in name:
				parameter.requires_grad = True
			else:
				parameter.requires_grad = False

	def step(
		self,
		model: nn.Module,
		optimizer: DPOptimizer,
		round: int,
		step: int,
		total_steps: int | None = None,
		target_rank: int | None = None,
	) -> None:
		"""
		Perform a single optimization step, then self-prune the client's LoRA
		rank towards target rank.

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
		total_steps : int | None
			The total number of local steps this round.
		target_rank : int | None
			This client's target LoRA rank. ``None`` means this client
			doesn't prune (e.g. it's already at the shared max rank).
		"""
		optimizer.step()
		optimizer.zero_grad()

		if target_rank is None or not total_steps:
			return

		target = unwrap(model)
		progress = (step + 1) / total_steps
		with torch.no_grad():
			for module in target.modules():
				if not isinstance(module, LoraLayer):
					continue
				lora_A = module.lora_A['default'].weight
				lora_B = module.lora_B['default'].weight
				r_init = lora_A.shape[0]
				if target_rank >= r_init:
					continue

				active_rank = int(r_init - (r_init - target_rank) * progress + 0.5)
				active_rank = max(target_rank, min(r_init, active_rank))

				importance = lora_A.norm(dim=1) * lora_B.norm(dim=0)
				keep = torch.topk(importance, active_rank).indices
				mask = torch.zeros_like(importance, dtype=torch.bool)
				mask[keep] = True

				lora_A[~mask] = 0.0
				lora_B[:, ~mask] = 0.0

	def aggregate(
		self,
		uploads: dict[int, dict[str, torch.Tensor]],
		num_examples: dict[int, int],
	) -> tuple[dict[int, dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
		"""
		Aggregate client trainable parameter state-dict updates to the model.

		Parameters
		----------
		uploads : dict[int, dict[str, torch.Tensor]]
			The client trainable parameter state-dict updates to the model.
		num_examples : dict[int, int]
			The total number of examples of each client.

		Returns
		-------
		tuple[dict[int, dict[str, torch.Tensor]], dict[str, torch.Tensor]]
			Per-client state to redistribute, and the aggregated global state.
		"""
		total = sum(num_examples.values())
		data_weights = {cid: n / total for cid, n in num_examples.items()}
		reference = next(iter(uploads.values()))
		a_keys = [k for k in reference if "lora_A" in k]

		# p_k = delta_W_k / Z for magnitude based aggregation weight
		magnitude: dict[int, float] = {}
		for cid, upload in uploads.items():
			sq_sum = 0.0
			for a_key in a_keys:
				b_key = a_key.replace('lora_A', 'lora_B')
				delta = upload[b_key].float() @ upload[a_key].float()
				sq_sum += torch.sum(delta ** 2).item()
			magnitude[cid] = sq_sum ** 0.5
		z = max(sum(magnitude.values()), 1e-12)
		weights = {cid: s / z for cid, s in magnitude.items()}

		per_client: dict[int, dict[str, torch.Tensor]] = {cid: {} for cid in uploads}
		eval_state: dict[str, torch.Tensor] = {}

		eps = 1e-12
		for a_key in a_keys:
			b_key = a_key.replace('lora_A', 'lora_B')

			r = reference[a_key].shape[0]
			n = reference[a_key].shape[1]
			m = reference[b_key].shape[0]

			a_sum = torch.zeros(r, n)
			b_sum = torch.zeros(m, r)
			# slot weight only accrues from clients whose slot survived pruning
			slot_weight = torch.zeros(r)

			for cid, upload in uploads.items():
				a_i = upload[a_key].float()
				b_i = upload[b_key].float()
				w = weights[cid]

				active = (a_i.pow(2).sum(dim=1) + b_i.pow(2).sum(dim=0)) > eps
				a_sum += w * a_i * active[:, None]
				b_sum += w * b_i * active[None, :]
				slot_weight += w * active.float()

			a_global = a_sum / slot_weight.clamp_min(eps)[:, None]
			b_global = b_sum / slot_weight.clamp_min(eps)[None, :]

			for cid in uploads:
				per_client[cid][a_key] = a_global.to(reference[a_key].dtype)
				per_client[cid][b_key] = b_global.to(reference[b_key].dtype)

			eval_state[a_key] = a_global.to(reference[a_key].dtype)
			eval_state[b_key] = b_global.to(reference[b_key].dtype)

		for key in reference:
			if 'lora_' in key:
				continue
			acc = sum(data_weights[cid] * uploads[cid][key].float() for cid in uploads)
			acc = acc.to(reference[key].dtype)
			eval_state[key] = acc
			for cid in per_client:
				per_client[cid][key] = acc

		return per_client, eval_state
