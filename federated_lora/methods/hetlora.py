from __future__ import annotations

import torch
import torch.nn as nn
from opacus.optimizers.optimizer import DPOptimizer

class HetLoRA:

	name = "hetlora"

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
		self, model: nn.Module, optimizer: DPOptimizer, round: int, step: int
	) -> None:
		"""
		Perform a single optimization step to update trainable parameters.

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
		"""
		optimizer.step()
		optimizer.zero_grad()

	def aggregate(
		self,
		uploads: dict[int, dict[str, torch.Tensor]],
		num_examples: dict[int, int],
	) -> tuple[dict[int, dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
		"""
		Aggregate client trainable parameter state-dict updates to the model.

		Parameters
		----------
		uploads : list[dict[str, torch.Tensor]]
			The client trainable parameter state-dict updates to the model.
		num_examples : list[int]
			The total number of examples of each client.

		Returns
		-------
		dict[str, torch.Tensor]
			The aggregated client updates to the model.
		"""
		
        # TODO: Handle the pruning case


		total = sum(num_examples.values())
		data_weights = {cid: n / total for cid, n in num_examples.items()}
		reference = next(iter(uploads.values()))
		a_keys = [k for k in reference if "lora_A" in k]
		
        # p_k = delta_W_k / Z for magniute based aggregation weight
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
		






		for a_key in a_keys:
			b_key = a_key.replace('lora_A', 'lora_B')
			
			# get r max for padding purposes
			r_max = max(upload[a_key].shape[0] for upload in uploads.values())
			n = reference[a_key].shape[1]
			m = reference[b_key].shape[0]
			a_sum = torch.zeros(r_max, n)
			b_sum = torch.zeros(m, r_max)
			# slot weight is used for accumulating the ranks to make sure that we get to full rank 
			slot_weight = torch.zeros(r_max)
			for cid, upload in uploads.items():
				r_i = upload[a_key].shape[0]
				w = weights[cid]
				a_sum[:r_i] += w * upload[a_key].float()
				b_sum[:, :r_i] += w *upload[b_key].float()
				slot_weight[:r_i] += w
			a_global = a_sum / slot_weight.clamp_min(1e-12)[:, None]
			b_global = b_sum / slot_weight.clamp_min(1e-12)[None, :]
			for cid, upload in uploads.items():
				r_i = upload[a_key].shape[0]
				per_client[cid][a_key] = a_global[:r_i].to(upload[a_key].dtype)
				per_client[cid][b_key] = b_global[:, :r_i].to(upload[b_key].dtype)
				
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