from __future__ import annotations

import torch
import torch.nn as nn
from opacus.optimizers.optimizer import DPOptimizer
from peft.tuners.lora import LoraLayer

from federated_lora.model import unwrap

class FlexLoRA:

	name = "flexlora"

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
		total = sum(num_examples.values())
		# get for redistribute()
		self._weights = {cid: n / total for cid, n in num_examples.items()}
		self._uploads = uploads
		# pass through, overwritten by redistribute
		return uploads, next(iter(uploads.values()))
	
	def redistribute(
		self,
		server_model: nn.Module,
		per_client_state: dict[int, dict[str, torch.Tensor]],
		client_models: dict[int, nn.Module],
	) -> tuple[dict[int, dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
		weights, uploads = self._weights, self._uploads
		reference = next(iter(uploads.values()))
		a_keys = [k for k in reference if 'lora_A' in k]

		def scaling_of(model: nn.Module, module_path: str) -> float:
			layer = unwrap(model).get_submodule(module_path)
			assert isinstance(layer, LoraLayer)
			return layer.scaling['default']
		
		per_client: dict[int, dict[str, torch.Tensor]] = {
			cid: {} for cid in client_models
		}
		eval_state: dict[str, torch.Tensor] = {}
		
		for a_key in a_keys:
			b_key = a_key.replace('lora_A', "lora_B")
			# use module path to turn it into loralayer
			module_path = a_key.replace('.lora_A.default.weight', '')
			
			# what is delta avg 
			# delta avg is thge weighted average full rank weight delta for one loramodule. 
			# s also gives the scaling factor per rank. higher rank = stronger weighting 
			delta_avg = None 
			for cid, upload in uploads.items():
				s_i = scaling_of(client_models[cid], module_path)
				delta_i = s_i * (upload[b_key] @ upload[a_key]).float()
				delta_avg = weights[cid] * delta_i if delta_avg is None else delta_avg + weights[cid] * delta_i
				
			# perform svd
			U, S, Vh = torch.linalg.svd(delta_avg, full_matrices=False)
			# have to do the sqrt of S because the two roots of S need to get multiplied together
			sqrt_S = S.clamp_min(0).sqrt()
		
			for cid, client_model in client_models.items():
				# clients current rank 
				r_i = uploads[cid][a_key].shape[0]
				s_i = scaling_of(client_model, module_path)
				a_new = (sqrt_S[:r_i, None] * Vh[:r_i]).to(uploads[cid][a_key].dtype)
				b_new = (U[:, :r_i] * sqrt_S[:r_i]).to(uploads[cid][b_key].dtype) / s_i
				per_client[cid][a_key] = a_new
				per_client[cid][b_key] = b_new
				
			r_eval = server_model.get_submodule(module_path).lora_A['default'].weight.shape[0]
			s_eval = scaling_of(server_model, module_path)
			eval_state[a_key] = (sqrt_S[:r_eval, None] *Vh[:r_eval])
			eval_state[b_key] = (U[:, :r_eval] * sqrt_S[:r_eval]) / s_eval
			
		# classifier head gets plain fedavg not rank dependent 
		for key in reference: 
			if 'lora_' in key:
				continue
			acc = sum(weights[cid] * uploads[cid][key].float() for cid in uploads)
			acc = acc.to(reference[key].dtype)
			eval_state[key] = acc
			for cid in per_client:
				per_client[cid][key] = acc
				
		return per_client, eval_state 
