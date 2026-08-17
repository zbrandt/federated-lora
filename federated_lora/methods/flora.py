from __future__ import annotations

import torch
import torch.nn as nn
from opacus.optimizers.optimizer import DPOptimizer
from federated_lora.model import get_base_state, reset_lora_layers

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
	) -> tuple[dict[int, dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
		# TODO: implement FLoRA aggregation
		# return Server.fedavg(uploads, num_examples)
		total = sum(num_examples.values())
		weights = {cid: n / total for cid, n in num_examples.items()}
		
		param_names = next(iter(uploads.values())).keys()
		aggregated: dict[str, torch.Tensor] = {}

		for name in param_names:
			if 'lora_A' in name:
				# scale each client A by FedAvg weight and then stack along rank dimension
				scaled = [
					upload[name] * weights[cid] for cid, upload in uploads.items()
				]
				aggregated[name] = torch.cat(scaled, dim=0)
			elif "lora_B" in name:
				# unweighted so B_stack @ A_stack = sum_k{w_k*(B_k @ A_k)}
				unscaled = [upload[name] for upload in uploads.values()]
				aggregated[name] = torch.cat(unscaled, dim=1)
			else:
				# handles non lora_A or lora_B, instead probably classifier head 
				# for anything else (classifier head) we just want to do fedavg
				reference = next(iter(uploads.values()))[name]
				accumulator = torch.zeros_like(reference, dtype=torch.float32)
				for cid, upload in uploads.items():
					accumulator += upload[name].to(torch.float32) * weights[cid]
				aggregated[name] = accumulator.to(reference.dtype)
		return {cid: aggregated for cid in uploads}, aggregated
	
	def redistribute(
			self, 
			server_model: nn.Module,
			per_client_state: dict[int, dict[str, torch.Tensor]],
			client_models: dict[int, nn.Module], 
			) -> tuple[dict[int, dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
			stacked = next(iter(per_client_state.values()))
			base_state = get_base_state(server_model)
			
			merged_base: dict[str, torch.Tensor] = {}
			for a_key in [key for key in stacked if "lora_A" in key]:
				b_key = a_key.replace('lora_A', 'lora_B')
				base_key = a_key.replace('.lora_A.default.weight', ".base_layer.weight")

				# merge the stacked delta into base
				delta_w = stacked[b_key] @ stacked[a_key]
				merged_base[base_key] = base_state[base_key] + delta_w

			# ferda classifier
			non_lora = {k: v for k, v in stacked.items() if "lora_" not in k}
			# now reinit in place, at each model's own existing rank w peft
			# reset lora params gives A random init and B zeros
			reset_lora_layers(server_model)
			for client_model in client_models.values():
				reset_lora_layers(client_model)


			per_client = {
				cid: {**merged_base, **non_lora}
				for cid in client_models
			}
			eval_state = {**merged_base, **non_lora}

			return per_client, eval_state

						