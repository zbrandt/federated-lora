from __future__ import annotations

from itertools import cycle

import torch
from torch.optim import AdamW
from transformers import PreTrainedModel

from federated_lora.core.aggregate import aggregate
from federated_lora.core.client import Client
from federated_lora.methods.base import ClientResult


class LaLoRA:
	name = 'lalora'

	def local_update(
		self,
		client: Client,
		model: PreTrainedModel,
		adapter_state: dict[str, torch.Tensor],
		round_index: int,
	) -> ClientResult:
		"""
		Perform a local client update.

		This function splits the LoRA parameters in two from the A and B
		matrices, freezes all other parameters, and performs part of the local
		client update from the LA-LoRA framework through alternating updates of
		the A and B matrices across local steps.

		Parameters
		----------
		model : PreTrainedModel
			Model with LoRA adapters modules to train.
		adapter_state : dict[str, torch.Tensor]
			Global LoRA parameters to load into ``model`` before training.
		round_index : int
			The global round index used for seeding client's data shuffles.

		Returns
		-------
		ClientResults
			The udpated LoRA tensors plus training metrics.
		"""
		# TODO
		model.load_state_dict(adapter_state, strict=False)

		# set the model in training mode
		model.train()

		lora_A, lora_B, head = client.split_adapter_params(model)

		# freeze all other parameters
		for parameter in model.parameters():
			parameter.requires_grad = False
		for parameter in (*lora_A, *lora_B, *head):
			parameter.requires_grad = True

			# declare respective optimizers for A and B matrix parameters
		opt_A = AdamW(lora_A, lr=client.config.lr_a)
		opt_B = AdamW(lora_B, lr=client.config.lr_b)
		opt_head = AdamW(head, lr=client.config.lr_head) if head else None

		batches = cycle(client.dataloader(round_index))

		total_loss = 0.0
		for step in range(1, client.config.local_steps + 1):
			batch = next(batches)
			batch = {
				key: value.to(client.device) for key, value in batch.items()
			}  # move batch items to the correct device

			for parameter in lora_A:
				parameter.requires_grad = step % 2 == 0

			for parameter in lora_B:
				parameter.requires_grad = step % 2 == 1

			# TODO: Add differential privacy via clipping and smoothing filter
			optimizer = opt_A if step % 2 == 0 else opt_B
			optimizer.zero_grad(
				set_to_none=True
			)  # reset the gradients of all optimized Tensors
			if opt_head:
				opt_head.zero_grad(set_to_none=True)
			loss = model(
				**batch
			).loss  # access scalar loss tensor calculated with forward pass
			loss.backward()  # backward pass and optimization step
			optimizer.step()
			if opt_head:
				opt_head.step()
			total_loss += float(loss.item())

		updated_state = {
			key: value.detach().cpu().clone()
			for key, value in model.state_dict().items()
			if 'lora_' in key or 'modules_to_save' in key
		}

		return ClientResult(
			state_dict=updated_state,
			n_examples=len(client.dataset),
			average_loss=total_loss / client.config.local_steps,
		)

	def aggregate(self, states):
		return aggregate(states)
