from __future__ import annotations

from dataclasses import dataclass
from itertools import cycle

import torch
from datasets import Dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, PreTrainedModel

from la_lora.config import Config


@dataclass(slots=True)
class ClientResult:
	state_dict: dict[str, torch.Tensor]
	n_examples: int
	average_loss: float


class Client:
	def __init__(
		self,
		client_id: int,
		dataset: Dataset,
		config: Config,
		device: torch.device,
		collator: DataCollatorWithPadding,
	) -> None:
		self.client_id = client_id
		self.dataset = dataset
		self.config = config
		self.device = device
		self.collator = collator

	def local_update(
		self, model: PreTrainedModel, adapter_state: dict[str, torch.Tensor]
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
		        TODO: Figure out what this is exactly

		Returns
		-------
		ClientResults
		        TODO: Evaluate this return class
		"""
		model.load_state_dict(
			adapter_state, strict=False
		)  # TODO: figure out what this is

		# set the model in training mode
		model.train()

		# split LoRA parameters from A and B matrices
		lora_A, lora_B = [], []
		head = []  # classifier head parameters
		for name, parameter in model.named_parameters():
			if 'lora_A' in name:
				lora_A.append(parameter)
			elif 'lora_B' in name:
				lora_B.append(parameter)
			elif 'modules_to_save' in name:
				head.append(parameter)

		# freeze all other parameters
		for name, parameter in model.named_parameters():
			parameter.requires_grad = False
		for parameter in (*lora_A, *lora_B, *head):
			parameter.requires_grad = True

		# declare respective optimizers for A and B matrix parameters
		opt_A = AdamW(lora_A, lr=self.config.lr_a)
		opt_B = AdamW(lora_B, lr=self.config.lr_b)
		opt_head = AdamW(head, lr=self.config.lr_head) if head else None

		# combine dataset and a sampler, and provide an iterable over dataset
		train_dataloader = DataLoader(
			self.dataset,
			batch_size=self.config.batch_size,  # how many samples to per batch to load
			shuffle=True,  # data reschuffled at every epoch
			collate_fn=self.collator,  # merges list of samples to form a mini-batch of Tensors
		)

		batches = cycle(train_dataloader)

		total_tokens = 0
		total_loss = 0.0
		for step in range(1, self.config.local_steps + 1):
			batch = next(batches)
			batch = {
				key: value.to(self.device) for key, value in batch.items()
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
			n_examples=len(self.dataset),
			average_loss=total_loss / self.config.local_steps,
		)
