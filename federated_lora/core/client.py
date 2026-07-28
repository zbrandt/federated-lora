from __future__ import annotations

from dataclasses import dataclass

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, PreTrainedModel

from federated_lora.config import Config


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

	def dataloader(self, round_index: int) -> DataLoader:
		"""
		Combine dataset and a sampler and provide an iterable over the dataset.

		Parameters
		----------
		round_index : int
			TODO

		Returns
		-------
		DataLoader
			TODO
		"""
		# isolate shuffle from global stream with client and round index offset
		generator = torch.Generator().manual_seed(
			self.config.seed + self.client_id + round_index
		)

		return DataLoader(
			self.dataset,
			batch_size=self.config.batch_size,  # how many samples to per batch to load
			shuffle=True,  # data reschuffled at every epoch
			collate_fn=self.collator,  # merges list of samples to form a mini-batch of Tensors
			generator=generator,
		)

	def split_adapter_params(self, model: PreTrainedModel):
		"""
		TODO

		Parameters
		----------
		model : PreTrainedModel
			TODO

		Returns
		-------
		TODO
		"""
		lora_A, lora_B, head = [], [], []
		for name, parameter in model.named_parameters():
			if 'lora_A' in name:
				lora_A.append(parameter)
			elif 'lora_B' in name:
				lora_B.append(parameter)
			elif 'modules_to_save' in name:
				head.append(parameter)
		return lora_A, lora_B, head

	def extract_adapter_state(
		self, model: PreTrainedModel
	) -> dict[str, torch.Tensor]:
		"""
		TODO

		Parameters
		----------
		model : PreTrainedModel
			TODO

		Returns
		-------
		TODO
		"""
		return {
			key: value.detach().cpu().clone()
			for key, value in model.state_dict().items()
			if 'lora_' in key or 'modules_to_save' in key
		}
