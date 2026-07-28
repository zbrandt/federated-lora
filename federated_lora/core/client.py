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
	def __init__(self, client_id: int, dataset: Dataset, config: Config, 
		device: torch.device, collator: DataCollatorWithPadding,
	) -> None:
		self.client_id = client_id
		self.dataset = dataset
		self.config = config
		self.device = device
		self.collator = collator
