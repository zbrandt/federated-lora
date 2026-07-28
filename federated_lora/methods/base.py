from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(slots=True)
class ClientResult:
	state_dict: dict[str, torch.Tensor]
	n_examples: int
	average_loss: float


class Method(Protocol):
	name: str

	def local_update(
		self, client, model, adapter_state, round_index
	) -> ClientResult: ...
	def aggregate(self, states: list[dict]) -> dict: ...
