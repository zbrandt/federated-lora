from __future__ import annotations

from typing import Protocol

import torch
import torch.nn as nn


class Method(Protocol):
	"""
	A federated fine-tuning method.
	"""

	name: str

	def set_target_modules(self, model: nn.Module) -> None:
		"""
		Configure which modules of the model to adapt.

		Parameters
		----------
		model : Module
			The freshly created PEFT model.
		"""
		...

	# TODO: rewrite docstring
	def local_update(self, model: nn.Module, round: int) -> None:
		"""
		Adjust the per-sample gradients in place, before the optimizer step.

		Called by ``Client.update`` after ``loss.backward()`` and before
		``optimizer.step()``. Implementations mutate ``parameter.grad_sample``
		(e.g. zeroing the frozen LoRA factor for the current round). Methods that
		need no per-step masking (standard DP-SGD) implement this as a no-op.

		Parameters
		----------
		model : Module
			The client's (Opacus-wrapped) model with populated ``grad_sample``s.
		round : int
			The current global communication round (1-indexed).
		"""
		...

	# TODO: rewrite docstring
	def aggregate(
		self, uploads: list[dict[str, torch.Tensor]], round: int
	) -> dict[str, torch.Tensor]:
		"""
		Combine the per-client trainable-state uploads into the new global state.

		Parameters
		----------
		uploads : list[dict[str, torch.Tensor]]
			One state-dict snapshot per client, each restricted to the federated
			(trainable) keys: the LoRA A/B factors and the classification head.
		round : int
			The current global communication round (1-indexed).

		Returns
		-------
		dict[str, torch.Tensor]
			The new global state, i.e. the per-key aggregate over clients.
		"""
		...
