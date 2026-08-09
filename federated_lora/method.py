from __future__ import annotations

from typing import Protocol

import torch
from torch.nn import Module

# TODO: rewrite docstrings
class Method(Protocol):
	"""
	A federated LoRA training method.

	Standard methods implement the "thin" contract below: ``Client.update`` runs
	the shared DP-SGD loop and calls ``local_update`` after each backward pass to
	mask or adjust the per-sample gradients in place, before the DP optimizer
	step. A method that must own its full local loop -- e.g. LaLoRA, whose
	gradient smoothing runs *after* Opacus clips and noises -- instead exposes a
	``run_local`` method and is detected via ``hasattr(method, 'run_local')``, so
	no ``bespoke`` flag is needed.
	"""

	name: str

	def prepare_model(self, model: Module) -> None:
		"""
		Select which parameters this method trains by setting ``requires_grad``,
		before the optimizer and Opacus are built.

		Called once by ``experiment.build`` on the global model; the flags are
		inherited by the per-client copies. Most methods train both LoRA factors
		and the classification head; FFA-LoRA freezes A here so it stays out of
		the DP budget entirely.

		Parameters
		----------
		model : Module
			The freshly created PEFT model.
		"""
		...

	def local_update(self, model: Module, round: int) -> None:
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
