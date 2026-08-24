from __future__ import annotations

import torch
import torch.nn as nn
from opacus import GradSampleModule


# TODO: figure out how to unrwap the standard nn.Module wrapped by the privacy engine
def unwrap(model: nn.Module) -> nn.Module:
	"""
	Unrwap the standard nn.Module.
	"""
	return model._module if isinstance(model, GradSampleModule) else model


def get_trainable_parameters(model: nn.Module) -> set[str]:
	"""
	Get the names of the trainable parameters.
	"""
	target = unwrap(model)
	return {
		name
		for name, param in target.named_parameters()
		if param.requires_grad
	}


def get_trainable_state(model: nn.Module) -> dict[str, torch.Tensor]:
	"""
	Get trainable parameters from model.
	"""
	target = unwrap(model)
	trainable = get_trainable_parameters(target)
	return {
		key: value.detach().cpu().clone()
		for key, value in target.state_dict().items()
		if key in trainable
	}


def load_trainable_state(
	model: nn.Module, state: dict[str, torch.Tensor]
) -> None:
	"""
	Load trainable parameters from ``state`` into model.
	"""
	target = unwrap(model)
	target.load_state_dict(state, strict=False)


def dewindow_grad_samples(
	model: GradSampleModule, batch_size: int
) -> torch.Tensor:
	"""
	Collapse a per-sample gradient with a windowed leading dimension back to one
	row per example.

	Parameters
	----------
	model : GradSampleModule
		Wrapped model to compute per-sample gradients.
	batch_size : int
		The number of examples processed before parameters are updated.

	Returns
	-------
	torch.Tensor
		A recovered gradient sample from summing over the windows dimension.
	"""
	for _, p in model.named_parameters():
		if p.requires_grad and getattr(p, 'grad_sample', None) is not None:
			n = p.grad_sample.shape[0]

			if n == batch_size:
				continue

			num_windows = n // batch_size
			reshaped = p.grad_sample.reshape(
				batch_size, num_windows, *p.grad_sample.shape[1:]
			)

			p.grad_sample = reshaped.sum(dim=1)


# Windowed attention folds windows into the batch dimension before transformer
# blocks, so Opacus sees shape (``batch_size`` * ``num_windows``, ``tokens``,
# ``C``) instead of (``batch_size``, ``tokens``, ``C``) for those layers.

# Hugging Face's Swin transformer's ``window_partition`` is batch-major, so
# each ``num_windows``-sized run of rows belongs to one example. Summing it
# recovers the true per-example gradient, the same way Opacus already sums
# over the token axis within one window via ``einsum`` contraction.

# A ``grad_sample`` with leading dimension already equal to batch_size passes
# through unchanged.
