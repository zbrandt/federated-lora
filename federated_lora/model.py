from __future__ import annotations

import torch
import torch.nn as nn
from opacus import GradSampleModule
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
	AutoModelForImageClassification,
	AutoModelForSequenceClassification,
)

from federated_lora.method import Method


def create_peft_model(
	name: str,
	num_labels: int,
	lora_rank: int,
	lora_alpha: int,
	target_modules: tuple[str, ...],
	lora_dropout: float,
	method: Method,
	device: torch.device,
	experiment: str,
) -> PeftModel:
	"""
	Create a trainable PeftModel from the base model and the parameters for
	training with LoRA.

	Parameters
	----------
	name : str
		The model id of a pretrained model.
	num_labels : int
		The number of labels to use in the classification layer.
	lora_rank : int
		The LoRA attention dimension (the "rank").
	lora_alpha : int
		The alpha parameter for LoRA scaling.
	target_modules : tuple[str, ...]
		The names of the modules to apply the adapter to.
	lora_dropout : float
		The dropout probability for LoRA layers.
	method : Method
		The federated fine-tuning method.
	device : torch.device
		The selected device type ("cpu" or "cuda").
	experiment : str
		The experiment setup ("vision" for image classification or "text" for
		language understanding).

	Returns
	-------
	PeftModel
		The PEFT model object from the in-place modified model and LoRA config.
	"""
	if experiment == 'text':
		base = AutoModelForSequenceClassification.from_pretrained(
			name, num_labels=num_labels
		).to(device)
	else:
		base = AutoModelForImageClassification.from_pretrained(
			name, num_labels=num_labels, ignore_mismatched_sizes=True
		).to(device)

	peft_config = LoraConfig(
		r=lora_rank,
		lora_alpha=lora_alpha,
		target_modules=list(target_modules),
		lora_dropout=lora_dropout,
		bias='none',
	)

	model = get_peft_model(base, peft_config)

	method.set_target_modules(model)

	return model


# TODO: figure out how to unrwap the standard nn.Module wrapped by the privacy engine
def unwrap(model: nn.Module) -> nn.Module:
	"""
	Unrwap the standard nn.Module.
	"""
	return getattr(model, '_module', model)


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


def group_trainable_parameters(model: nn.Module) -> tuple[list, list, list]:
	"""
	Group trainable parameters from LoRA matrices A & B and the classifier head.
	"""
	params_A, params_B, params_head = [], [], []
	for name, param in model.named_parameters():
		if not param.requires_grad:
			continue
		if 'lora_A' in name:
			params_A.append(param)
		elif 'lora_B' in name:
			params_B.append(param)
		else:
			params_head.append(param)
	return params_A, params_B, params_head


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
