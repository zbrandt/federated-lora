from __future__ import annotations

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import AutoModelForImageClassification

from federated_lora.method import Method

def reset_lora_layers(model: nn.Module) -> None:
	'''
	Reinitalize every LoRA adapter's AB weights in place, at whatever rank the layer is already constructed with.
	'''
	target = unwrap(model)
	for module in target.modules():
		if isinstance(module, LoraLayer):
			module.reset_lora_parameters('default', True)

# TODO: Modify so that the lora_rank can be a list of ranks
def create_peft_model(
	name: str,
	num_labels: int,
	lora_rank: int,
	lora_alpha: int,
	target_modules: tuple[str, ...],
	lora_dropout: float,
	method: Method,
	device: torch.device,
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
	target_modules : tuple[str, ...]
		The names of the modules to apply the adapter to.
	lora_alpha : int
		The alpha parameter for LoRA scaling.
	lora_dropout : int
		The dropout probability for LoRA layers.
	method : Method
		A federated fine-tuning method.
	device : torch.device
		The selected device type ("cpu" or "cuda").

	Returns
	-------
	PeftModel
		The PEFT model object from the in-place modified model and LoRA config.
	"""
	base = AutoModelForImageClassification.from_pretrained(
		name, num_labels=num_labels, ignore_mismatched_sizes=True
	).to(device)

	# use the last rank for initial configuration, but the rank can be changed later
	peft_config = LoraConfig(
		r=lora_rank,
		lora_alpha=lora_alpha,
		target_modules=target_modules,
		lora_dropout=lora_dropout,
		bias='none',
		# modules_to_save=['classifier'],
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


# for every lora adapted module, get the frozen base-layer
def get_base_state(model: nn.Module) -> dict[str, torch.Tensor]:
	"""
	Get the frozen base-layer weights for every LoRA- adapted module. 
	"""
	target = unwrap(model)
	trainable = get_trainable_parameters(target)
	return {
		key: value.detach().cpu().clone()
		for key, value in target.state_dict().items()
		if key not in trainable and 'base_layer' in key
	}

def load_trainable_state(
	model: nn.Module, state: dict[str, torch.Tensor]
) -> None:
	"""
	Load trainable parameters from ``state`` into model.
	"""
	target = unwrap(model)
	target.load_state_dict(state, strict=False)
