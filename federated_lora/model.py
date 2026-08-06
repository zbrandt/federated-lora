from __future__ import annotations

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForImageClassification

from federated_lora.config import Config


def create_peft_model(
	model: str,
	num_labels: int,
	lora_rank: int,
	lora_alpha: int,
	target_modules: tuple[str, ...],
	lora_dropout: float,
	device: torch.device,
) -> PeftModel:
	"""
	Create a trainable PeftModel from the base model and the parameters for how
	to configure the model for training with LoRA.

	Parameters
	----------
	model : str
		The model id of a pretrained model.
	num_labels : int
		TODO
	lora_rank : int
		The LoRA attention dimension (the "rank").
	target_modules : tuple[str, ...]
		The names of the modules to apply the adapter to.
	lora_alpha : int
		The alpha parameter for LoRA scaling.
	lora_dropout : int
		The dropout probability for LoRA layers.
	device : torch.device
		The selected device type ("cpu" or "cuda").

	Returns
	-------
	PeftModel
		The PEFT model object from the in-place modified model and LoRA config.
	"""
	base = AutoModelForImageClassification.from_pretrained(
		model,
		num_labels=num_labels,
		ignore_mismatched_sizes=True,  # TODO
	).to(device)

	peft_config = LoraConfig(
		r=lora_rank,
		lora_alpha=lora_alpha,
		target_modules=target_modules,
		lora_dropout=lora_dropout,
		modules_to_save=['classifier'],
	)

	model = get_peft_model(base, peft_config)

	return model
