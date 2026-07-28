from __future__ import annotations

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForSequenceClassification

from federated_lora.config import Config


def create_peft_model(config: Config, device: torch.device) -> PeftModel:
	"""
	Configures the base model for training with LoRA.

	Sets up the LoRA config and loads the base model to create a PEFT model.

	Parameters
	----------
	config : Config
	    TODO
	device : torch.device
	    The selected device type ("cpu" or "cude").

	Returns
	-------
	PeftModel
	    The in-place modified base model.
	"""
	peft_config = LoraConfig(
		task_type=config.task_type,
		r=config.lora_rank,
		target_modules=list(config.target_modules),
		lora_alpha=config.lora_alpha,
		lora_dropout=config.lora_dropout,
		modules_to_save=config.modules_to_save
		if config.train_classifier_head
		else None,
	)

	base = AutoModelForSequenceClassification.from_pretrained(
		config.model_name, num_labels=config.num_labels
	).to(device)

	return get_peft_model(base, peft_config)
