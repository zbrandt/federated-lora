from __future__ import annotations

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import RobertaForSequenceClassification

from federated_lora.config import Config


def create_peft_model(config: Config, device: torch.device) -> PeftModel:
	"""
	Configures the base model for training with LoRA.

	Sets up the LoRA config and loads the RoBERTa model transformer with a
	sequence classification/regression head on top to create a PEFT model.
	Freezes all the parameters of the model.

	Parameters
	----------
	config : Config
		TODO
	device : torch.device
		The selected device type ("cpu" or "cuda").

	Returns
	-------
	PeftModel
		The in-place modified and frozen base model.
	"""
	peft_config = LoraConfig(
		task_type=TaskType.SEQ_CLS,
		r=config.lora_rank,
		target_modules=list(config.target_modules),
		lora_alpha=config.lora_alpha,
		lora_dropout=config.lora_dropout,
	)

	base = RobertaForSequenceClassification.from_pretrained(
		'FacebookAI/roberta-base', num_labels=config.num_labels
	).to(device)

	model = get_peft_model(base, peft_config)

	for name, parameter in model.named_parameters():
		if 'lora_A' not in name and 'lora_B' not in name:
			parameter.requires_grad = False

	return model
