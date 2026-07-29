from __future__ import annotations

import torch
from peft import LoraConfig, PeftModel, get_peft_model, TaskType
from transformers import RobertaForSequenceClassification

from federated_lora.config import Config


def create_peft_model(config: Config, device: torch.device) -> PeftModel:
	"""
	Configures the base model for training with LoRA.

	Sets up the LoRA config and loads the RoBERTa model transformer with a 
	sequence classification/regression head on top to create a PEFT model.

	Parameters
	----------
	config : Config
	    TODO
	device : torch.device
	    The selected device type ("cpu" or "cuda").

	Returns
	-------
	PeftModel
	    The in-place modified base model.
	"""
	peft_config = LoraConfig(
		task_type=TaskType.SEQ_CLS,
		r=config.lora_rank,
		target_modules=list(config.target_modules),
		lora_alpha=config.lora_alpha,
		lora_dropout=config.lora_dropout,
		modules_to_save=config.modules_to_save
		if config.train_classifier_head
		else None,
	)

	base = RobertaForSequenceClassification.from_pretrained(
		"FacebookAI/roberta-base", num_labels=config.num_labels
	).to(device)

	return get_peft_model(base, peft_config)
