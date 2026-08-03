from __future__ import annotations

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch import nn
from transformers import AutoModelForImageClassification

from federated_lora.config import Config


def create_peft_model(config: Config, device: torch.device) -> PeftModel:
	"""
	Configure a Vision Transformer for LoRA fine-tuning on image classification.

	Loads a pre-trained ViT backbone with a fresh classification head sized to
	the target dataset, then attaches LoRA adapters to the attention query/value
	projections. ``modules_to_save=['classifier']`` keeps the classification head
	trainable *and* part of the federated (aggregated) state, matching the
	paper's vision setting, which updates both the adapter and the head so the
	head can adapt to the target label space. Everything else in the backbone is
	frozen by ``get_peft_model``.

	Parameters
	----------
	config : Config
		Run configuration (model name, num_labels, LoRA rank/alpha/dropout,
		target modules).
	device : torch.device
		The selected device type ("cpu" or "cuda").

	Returns
	-------
	PeftModel
		The frozen backbone with trainable LoRA adapters and classification head.
	"""
	base = AutoModelForImageClassification.from_pretrained(
		config.model,
		num_labels=config.num_labels,
		ignore_mismatched_sizes=True, # TODO: replace pre-trained head with different label count
	).to(device)

	peft_config = LoraConfig(
		r=config.lora_rank,
		target_modules=['q_proj','v_proj'],
		lora_alpha=config.lora_alpha,
		lora_dropout=config.lora_dropout,
		modules_to_save=['classifier'],
	)

	model = get_peft_model(base, peft_config)

	return model
