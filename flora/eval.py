"""
This file contains functions for evaluating the performance of a model on a given dataset.
Computes average loss and accuracy over the dataset using a specified model, tokenizer, and configuration. 
The evaluation is performed in batches, with the model set to evaluation mode and gradients disabled for efficiency.
"""

from __future__ import annotations

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, PreTrainedModel, PreTrainedTokenizerBase

from flora.config import Config

# Evaluates the accuracy and average loss of a model on a given dataset.
# parameters: model: PreTrainedModel - The model to evaluate. 
# dataset: Dataset - The dataset to evaluate on. 
# tokenizer: PreTrainedTokenizerBase - The tokenizer used for padding and batching. 
# config: Config - The hyperparameter configuration from config.py. 
# device: torch.device - The device to run the evaluation on (CPU or GPU).
# returns: tuple[float, float] - A tuple containing the average loss and accuracy over the dataset.
def evaluate_accuracy(
	model: PreTrainedModel,
	dataset: Dataset,
	tokenizer: PreTrainedTokenizerBase,
	config: Config,
	device: torch.device,
) -> tuple[float, float]:
	collator = DataCollatorWithPadding(tokenizer=tokenizer)
	loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collator)

	model.eval()
	total_loss = 0.0
	correct = 0
	total = 0
	with torch.no_grad():
		for batch in loader:
			batch = {key: value.to(device) for key, value in batch.items()}
			out = model(**batch)
			n = batch["labels"].size(0)
			total_loss += float(out.loss.item()) * n
			correct += int((out.logits.argmax(dim=-1) == batch["labels"]).sum().item())
			total += n

	return total_loss / max(1, total), correct / max(1, total)
