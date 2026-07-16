from __future__ import annotations

import math

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling, PreTrainedModel, PreTrainedTokenizerBase

from config import HarnessConfig


def evaluate_perplexity(
	model: PreTrainedModel,
	dataset: Dataset,
	tokenizer: PreTrainedTokenizerBase,
	config: HarnessConfig,
	device: torch.device,
) -> tuple[float, float]:
	# Use the same collator as training so the eval metric is comparable.
	collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
	data_loader = DataLoader(
		dataset,
		batch_size=config.batch_size,
		shuffle=False,
		collate_fn=collator,
	)

	model.eval()
	total_loss = 0.0
	total_tokens = 0
	with torch.no_grad():
		# Accumulate token-weighted loss, then convert to perplexity.
		for batch in data_loader:
			batch = {key: value.to(device) for key, value in batch.items()}
			loss = model(**batch).loss
			token_count = int(batch["attention_mask"].sum().item())
			total_loss += float(loss.item()) * token_count
			total_tokens += token_count

	mean_loss = total_loss / max(1, total_tokens)
	return mean_loss, math.exp(mean_loss)
