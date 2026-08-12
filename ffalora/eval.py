# Evaluate a model's loss and accuracy on a dataset.
from __future__ import annotations

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, PreTrainedModel, PreTrainedTokenizerBase

from ffalora.config import Config


# Run the model over a dataset in batches and return its average loss and accuracy.
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
