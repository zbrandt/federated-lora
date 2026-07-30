from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedModel


def evaluate(
	model: PreTrainedModel,
	test_dataloader: DataLoader,
	device: torch.device,
) -> tuple[float, float]:
	""" """
	model.eval()

	total_loss = 0.0
	correct = 0
	total = 0
	for batch in test_dataloader:
		batch = {key: value.to(device) for key, value in batch.items()}

		with torch.no_grad():
			outputs = model(**batch)
			n = batch['labels'].size(0)
			total_loss += float(outputs.loss.item()) * n
			correct += int(
				(outputs.logits.argmax(dim=-1) == batch['labels']).sum().item()
			)
			total += n

	return total_loss / max(1, total), correct / max(1, total)
