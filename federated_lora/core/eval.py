from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedModel


def evaluate(
	model: PreTrainedModel,
	dataloader: DataLoader,
	device: torch.device,
) -> tuple[float, float]:
	""" """
	model.eval()
	criterion = nn.CrossEntropyLoss()

	total_loss = 0
	correct_top1 = 0
	correct_top5 = 0
	total_samples = 0

	with torch.inference_mode():
		for batch in dataloader:
			pixel_values = batch['pixel_values'].to(device, non_blocking=True)
			labels = batch['labels'].to(device, non_blocking=True)

			outputs = model(pixel_values=pixel_values)
			logits = outputs.logits

			loss = criterion(logits, labels)
			total_loss = loss.item() * pixel_values.size(0)

			# calculate top-1 accuracy
			top1_preds = logits.argmax(dim=-1)
			correct_top1 += (top1_preds == labels).sum().item()

			# calculate top-5 accuracy
			_, top5_preds = logits.topk(5, dim=-1)
			correct_top5 += (
				torch.eq(top5_preds, labels.view(-1, 1)).sum().item()
			)

			total_samples += labels.size(0)

	avg_loss = total_loss / total_samples
	top1_acc = (correct_top1 / total_samples) * 100.0
	top5_acc = (correct_top5 / total_samples) * 100.0

	return avg_loss, top1_acc, top5_acc
