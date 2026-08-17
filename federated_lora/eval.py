from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import PreTrainedModel


def evaluate(
	model: PreTrainedModel,
	dataloader: DataLoader,
	device: torch.device,
	top_k: int = 5,
) -> tuple[float, float, float]:
	"""
	Evaluate the model.

	Modality-agnostic: every non-``labels`` key in a batch is forwarded to the
	model, so this works for both vision batches (``pixel_values``) and text
	batches (``input_ids`` + ``attention_mask``). The reported "top-5" accuracy
	is a top-``min(top_k, num_labels)`` accuracy, so it degrades gracefully for
	tasks with fewer than five classes (e.g. SST-2 reports top-2 == 100%).

	Parameters
	----------
	model : PreTrainedModel
		The model to evaluate.
	dataloader : DataLoader
		The evaluation data loader.
	device : torch.device
		The selected device type ("cpu" or "cuda").
	top_k : int
		The k for the secondary top-k accuracy (capped at the number of labels).

	Returns
	-------
	tuple[float, float, float]
		A tuple with the average loss, top-1 accuracy, and top-k accuracy.
	"""
	model.eval()
	criterion = nn.CrossEntropyLoss()

	total_loss = 0
	correct_top1 = 0
	correct_topk = 0
	total_samples = 0

	# disable gradient calculation for inference
	# use bfloat16 on cuda
	with (
		torch.inference_mode(),
		torch.autocast(
			'cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'
		),
	):
		for batch in dataloader:
			labels = batch['labels'].to(device, non_blocking=True)
			inputs = {
				key: value.to(device, non_blocking=True)
				for key, value in batch.items()
				if key != 'labels'
			}

			outputs = model(**inputs)
			logits = outputs.logits

			loss = criterion(logits, labels)
			total_loss += loss.item() * labels.size(0)

			# calculate top-1 accuracy
			top1_preds = logits.argmax(dim=-1)
			correct_top1 += (top1_preds == labels).sum().item()

			# calculate top-k accuracy (k capped at the number of classes)
			k = min(top_k, logits.size(-1))
			_, topk_preds = logits.topk(k, dim=-1)
			correct_topk += (
				torch.eq(topk_preds, labels.view(-1, 1)).sum().item()
			)

			total_samples += labels.size(0)

	avg_loss = total_loss / total_samples
	top1_acc = (correct_top1 / total_samples) * 100.0
	topk_acc = (correct_topk / total_samples) * 100.0

	return avg_loss, top1_acc, topk_acc
