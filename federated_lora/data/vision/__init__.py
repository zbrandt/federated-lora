from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor

from federated_lora.data import provide_dataloader


def prepare_dataloaders(
	model: str,
	data_dir: Path,
	num_clients: int,
	batch_size: int,
	task: str,
) -> tuple[list[DataLoader], DataLoader]:
	processor = AutoImageProcessor.from_pretrained(model, use_fast=True)
	pin_memory = torch.cuda.is_available()

	def collate(batch):
		images = [item[0] for item in batch]
		labels = [item[1] for item in batch]

		proc = processor(images=images, return_tensors='pt')

		return {
			'pixel_values': proc['pixel_values'],
			'labels': torch.tensor(labels, dtype=torch.long),
		}

	train_dataloaders = []
	for i in range(num_clients):
		shard = np.load(data_dir / f'client_{i}.npz')
		dataset = list(zip(shard['images'], shard['labels'], strict=False))

		if not dataset:
			print(f'WARNING: {task} client {i} shard is empty — skipping')
			continue

		train_dataloaders.append(
			provide_dataloader(dataset, batch_size, collate, pin_memory)
		)

	test = np.load(data_dir / 'test.npz')
	test_dataset = list(zip(test['images'], test['labels'], strict=False))
	test_dataloader = DataLoader(
		test_dataset,
		batch_size=256,
		shuffle=False,
		collate_fn=collate,
		num_workers=0,
		pin_memory=pin_memory,
	)

	return train_dataloaders, test_dataloader
