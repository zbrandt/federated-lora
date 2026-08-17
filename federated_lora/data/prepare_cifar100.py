from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor

from federated_lora.data import partition_dirichlet, provide_dataloader


def load_datasets(
	dataset: str, num_clients: int, alpha: float, task: str, seed: int
) -> Path:
	output_dir = Path('data') / f'{task}_{num_clients}_clients_alpha_{alpha}'
	if output_dir.is_dir():
		print(f'{dataset} data already saved to {output_dir.resolve()}')
		return output_dir

	dataset = load_dataset(dataset)

	train, test = dataset['train'], dataset['test']

	X_train = np.stack([np.array(im) for im in train['img']]).astype(np.uint8)
	y_train = np.array(train['fine_label'], dtype=np.int64)

	X_test = np.stack([np.array(im) for im in test['img']]).astype(np.uint8)
	y_test = np.array(test['fine_label'], dtype=np.int64)

	partitions = partition_dirichlet(
		labels=y_train.tolist(),
		num_clients=num_clients,
		alpha=alpha,
		seed=seed,
	)

	output_dir.mkdir(parents=True, exist_ok=True)

	for i, shard in enumerate(partitions):
		images, labels = X_train[shard], y_train[shard]
		np.savez_compressed(
			output_dir / f'shard_{i}.npz', images=images, labels=labels
		)

		print(f'shard {i}: n={len(labels)}')

	np.savez_compressed(output_dir / 'test.npz', images=X_test, labels=y_test)

	print(f'{dataset} data saved to {output_dir.resolve()}')
	return output_dir


def prepare_dataloaders(
	model: str,
	data_dir: Path,
	num_clients: int,
	batch_size: int,
	task: str,
	eval_batch_size: int = 256,
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
		shard = np.load(data_dir / f'shard_{i}.npz')
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
		batch_size=eval_batch_size,
		shuffle=False,
		collate_fn=collate,
		num_workers=0,
		pin_memory=pin_memory,
	)

	return train_dataloaders, test_dataloader
