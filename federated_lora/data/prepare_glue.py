from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from federated_lora.data import partition_dirichlet, provide_dataloader


def load_datasets(
	dataset: str, num_clients: int, alpha: float, task: str, seed: int
) -> Path:
	output_dir = Path('data') / f'{task}_{num_clients}_clients_alpha_{alpha}'
	if output_dir.is_dir():
		print(f'{dataset} data already saved to {output_dir.resolve()}')
		return output_dir

	dataset = load_dataset(dataset, 'sst2')

	train = dataset['train']
	y_train = np.array(train['label'], dtype=np.int64)
	# unicode arrays (not object/pickle) so shards load without allow_pickle
	train_fields = {f: np.array(train[f]) for f in ('sentence',)}

	partitions = partition_dirichlet(
		labels=y_train.tolist(),
		num_clients=num_clients,
		alpha=alpha,
		seed=seed,
	)

	output_dir.mkdir(parents=True, exist_ok=True)

	for i, shard in enumerate(partitions):
		shard = np.array(shard, dtype=np.int64)
		np.savez_compressed(
			output_dir / f'shard_{i}.npz',
			labels=y_train[shard],
			**{f: train_fields[f][shard] for f in ('sentence',)},
		)
		print(f'shard {i}: n={len(shard)}')

	# GLUE test labels are hidden; eval_split is 'validation' for glue tasks
	ev = dataset['validation']
	np.savez_compressed(
		output_dir / 'test.npz',
		labels=np.array(ev['label'], dtype=np.int64),
		**{f: np.array(ev[f]) for f in ('sentence',)},
	)

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
	tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
	pin_memory = torch.cuda.is_available()
	fields = ('sentence',)

	def collate(batch):
		labels = [item[1] for item in batch]
		texts = [[item[0][f] for item in batch] for f in fields]
		# one or two text columns (single-sentence vs sentence-pair tasks)
		enc = tokenizer(
			*texts,
			padding=True,
			truncation=True,
			max_length=128,
			return_tensors='pt',
		)
		return {
			'input_ids': enc['input_ids'],
			'attention_mask': enc['attention_mask'],
			'labels': torch.tensor(labels, dtype=torch.long),
		}

	def examples_from(shard) -> list[tuple[dict[str, str], int]]:
		labels = shard['labels']
		cols = {f: shard[f] for f in fields}
		return [
			({f: str(cols[f][j]) for f in fields}, int(labels[j]))
			for j in range(len(labels))
		]

	train_dataloaders = []
	for i in range(num_clients):
		shard = np.load(data_dir / f'shard_{i}.npz', allow_pickle=False)
		dataset = examples_from(shard)

		if not dataset:
			print(f'WARNING: {task} client {i} shard is empty — skipping')
			continue

		train_dataloaders.append(
			provide_dataloader(dataset, batch_size, collate, pin_memory)
		)

	test = np.load(data_dir / 'test.npz', allow_pickle=False)
	test_dataloader = DataLoader(
		examples_from(test),
		batch_size=eval_batch_size,
		shuffle=False,
		collate_fn=collate,
		num_workers=0,
		pin_memory=pin_memory,
	)

	return train_dataloaders, test_dataloader
