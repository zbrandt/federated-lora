from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader


def partition_dirichlet(
	labels: list[int], num_clients: int, alpha: float, seed: int
) -> list[list[int]]:
	"""
	Partition dataset indices into non-iid shards from a Dirichlet distribution.

	Parameters
	----------
	labels : list[int]
		The class labels.
	num_clients : int
		The total number of clients.
	alpha : float
		The parameter of the Dirichlet distribution.
	seed : int
		The starting number used to initialize a random seed generator.

	Returns
	-------
	list[list[int]]
		The non-iid client indices.
	"""
	rng = np.random.default_rng(seed)
	labels = np.array(labels)
	unique_classes = np.unique(labels)

	client_indices = [[] for _ in range(num_clients)]

	for c in unique_classes:
		idx = np.where(labels == c)[0]
		rng.shuffle(idx)

		# draw samples from the Dirichlet distribution
		# repeat alpha num_clients times
		proportions = rng.dirichlet(np.repeat(alpha, num_clients))

		splits = np.round((np.cumsum(proportions) * len(idx))).astype(int)[:-1]
		client_splits = np.split(idx, splits)

		for client_id, split in enumerate(client_splits):
			client_indices[client_id].extend(split.tolist())

	return client_indices


def cycle(dataloader: DataLoader) -> Iterable:
	while True:
		yield from dataloader


def load_datasets(name: str, num_clients: int, alpha: float, seed: int):
	"""
	Load the dataset and save train shards and test split.

	Parameters
	----------
	name : str
		The name of the dataset.
	num_clients : int
		The total number of clients.
	alpha : float
		The parameter of the Dirichlet distribution.
	seed : int
		The starting number used to initialize a random seed generator.
	"""
	output_dir = Path(f'cifar100_{num_clients}_clients_alpha_{alpha}')
	if output_dir.is_dir():
		print(f'{name} data already saved to {output_dir.resolve()}')
		return output_dir

	dataset = load_dataset(name)

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

	print(f'{name} data saved to {output_dir.resolve()}')
	return output_dir
