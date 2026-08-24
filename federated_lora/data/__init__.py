from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from opacus.utils.uniform_sampler import UniformWithReplacementSampler
from torch.utils.data import DataLoader


def cycle(dataloader: DataLoader) -> Iterable:
	while True:
		yield from dataloader


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

		splits = np.round(np.cumsum(proportions) * len(idx)).astype(int)[:-1]
		client_splits = np.split(idx, splits)

		for client_id, split in enumerate(client_splits):
			client_indices[client_id].extend(split.tolist())

	return client_indices


def provide_dataloader(dataset, batch_size, collate, pin_memory) -> DataLoader:
	"""
	Wrap a dataset in a Poisson-sampled loader for Opacus make_private.
	"""
	n = len(dataset)
	sample_rate = min(1.0, batch_size / max(1, n))
	batch_sampler = UniformWithReplacementSampler(
		num_samples=n,
		sample_rate=sample_rate,
	)
	return DataLoader(
		dataset,
		batch_sampler=batch_sampler,
		collate_fn=collate,
		num_workers=0,
		pin_memory=pin_memory,
	)
