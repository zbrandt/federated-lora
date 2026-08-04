from __future__ import annotations

import numpy as np
import torch

from federated_lora.config import Config


def partition_dirichlet(
	labels: list[int], num_clients: int, alpha: float, seed: int
) -> list[list[int]]:
	"""
	Partition dataset indices into non-iid shards from a Dirichlet distribution.

	Parameters
	----------
	labels : list[int]

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
	num_classes = len(np.unique(labels))

	class_indices = [np.where(labels == c)[0] for c in range(num_classes)]
	client_indices = [[] for _ in range(num_clients)]

	for c in range(num_classes):
		idx = class_indices[c]
		rng.shuffle(idx)

		# draw samples from the Dirichlet distribution
		# repeat alpha num_clients times
		proportions = rng.dirichlet(np.repeat(alpha, num_clients))

		splits = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
		client_splits = np.split(idx, splits)

		for client_id, split in enumerate(client_splits):
			client_indices[client_id].extend(split.tolist())

	return client_indices
