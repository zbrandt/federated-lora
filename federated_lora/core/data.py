from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset

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


# TODO: refactor as load_datasets
def build_cache(split, processor):
	""" """
	n = len(split)
	cache = torch.empty((n, 3, 224, 224), dtype=torch.uint8)
	chunk = 512
	for i in range(0, n, chunk):
		rows = split[i : i + chunk]
		images = [im.convert('RGB') for im in rows['img']]
		pixels = processor(
			images,
			do_rescale=False,
			do_normalize=False,
			return_tensors='pt',
		)['pixel_values']
		cache[i : i + chunk] = pixels.round_().clamp_(0, 255).to(torch.uint8)
	labels = torch.tensor(split['fine_label'], dtype=torch.long)

	return TensorDataset(cache, labels)
