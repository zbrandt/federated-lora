from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
from datasets import load_dataset
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


# TODO: add parameter descriptions
def load_datasets(
	dataset: str,
	num_clients: int,
	alpha: float,
	task: str,
	experiment: str,
	seed: int,
	label_field: str,
	image_field: str,
	eval_split: str,
) -> Path:
	"""
	Load the dataset and save train shards and test split.

	Parameters
	----------
	dataset : str
		The name of the dataset.
	num_clients : int
		The total number of clients.
	alpha : float
		The parameter of the Dirichlet distribution.
	seed : int
		The starting number used to initialize a random seed generator.

	Returns
	-------
	Path
		The directory holding ``shard_{i}.npz`` per client and ``test.npz``.
	"""
	if experiment == 'text':
		from federated_lora.data.prepare_glue import load_datasets

		return load_datasets(
			dataset=dataset,
			num_clients=num_clients,
			alpha=alpha,
			task=task,
			seed=seed,
		)

	from federated_lora.data.prepare_vision import load_datasets

	return load_datasets(
		dataset=dataset,
		num_clients=num_clients,
		alpha=alpha,
		task=task,
		seed=seed,
		label_field=label_field,
		image_field=image_field,
		eval_split=eval_split,
	)


def prepare_dataloaders(
	model: str,
	data_dir: Path,
	num_clients: int,
	batch_size: int,
	task: str,
	experiment: str,
	eval_batch_size: int = 256,
) -> tuple[list[DataLoader], DataLoader]:
	"""
	Build one DP (Poisson-sampled) train loader per client plus a single test
	loader, all sharing one image processor.

	The image processor is loaded once here and bound into a single ``collate``
	closure, so every loader shares the same stable reference. Train loaders use
	Opacus' ``UniformWithReplacementSampler`` (Poisson sampling at rate
	``batch_size / n``) as required for DP-SGD so they can be wrapped by
	``make_private``; the test loader uses fixed-size, unshuffled batches.

	Parameters
	----------
	model : str
		The model id whose image processor collates the raw images.
	data_dir : Path
		The directory holding ``shard_{i}.npz`` per client and ``test.npz``.
	num_clients : int
		The number of client shards to load.
	batch_size : int
		The train lot size (also the Poisson sampling target).
	task : str
		TODO
	experiment : str
		TODO
	eval_batch_size : int
		The fixed batch size for the test loader.

	Returns
	-------
	tuple[list[DataLoader], DataLoader]
		The per-client train loaders and the shared test loader.
	"""
	if experiment == 'text':
		from federated_lora.data.prepare_glue import prepare_dataloaders
	else:
		from federated_lora.data.prepare_vision import prepare_dataloaders

	return prepare_dataloaders(
		model=model,
		data_dir=data_dir,
		num_clients=num_clients,
		batch_size=batch_size,
		task=task,
		eval_batch_size=eval_batch_size,
	)
