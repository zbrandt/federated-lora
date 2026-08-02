from __future__ import annotations

import numpy as np
from datasets import Dataset, load_dataset

from federated_lora.config import Config


# TODO
def _make_transform(config: Config, processor):
	"""
	Build an on-the-fly transform that turns raw PIL images into pixel tensors
	using the model's own image processor (exact resize/crop/normalization).

	Applied lazily per batch (via ``with_transform``) so the full image set is
	never materialized in memory.
	"""
	image_field = config.image_field
	label_field = config.label_field

	def transform(batch):
		images = [img.convert('RGB') for img in batch[image_field]]
		pixel_values = processor(images, return_tensors='pt')['pixel_values']
		return {'pixel_values': pixel_values, 'labels': batch[label_field]}

	return transform

# TODO
def _partition_dirichlet(
	labels: list[int], num_clients: int, alpha: float, rng: np.random.Generator
) -> list[list[int]]:
	"""
	"""
	labels = np.asarray(labels)
	client_indices: list[list[int]] = [[] for _ in range(num_clients)]
	for cls in np.unique(labels):
		idx = np.where(labels == cls)[0]
		rng.shuffle(idx)
		proportions = rng.dirichlet(np.full(num_clients, alpha))
		cuts = (np.cumsum(proportions)[:-1] * len(idx)).astype(int)
		for client_id, shard in enumerate(np.split(idx, cuts)):
			client_indices[client_id].extend(shard.tolist())
	for shard in client_indices:
		rng.shuffle(shard)
	return client_indices

# TODO
def load_datasets(config: Config, processor) -> tuple[list[Dataset], Dataset]:
	""" 
	"""
	train = load_dataset(config.dataset_id, split=config.train_split)
	eval = load_dataset(config.dataset_id, split=config.eval_split)

	rng = np.random.default_rng(config.seed)

	if config.partition_strategy == 'noniid':
		indices = _partition_dirichlet(
			train[config.label_field],
			config.num_clients,
			config.dirichlet_alpha,
			rng,
		)
	else:
		order = rng.permutation(len(train))
		indices = [
			order[i :: config.num_clients].tolist()
			for i in range(config.num_clients)
		]

	transform = _make_transform(config, processor)

	train_shards = [
		train.select(idx).with_transform(transform) for idx in indices
	]
	eval_dataset = eval.with_transform(transform)

	return train_shards, eval_dataset
