from __future__ import annotations

import numpy as np
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from federated_lora.config import Config


def _tokenize(
	dataset: Dataset, tokenizer: PreTrainedTokenizerBase, config: Config
) -> Dataset:
	""" """
	field_a, field_b = config.text_fields

	def encode(batch):
		pair = (
			(batch[field_a],)
			if field_b is None
			else (batch[field_a], batch[field_b])
		)
		out = tokenizer(
			*pair, truncation=True, max_length=config.max_length
		)  # TODO
		out['labels'] = batch['label']
		return out

	return dataset.map(
		encode, batched=True, remove_columns=dataset.column_names
	)  # TODO


def _partition_dirichlet(
	labels: list[int], num_clients: int, alpha: float, rng: np.random.Generator
) -> list[list[int]]:
	""" """
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


def load_datasets(
	config: Config, tokenizer: PreTrainedTokenizerBase
) -> tuple[list[Dataset], Dataset]:
	""" """
	train = load_dataset('nyu-mll/glue', config.task, split=config.train_split)
	eval = load_dataset('nyu-mll/glue', config.task, split=config.eval_split)

	rng = np.random.default_rng(config.seed)

	# TODO
	if config.partition_strategy == 'noniid':
		indices = _partition_dirichlet(
			train['label'], config.num_clients, config.dirichlet_alpha, rng
		)
	else:
		order = rng.permutation(len(train))
		indices = [
			order[i :: config.num_clients].tolist()
			for i in range(config.num_clients)
		]

	# TODO
	train_shards = [
		_tokenize(train.select(idx), tokenizer, config) for idx in indices
	]
	eval_dataset = _tokenize(eval, tokenizer, config)

	# summary = ", ".join(f"client {i}: {len(s)} ex" for i, s in enumerate(train_shards))
	# print(f"[partition {config.partition_strategy}] {summary}")

	return train_shards, eval_dataset
