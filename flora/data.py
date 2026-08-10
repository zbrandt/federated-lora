"""
This file contains functions for loading and partitioning datasets for federated learning experiments. 
It includes functionality to tokenize datasets using a specified tokenizer, partition the training data among clients using either a non-iid Dirichlet distribution or an iid strategy, and return the resulting client-specific training shards along with a shared evaluation dataset. 
The partitioning strategy and other parameters are configurable via the provided Config dataclass.
"""

from __future__ import annotations

import numpy as np
from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizerBase

from flora.config import Config


def _tokenize(dataset: Dataset, tokenizer: PreTrainedTokenizerBase, config: Config) -> Dataset:
	field_a, field_b = config.text_fields

	def encode(batch):
		pair = (batch[field_a],) if field_b is None else (batch[field_a], batch[field_b])
		out = tokenizer(*pair, truncation=True, max_length=config.max_length)
		out["labels"] = batch["label"]
		return out

	# Tokenize the dataset using the provided tokenizer, removing original columns and keeping it in memory for faster access.
	return dataset.map(encode, batched=True, remove_columns=dataset.column_names, keep_in_memory=True)

# Partitions the training dataset into shards for each client using a Dirichlet distribution to create non-iid splits. 
#Each class's samples are distributed among clients according to proportions drawn from a Dirichlet distribution with parameter alpha. 
#The resulting indices for each client are shuffled to ensure randomness.
def _partition_dirichlet(labels: list[int], num_clients: int, alpha: float, rng: np.random.Generator) -> list[list[int]]:
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

# Loads the specified dataset and partitions it into training shards for each client and a shared evaluation dataset.
# The training dataset is partitioned according to the specified strategy (iid or non-iid Dirichlet) and tokenized using the provided tokenizer. The evaluation dataset is also tokenized.
# parameters: config: Config - The hyperparameter configuration from config.py. tokenizer: PreTrainedTokenizerBase - The tokenizer to use for tokenizing the datasets.
# returns: tuple[list[Dataset], Dataset] - A list of tokenized training shards for each client and a tokenized evaluation dataset.
def load_datasets(config: Config, tokenizer: PreTrainedTokenizerBase) -> tuple[list[Dataset], Dataset]:
	train = load_dataset(config.dataset_name, config.dataset_task, split=config.train_split)
	eval = load_dataset(config.dataset_name, config.dataset_task, split=config.eval_split)

	rng = np.random.default_rng(config.seed)

	if config.partition_strategy == "noniid":
		indices = _partition_dirichlet(train["label"], config.num_clients, config.dirichlet_alpha, rng)
	else:
		order = rng.permutation(len(train))
		indices = [order[i::config.num_clients].tolist() for i in range(config.num_clients)]

	train_shards = [_tokenize(train.select(idx), tokenizer, config) for idx in indices]
	eval_dataset = _tokenize(eval, tokenizer, config)

	return train_shards, eval_dataset
