from __future__ import annotations

from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizerBase

from config import HarnessConfig


def _filter_empty_text(dataset: Dataset, text_field: str) -> Dataset:
	# Remove blank rows before tokenization so we do not waste client steps.
	return dataset.filter(lambda row: bool(row[text_field].strip()))


def _tokenize_dataset(
	dataset: Dataset,
	tokenizer: PreTrainedTokenizerBase,
	max_length: int,
	text_field: str,
) -> Dataset:
	# Tokenize the text column into GPT-2-sized language-model examples.
	def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
		return tokenizer(batch[text_field], truncation=True, max_length=max_length)

	return dataset.map(tokenize_batch, batched=True, remove_columns=dataset.column_names)


def _split_dataset(dataset: Dataset, num_clients: int) -> list[Dataset]:
	# Use contiguous shards for the first pass; this keeps the harness simple.
	if num_clients < 1:
		raise ValueError("num_clients must be at least 1")
	if len(dataset) < num_clients:
		raise ValueError("dataset is too small for the requested number of clients")

	shard_sizes = [len(dataset) // num_clients] * num_clients
	for shard_index in range(len(dataset) % num_clients):
		shard_sizes[shard_index] += 1

	shards: list[Dataset] = []
	start_index = 0
	for shard_size in shard_sizes:
		stop_index = start_index + shard_size
		shards.append(dataset.select(range(start_index, stop_index)))
		start_index = stop_index
	return shards


def load_datasets(
	config: HarnessConfig,
	tokenizer: PreTrainedTokenizerBase,
) -> tuple[list[Dataset], Dataset]:
	# Load the same corpus split used by the original single-node prototype.
	train_dataset = load_dataset(
		config.dataset_name,
		config.dataset_config,
		split=config.train_split,
	)
	eval_dataset = load_dataset(
		config.dataset_name,
		config.dataset_config,
		split=config.eval_split,
	)

	train_dataset = _filter_empty_text(train_dataset, config.text_field)
	eval_dataset = _filter_empty_text(eval_dataset, config.text_field)

	tokenized_train = _tokenize_dataset(
		train_dataset,
		tokenizer,
		config.max_length,
		config.text_field,
	).shuffle(seed=config.seed)
	tokenized_eval = _tokenize_dataset(
		eval_dataset,
		tokenizer,
		config.max_length,
		config.text_field,
	)

	# Return one shard per client plus a shared evaluation set.
	return _split_dataset(tokenized_train, config.num_clients), tokenized_eval
