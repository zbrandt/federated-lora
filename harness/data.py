from __future__ import annotations

import numpy as np
from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizerBase

from harness.config import Config


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


def _is_article_heading(text: str) -> bool:
	# WikiText-2 raw marks article titles as " = Title = " (single '='); section
	# headings use two or more ("= = Section = ="), so exclude those.
	stripped = text.strip()
	return (
		stripped.startswith("= ")
		and stripped.endswith(" =")
		and not stripped.startswith("= =")
	)


def _assign_article_ids(dataset: Dataset, text_field: str) -> list[int]:
	# Walk the raw rows in order and give every row the id of the article it
	# belongs to. Rows before the first heading join article 0.
	article_ids: list[int] = []
	current_article = 0
	seen_heading = False
	for text in dataset[text_field]:
		if _is_article_heading(text):
			if seen_heading:
				current_article += 1
			seen_heading = True
		article_ids.append(current_article)
	return article_ids


def _partition_articles_iid(
	num_articles: int,
	num_clients: int,
	rng: np.random.Generator,
) -> list[int]:
	# Shuffle article order, then deal articles round-robin so every client
	# gets an equal share of (randomly chosen) whole articles.
	article_to_client = [0] * num_articles
	for position, article_id in enumerate(rng.permutation(num_articles)):
		article_to_client[int(article_id)] = position % num_clients
	return article_to_client


def _partition_articles_noniid(
	num_articles: int,
	num_clients: int,
	alpha: float,
	rng: np.random.Generator,
) -> list[int]:
	# Dirichlet skew over whole articles: each client's shard differs both in
	# size and in topical content, which is what induces divergent local updates.
	if alpha <= 0:
		raise ValueError("dirichlet_alpha must be positive")

	proportions = rng.dirichlet(np.full(num_clients, alpha))
	article_to_client = [0] * num_articles
	shuffled_articles = [int(article_id) for article_id in rng.permutation(num_articles)]

	# Guarantee every client at least one article, then sample the rest.
	for client_id in range(num_clients):
		article_to_client[shuffled_articles[client_id]] = client_id
	for article_id in shuffled_articles[num_clients:]:
		article_to_client[article_id] = int(rng.choice(num_clients, p=proportions))
	return article_to_client


def load_datasets(
	config: Config,
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

	# Partition at the article level BEFORE any shuffling, so clients hold
	# whole documents and genuinely different data distributions. Shuffling
	# first would launder away the heterogeneity we are trying to induce.
	article_ids = _assign_article_ids(train_dataset, config.text_field)
	num_articles = max(article_ids) + 1 if article_ids else 0
	if num_articles < config.num_clients:
		raise ValueError(
			f"only {num_articles} articles in split {config.train_split!r} for "
			f"{config.num_clients} clients; use a larger --train-split"
		)

	rng = np.random.default_rng(config.seed)
	if config.partition_strategy == "noniid":
		article_to_client = _partition_articles_noniid(
			num_articles,
			config.num_clients,
			config.dirichlet_alpha,
			rng,
		)
	else:
		article_to_client = _partition_articles_iid(
			num_articles,
			config.num_clients,
			rng,
		)

	# Collect the raw row indices belonging to each client's articles.
	client_row_indices: list[list[int]] = [[] for _ in range(config.num_clients)]
	for row_index, article_id in enumerate(article_ids):
		client_row_indices[article_to_client[article_id]].append(row_index)

	# Filter, tokenize, and shuffle WITHIN each shard only.
	train_shards: list[Dataset] = []
	for client_id, row_indices in enumerate(client_row_indices):
		shard = train_dataset.select(row_indices)
		shard = _filter_empty_text(shard, config.text_field)
		shard = _tokenize_dataset(shard, tokenizer, config.max_length, config.text_field)
		train_shards.append(shard.shuffle(seed=config.seed + client_id))

	eval_dataset = _filter_empty_text(eval_dataset, config.text_field)
	tokenized_eval = _tokenize_dataset(
		eval_dataset,
		tokenizer,
		config.max_length,
		config.text_field,
	)

	# Log the partition so per-seed variance in results can be attributed to
	# the article draw (article and example counts per client).
	articles_per_client = [0] * config.num_clients
	for article_id in range(num_articles):
		articles_per_client[article_to_client[article_id]] += 1
	summary = ", ".join(
		f"client {client_id}: {articles_per_client[client_id]} articles / "
		f"{len(shard)} examples"
		for client_id, shard in enumerate(train_shards)
	)
	print(f"[partition {config.partition_strategy}] {summary}")

	return train_shards, tokenized_eval
