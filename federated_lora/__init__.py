from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch
from opacus.accountants.utils import get_noise_multiplier
from torch.utils.data import DataLoader
from transformers import (
	AutoTokenizer,
	DataCollatorWithPadding,
)

from federated_lora.config import Config
from federated_lora.core.client import Client
from federated_lora.core.data import load_datasets
from federated_lora.core.model import create_peft_model
from federated_lora.core.server import Server
from federated_lora.privacy.accountant import achieved_epsilon
from federated_lora.registry import get_method


def build(config: Config) -> Server:
	"""
	Build the federated LoRA project server and clients.

	Parameters
	----------
	config : Config
		TODO

	Returns
	-------
	Server
		TODO
	"""
	# set the device type
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# set the seed for generating random numbers on all devices
	generator = torch.manual_seed(config.seed)

	# TODO
	tokenizer = AutoTokenizer.from_pretrained('FacebookAI/roberta-base')

	# TODO
	train_shards, eval_dataset = load_datasets(config, tokenizer)

	shard_size = sum(len(s) for s in train_shards) / len(train_shards)
	sample_rate = config.batch_size / shard_size
	# steps is the number of noised gradient releases that touch data — one per
	# local step, i.e. ``rounds * local_steps`` (both A- and B-steps consume a
	# batch, so both count)
	steps = config.rounds * config.local_steps
	config.privacy.noise_multiplier = get_noise_multiplier(
		target_epsilon=config.privacy.target_epsilon,
		target_delta=config.privacy.target_delta,
		sample_rate=sample_rate,
		epochs=steps,
	)

	# TODO
	method = get_method(config.method)

	# TODO
	model = create_peft_model(config, device)

	# TODO
	collator = DataCollatorWithPadding(tokenizer=tokenizer)

	# TODO
	clients = []
	for i, shard in enumerate(train_shards):
		generator = torch.Generator().manual_seed(config.seed + i)

		train_dataloader = DataLoader(
			shard,
			config.batch_size,
			shuffle=True,
			collate_fn=collator,
			generator=generator,
		)
		clients.append(Client(i, shard, train_dataloader))

	return Server(
		config,
		model,
		method,
		clients,
		eval_dataset,
		tokenizer,
		device,
		generator,
	)


def run(argv: list[str] | None = None) -> dict:
	"""
	Run the LA-LoRA method.

	Get the configuration of the hyperparameters from argument parsers,
	instantiate and run the server from the configuration. Saves results to
	specified or default result directory indexed by method name (label only),
	GLUE benchmark task, and seed.

	Parameters
	----------
	argv : list[str] | None
		List of arguments from the user.

	Returns
	-------
	dict
		result dictionary of configured hyperparameters as well as training and
		evaluation history.
	"""
	config = Config.from_argv(argv)
	server = build(config)
	history = [asdict(metric) for metric in server.run()]
	result = {'config': asdict(config), 'history': history}

	# Record the epsilon actually spent so accuracy can be plotted against the
	# privacy budget (matches the calibration q and step count from build()).
	client_dataset_size = sum(
		len(client.train_shard) for client in server.clients
	) / len(server.clients)
	sample_rate = config.batch_size / client_dataset_size
	steps = config.rounds * config.local_steps
	result['achieved_epsilon'] = achieved_epsilon(
		config.privacy.noise_multiplier,
		sample_rate,
		steps,
		config.privacy.target_delta,
	)

	if config.output:
		path = Path(config.output)
	else:
		path = (
			Path(config.results_dir)
			/ f'{config.method}_{config.task}_seed{config.seed}.json'
		)

	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(result, indent=2), encoding='utf-8')
	return result
