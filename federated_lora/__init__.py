from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
	AutoModelForSequenceClassification,
	AutoTokenizer,
	DataCollatorWithPadding,
)

from federated_lora.config import Config
from federated_lora.core.client import Client
from federated_lora.core.data import load_datasets
from federated_lora.core.model import create_peft_model
from federated_lora.core.server import Server
from federated_lora.privacy.accountant import (
	achieved_epsilon,
	calibrate_noise_multiplier,
)
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

	tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")

	train_shards, eval_dataset = load_datasets(config, tokenizer)

	# Calibrate the DP noise to the privacy budget the way the paper does:
	# solve for the noise multiplier sigma so a subsampled Gaussian, composed
	# over rounds * local_steps local steps at data sampling rate
	# q = batch_size / client_dataset_size, spends exactly target_epsilon.
	if config.privacy.dp and config.privacy.target_epsilon is not None:
		client_dataset_size = sum(len(s) for s in train_shards) / len(
			train_shards
		)
		sample_rate = config.batch_size / client_dataset_size
		steps = config.rounds * config.local_steps
		config.privacy.noise_multiplier = calibrate_noise_multiplier(
			config.privacy.target_epsilon,
			config.privacy.target_delta,
			sample_rate,
			steps,
		)

	method = get_method(config.method)

	model = create_peft_model(config, device)

	collator = DataCollatorWithPadding(tokenizer=tokenizer)

	clients = [
		Client(i, shard, config, device, collator)
		for i, shard in enumerate(train_shards)
	]

	return Server(
		config,
		model,
		method,
		clients,
		eval_dataset,
		tokenizer,
		device,
		generator
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
	if config.privacy.dp and config.privacy.noise_multiplier is not None:
		client_dataset_size = sum(
			len(client.dataset) for client in server.clients
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
