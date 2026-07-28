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
from federated_lora.data import load_datasets
from federated_lora.model import create_peft_model
from federated_lora.client import Client
from federated_lora.server import Server
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

	tokenizer = AutoTokenizer.from_pretrained(config.model_name)

	train_shards, eval_dataset = load_datasets(config, tokenizer)

	method = get_method(config.method)

	model = create_peft_model(config, device)

	collator = DataCollatorWithPadding(tokenizer=tokenizer)

	clients = [
		Client(i, shard, config, device, collator)
		for i, shard in enumerate(train_shards)
	]

	return Server(
		config, model, method, clients, eval_dataset, tokenizer, device, 
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

	if config.output:
		path = Path(config.output)
	else:
		path = (
			Path(config.results_dir)
			/ f'{config.method}_{config.dataset_task}_seed{config.seed}.json'
		)

	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(result, indent=2), encoding='utf-8')
	return result
