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

from la_lora.client import Client
from la_lora.config import Config
from la_lora.data import load_datasets
from la_lora.server import Server


def build(config: Config) -> Server:
	"""
	Build the LA-LoRA server.

	Builds the server with device, random number generator, tokenizer, train
	sharding, evaluation dataset, base model, and PEFT model configuration.
	Instantiates the clients with their local datasets.

	Parameters
	----------
	config : Config
		The hyperparameter configuration from config.py.

	Returns
	-------
	Server
		An instance of the Server class from server.py.
	"""
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# set the seed for generating random numbers on all devices
	generator = torch.manual_seed(config.seed)

	tokenizer = AutoTokenizer.from_pretrained(config.model_name)

	train_shards, eval_dataset = load_datasets(config, tokenizer)

	base = AutoModelForSequenceClassification.from_pretrained(
		config.model_name, num_labels=config.num_labels
	).to(device)

	lora_config = LoraConfig(
		task_type=TaskType.SEQ_CLS,
		target_modules=list(config.target_modules),
		r=config.lora_rank,
		lora_alpha=config.lora_alpha,
		lora_dropout=config.lora_dropout,
		modules_to_save=['classifier']
		if config.train_classifier_head
		else None,  # train classifier head as well
	)

	model = get_peft_model(base, lora_config)

	collator = DataCollatorWithPadding(tokenizer=tokenizer)

	clients = [
		Client(i, shard, config, device, collator)
		for i, shard in enumerate(train_shards)
	]

	return Server(
		config, model, clients, eval_dataset, tokenizer, device, generator
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
	print(f'[la_lora] wrote {path}', flush=True)
	return result
