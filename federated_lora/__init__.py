from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch
from opacus.accountants.utils import get_noise_multiplier
from torch.optim import SGD
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor

from federated_lora.config import Config
from federated_lora.core.client import Client
from federated_lora.core.data import load_datasets
from federated_lora.core.model import create_peft_model
from federated_lora.core.server import Server
from federated_lora.privacy.accountant import achieved_epsilon
from federated_lora.registry import get_method


def build(config: Config) -> Server:
	"""
	Build the federated LoRA server and clients for image classification.

	Parameters
	----------
	config : Config
		Run configuration.

	Returns
	-------
	Server
		The configured federated server.
	"""
	# set the device type
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# set the seed for generating random numbers on all devices
	generator = torch.manual_seed(config.seed)

	image_processor = AutoImageProcessor.from_pretrained(config.model)

	train_shards, test_dataset = load_datasets(config, image_processor)

	shard_sizes = [len(s) for s in train_shards]
	mean_shard = sum(shard_sizes) / len(shard_sizes)
	sample_rate = config.batch_size / mean_shard
	if min(shard_sizes) < config.batch_size:
		raise ValueError(
			f'smallest shard ({min(shard_sizes)}) < batch_size '
			f'({config.batch_size}): per-example sampling rate would exceed 1. '
			'Lower batch_size, or reduce client count / skew.'
		)
	steps = config.global_rounds * config.local_steps
	if config.privacy.target_epsilon is None:
		config.privacy.noise_multiplier = 0.0
	else:
		config.privacy.noise_multiplier = get_noise_multiplier(
			target_epsilon=config.privacy.target_epsilon,
			target_delta=config.privacy.target_delta,
			sample_rate=sample_rate,
			steps=steps,
		)
		# Round-trip check: the sigma we just picked must account back to
		# (approximately) the target epsilon at this rate and step count.
		spent = achieved_epsilon(
			config.privacy.noise_multiplier,
			sample_rate,
			steps,
			config.privacy.target_delta,
		)
		assert spent <= config.privacy.target_epsilon + 0.1, (
			f'calibration mismatch: target {config.privacy.target_epsilon}, '
			f'accounted {spent}'
		)

	method = get_method(config.method)

	model = create_peft_model(config, device)

	params_A, params_B, params_head = [], [], []
	for name, param in model.named_parameters():
		if not param.requires_grad:
			continue
		if 'lora_A' in name:
			params_A.append(param)
		elif 'lora_B' in name:
			params_B.append(param)
		else:
			params_head.append(param)

	clients = []
	for i, shard in enumerate(train_shards):
		generator = torch.Generator().manual_seed(config.seed + i)

		optimizer = SGD(
			[
				{'params': params_A, 'lr': config.lr_a},
				{'params': params_B, 'lr': config.lr_b},
				{'params': params_head, 'lr': config.lr_head},
			]
		)

		train_dataloader = DataLoader(
			shard,
			batch_size=config.batch_size,
			shuffle=True,
			# collate_fn=,
			generator=generator,
		)

		clients.append(
			Client(
				id=i,
				shard=shard,
				steps=config.local_steps,
				dataloader=train_dataloader,
				optimizer=optimizer,
			)
		)

	test_dataloader = DataLoader(
		test_dataset,
		batch_size=config.batch_size,
		shuffle=False,
		# collate_fn=,
	)

	return Server(
		model=model,
		method=method,
		rounds=config.global_rounds,
		sample_rate=config.client_sample_rate,
		clients=clients,
		generator=generator,
		noise_multiplier=config.privacy.noise_multiplier,
		max_grad_norm=config.privacy.clip_norm,
		test_dataloader=test_dataloader,
		device=device,
		lr_decay=config.lr_decay,
	)


def run(argv: list[str] | None = None) -> dict:
	"""
	Run the LA-LoRA method.

	Get the configuration of the hyperparameters from argument parsers,
	instantiate and run the server from the configuration. Saves results to
	specified or default result directory indexed by method name (label only),
	task, and seed.

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
	history = server.run()
	result = {'config': asdict(config), 'history': history}

	if config.privacy.noise_multiplier == 0.0:
		result['achieved_epsilon'] = None
		result['epsilon_breakdown'] = None
	else:
		steps = config.global_rounds * config.local_steps
		per_client = [
			achieved_epsilon(
				config.privacy.noise_multiplier,
				config.batch_size / len(client.shard),
				steps,
				config.privacy.target_delta,
			)
			for client in server.clients
		]
		result['achieved_epsilon'] = max(per_client)  # worst-case client
		result['epsilon_breakdown'] = {
			'min': min(per_client),
			'mean': sum(per_client) / len(per_client),
			'max': max(per_client),
			'per_client': per_client,
		}

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
