from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch
from datasets import load_dataset
from opacus.accountants.utils import get_noise_multiplier
from torch.optim import SGD
from torch.utils.data import DataLoader, Subset
from transformers import AutoImageProcessor

from federated_lora.config import Config
from federated_lora.core.client import Client
from federated_lora.core.data import build_cache, partition_dirichlet
from federated_lora.core.model import create_peft_model
from federated_lora.core.privacy import compute_noise_level, perform_accounting
from federated_lora.core.server import Server
from federated_lora.registry import get_method


# TODO
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
	torch.backends.cudnn.benchmark = True
	torch.backends.cuda.matmul.allow_tf32 = True
	torch.backends.cudnn.allow_tf32 = True

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	method = get_method(config.method)

	model = create_peft_model(
		model=config.model,
		num_labels=config.num_labels,
		lora_rank=config.lora_rank,
		target_modules=config.target_modules,
		lora_alpha=config.lora_alpha,
		lora_dropout=config.lora_dropout,
		device=device,
	)

	# TODO: move to models
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

	dataset = load_dataset(path=config.dataset)
	processor = AutoImageProcessor.from_pretrained(
		pretrained_model_name_or_path=config.model,
		use_fast=True,
	)

	mean = torch.tensor(processor.image_mean).view(1, 3, 1, 1)
	std = torch.tensor(processor.image_std).view(1, 3, 1, 1)

	train_dataset = build_cache(dataset['train'], processor)
	test_dataset = build_cache(dataset['test'], processor)

	def collate_fn(batch):
		images = torch.stack(
			[item[0] for item in batch]
		)  # (B,3,224,224) uint8
		labels = torch.stack([item[1] for item in batch])  # (B,) long
		pixel_values = images.float().div_(255.0).sub_(mean).div_(std)
		return {'pixel_values': pixel_values, 'labels': labels}

	train_shards = partition_dirichlet(
		dataset['train']['fine_label'],
		config.num_clients,
		config.dirichlet_alpha,
		config.seed,
	)

	test_dataloader = DataLoader(
		test_dataset,
		batch_size=256,
		shuffle=False,
		collate_fn=collate_fn,
		num_workers=config.num_workers,
		pin_memory=(device.type == 'cuda'),
		persistent_workers=config.num_workers > 0,
	)

	clients = []
	steps = config.global_rounds * config.local_steps
	for client_id, indices in enumerate(train_shards):
		client_dataset = Subset(train_dataset, list(indices))

		client_seed = config.seed + client_id

		client_generator = torch.Generator()
		client_generator.manual_seed(client_seed)

		client_dataloader = DataLoader(
			client_dataset,
			batch_size=config.batch_size,
			shuffle=True,
			collate_fn=collate_fn,
			generator=client_generator,
			# worker_init_fn=
			num_workers=config.num_workers,
			pin_memory=(device.type == 'cuda'),
		)

		sample_rate = config.batch_size / len(client_dataset)
		client_noise_multiplier = compute_noise_level(
			target_epsilon=config.privacy.target_epsilon,
			target_delta=config.privacy.target_delta,
			sample_rate=sample_rate,
			steps=steps,
		)

		client_optimizer = SGD(
			[
				{'params': params_A, 'lr': config.lr_a},
				{'params': params_B, 'lr': config.lr_b},
				{'params': params_head, 'lr': config.lr_head},
			]
		)

		clients.append(
			Client(
				id=client_id,
				dataset=client_dataset,
				steps=config.local_steps,
				dataloader=client_dataloader,
				optimizer=client_optimizer,
				noise_multiplier=client_noise_multiplier,
			)
		)

	return Server(
		model=model,
		method=method,
		rounds=config.global_rounds,
		sample_rate=config.client_sample_rate,
		clients=clients,
		generator=torch.Generator().manual_seed(config.seed),
		max_grad_norm=config.privacy.clip_norm,
		test_dataloader=test_dataloader,
		device=device,
		lr_decay=config.lr_decay,
	)


# TODO
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
	client_noise_multipliers = [
		client.noise_multiplier for client in server.clients
	]
	steps = config.global_rounds * config.local_steps

	if config.privacy.target_epsilon is None:
		result['target_epsilon'] = None
		result['epsilon_breakdown'] = None
	else:
		result['target_epsilon'] = config.privacy.target_epsilon
		per_client = [
			perform_accounting(
				client.noise_multiplier,
				config.batch_size / len(client.dataset),
				steps,
				config.privacy.target_delta,
			)
			for client in server.clients
		]
		result['epsilon_breakdown'] = {
			'noise_multiplier_min': min(client_noise_multipliers),
			'noise_multiplier_mean': sum(client_noise_multipliers)
			/ len(client_noise_multipliers),
			'noise_multiplier_max': max(client_noise_multipliers),
			'noise_multiplier_per_client': client_noise_multipliers,
			'per_shard_min': min(per_client),
			'per_shard_mean': sum(per_client) / len(per_client),
			'per_shard_max': max(per_client),
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
