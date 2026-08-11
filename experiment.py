from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from opacus import PrivacyEngine
from torch.optim import SGD

from federated_lora.client import Client
from federated_lora.config import Config
from federated_lora.data import load_datasets, prepare_dataloaders
from federated_lora.methods import get_method
from federated_lora.model import create_peft_model, group_trainable_parameters
from federated_lora.privacy import compute_noise_level
from federated_lora.server import Server


def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument('--task', type=str, required=True)
	parser.add_argument('--method', type=str, required=True)
	parser.add_argument('--batch-size', type=int, default=64)
	parser.add_argument('--epsilon', type=float, default=3.0)
	parser.add_argument('--lr-a', type=float, default=0.25)
	parser.add_argument('--lr-b', type=float, default=0.25)
	parser.add_argument('--lr-head', type=float, default=0.15)
	parser.add_argument('--seed', type=int, default=42)
	parser.add_argument('--output-dir', type=str, default='results/')

	return parser.parse_args()


def build(config: Config) -> Server:
	""" """
	random.seed(config.seed)
	np.random.seed(config.seed)

	torch.manual_seed(config.seed)
	torch.cuda.manual_seed(config.seed)
	torch.cuda.manual_seed_all(config.seed)

	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	model = create_peft_model(
		name=config.model,
		num_labels=config.num_labels,
		lora_rank=config.lora_rank,
		target_modules=config.target_modules,
		lora_alpha=config.lora_alpha,
		lora_dropout=config.lora_dropout,
		device=device,
	)

	method = get_method(config.method)

	data_dir = load_datasets(
		name=config.dataset,
		num_clients=config.num_clients,
		alpha=config.dirichlet_alpha,
		seed=config.seed,
	)

	train_dataloaders, test_dataloader = prepare_dataloaders(
		model=config.model,
		data_dir=data_dir,
		num_clients=config.num_clients,
		batch_size=config.batch_size,
	)

	clients = []
	for i, dataloader in enumerate(train_dataloaders):
		local_model = copy.deepcopy(model).to('cpu')

		params_A, params_B, params_head = group_trainable_parameters(
			local_model
		)
		optimizer = SGD(
			[
				{'params': params_A, 'lr': config.lr_a},
				{'params': params_B, 'lr': config.lr_b},
				{'params': params_head, 'lr': config.lr_head},
			],
			weight_decay=0.0,
		)

		sigma = compute_noise_level(
			num_examples=len(dataloader.dataset),
			batch_size=config.batch_size,
			global_rounds=config.global_rounds,
			local_steps=config.local_steps,
			target_epsilon=config.target_epsilon,
			target_delta=config.target_delta,
		)

		privacy_engine = PrivacyEngine()
		local_model, optimizer, dataloader = privacy_engine.make_private(
			module=local_model,
			optimizer=optimizer,
			data_loader=dataloader,
			noise_multiplier=sigma,
			max_grad_norm=config.clip_norm,
		)

		clients.append(
			Client(
				id=i,
				model=model,
				dataloader=dataloader,
				optimizer=optimizer,
				privacy_engine=privacy_engine,
				steps=config.local_steps,
				device=device,
			)
		)

	return Server(
		model=model,
		method=method,
		rounds=config.global_rounds,
		clients=clients,
		max_grad_norm=config.clip_norm,
		test_dataloader=test_dataloader,
		device=device,
		seed=config.seed,
	)


def run(args: list[str]) -> dict:
	""" """
	config = Config(
		task=args.task,
		method=args.method,
		batch_size=args.batch_size,
		target_epsilon=args.epsilon,
		lr_a=args.lr_a,
		lr_b=args.lr_b,
		seed=args.seed,
		output_dir=args.output_dir,
	)
	server = build(config)
	history = server.run()
	result = {'config': asdict(config), 'history': history}

	epsilons = []
	for client in server.clients:
		privacy_engine = client.privacy_engine
		epsilons.append(privacy_engine.get_epsilon(config.target_delta))

	# TODO: add more metrics
	result['epsilon_breakdown'] = {'per_client': epsilons}

	path = (
		Path(config.output_dir)
		/ f'{config.method}_{config.task}_seed{config.seed}.json'
	)

	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(result, indent=2), encoding='utf-8')
	return result


if __name__ == '__main__':
	args = parse_args()
	run(args)
