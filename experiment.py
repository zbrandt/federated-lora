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
from torch.optim.lr_scheduler import ExponentialLR

from federated_lora.client import Client
from federated_lora.config import TASKS, Config
from federated_lora.data import load_datasets, prepare_dataloaders
from federated_lora.methods import METHODS
from federated_lora.model import create_peft_model, group_trainable_parameters
from federated_lora.privacy import compute_noise_level
from federated_lora.server import Server


def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument('--task', type=str, required=True)
	parser.add_argument('--method', type=str, required=True)
	parser.add_argument('--batch-size', type=int, default=16)
	# parser.add_argument('--rounds', type=int, default=100)
	parser.add_argument('--epsilon', type=float, default=3.0)
	parser.add_argument('--clipping', type=str, default='median')
	parser.add_argument('--lr-a', type=float, default=0.20)
	parser.add_argument('--lr-b', type=float, default=0.20)
	parser.add_argument('--lr-head', type=float, default=0.20)
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

	method = METHODS[config.method]()

	model = create_peft_model(
		name=config.model,
		num_labels=config.num_labels,
		lora_rank=config.lora_rank,
		target_modules=config.target_modules,
		lora_alpha=config.lora_alpha,
		lora_dropout=config.lora_dropout,
		method=method,
		device=device,
		experiment=config.experiment,
	)

	data_dir = load_datasets(
		dataset=config.dataset,
		num_clients=config.num_clients,
		alpha=config.dirichlet_alpha,
		task=config.task,
		experiment=config.experiment,
		seed=config.seed,
	)

	train_dataloaders, test_dataloader = prepare_dataloaders(
		model=config.model,
		data_dir=data_dir,
		num_clients=config.num_clients,
		batch_size=config.batch_size,
		task=config.task,
		experiment=config.experiment,
	)

	clients = []
	for i, dataloader in enumerate(train_dataloaders):
		local_model = copy.deepcopy(model).to('cpu')
		num_examples = len(dataloader.dataset)

		params_A, params_B, params_head = group_trainable_parameters(
			local_model
		)
		optimizer = SGD(
			[
				{'params': params_A, 'lr': config.lr_a},
				{'params': params_B, 'lr': config.lr_b},
				{'params': params_head, 'lr': config.lr_head},
			],
		)

		noise_multiplier = compute_noise_level(
			num_examples=num_examples,
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
			noise_multiplier=noise_multiplier,
			max_grad_norm=config.max_grad_norm,
		)

		optimizer.clipping_strategy = config.clipping

		scheduler = ExponentialLR(optimizer, gamma=config.lr_decay)

		clients.append(
			Client(
				id=i,
				model=local_model,
				dataloader=dataloader,
				optimizer=optimizer,
				scheduler=scheduler,
				privacy_engine=privacy_engine,
				num_examples=num_examples,
				steps=config.local_steps,
				device=device,
			)
		)

	return Server(
		model=model,
		method=method,
		rounds=config.global_rounds,
		clients=clients,
		sample_rate=config.client_sample_rate,
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
		# global_rounds=args.rounds,
		target_epsilon=args.epsilon,
		lr_a=args.lr_a,
		lr_b=args.lr_b,
		lr_head=args.lr_head,
		seed=args.seed,
		output_dir=args.output_dir,
		clipping=args.clipping,
	)

	for key, value in TASKS[args.task].items():
		setattr(config, key, value)

	if not args.epsilon:
		config.clipping = 'none'

	server = build(config)
	history = server.run()
	result = {'config': asdict(config), 'history': history}

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
