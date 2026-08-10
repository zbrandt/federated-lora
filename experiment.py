from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from federated_lora.client import Client
from federated_lora.config import Config
from federated_lora.data import load_datasets, prepare_dataloaders
from federated_lora.methods import get_method
from federated_lora.model import create_peft_model
from federated_lora.server import Server


def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument('--task', type=str, required=True)
	parser.add_argument('--method', type=str, required=True)
	parser.add_argument('--batch-size', type=int, default=64)
	parser.add_argument('--epsilon', type=float, default=3.0)
	parser.add_argument('--lr-a', type=float, default=0.25)
	parser.add_argument('--lr-b', type=float, default=0.25)
	parser.add_argument('--seed', type=int, default=42)
	parser.add_argument('--output-dir', type=str, default='outputs/')

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

	# select trainable parameters
	method.prepare_model(model)

	# TODO: train shards, test OR data loaders
	data_dir = load_datasets(
		name=config.dataset,
		num_clients=config.num_clients,
		alpha=config.dirichlet_alpha,
		seed=config.seed,
	)

	client_dataloaders, test_dataloader = prepare_dataloaders(
		model=config.model,
		data_dir=data_dir,
		num_clients=config.num_clients,
		batch_size=config.batch_size,
	)

	clients = [
		Client(
			id=i,
			model=model,
			dataloader=dataloader,
			config=config,
			device=device,
		)
		for i, dataloader in enumerate(client_dataloaders)
	]

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
		output_dir=args.output_dir
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


if __name__ == "__main__":
	args = parse_args()
	run(args)
