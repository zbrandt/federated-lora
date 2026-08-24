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
from peft import LoraConfig, get_peft_model
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR
from transformers import AutoModelForImageClassification

from federated_lora.client import Client
from federated_lora.data.vision import prepare_dataloaders, prepare_cifar100, prepare_tinyimagenet
from federated_lora.methods import METHODS
from federated_lora.privacy import compute_noise_level
from federated_lora.server import Server

NOISE_MULTIPLIERS = {
	'cifar100': {1.0: 0.56, 2.0: 0.29, 3.0: 0.195},
	'tinyimagenet': {1.0: 0.283, 2.0: 0.146, 3.0:0.098}
}

MODELS = {
	'vit': 'google/vit-base-patch16-224-in21k',
	'swin': 'microsoft/swin-tiny-patch4-window7-224'
}

DATASETS = {
	'cifar100': prepare_cifar100.load_datasets,
	'tinyimagenet': prepare_tinyimagenet.load_datasets
}

LABELS = {
	'cifar100': 100,
	'tinyimagenet': 200,
}


def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument('--task', type=str, required=True)
	parser.add_argument('--method', type=str, required=True)

	parser.add_argument('--batch_size', type=int, default=16)
	parser.add_argument('--epsilon', type=float, default=3.0)
	parser.add_argument('--delta', type=float, default=1e-5)
	parser.add_argument('--max_grad_norm', type=float, default=2.0)

	parser.add_argument('--model', type=str, required=True)

	parser.add_argument('--num_clients', type=int, default=8)
	parser.add_argument('--sample_rate', type=float, default=0.5)

	parser.add_argument('--dirichlet_alpha', type=float, default=0.1)

	parser.add_argument('--global_rounds', type=int, default=100)
	parser.add_argument('--local_steps', type=int, default=20)

	parser.add_argument('--lora_rank', type=int, default=16)
	parser.add_argument('--lora_alpha', type=int, default=16)
	parser.add_argument('--lora_dropout', type=float, default=0.05)

	parser.add_argument('--lr_head', type=float, default=0.20)
	parser.add_argument('--lr_a', type=float, default=0.20)
	parser.add_argument('--lr_b', type=float, default=0.20)
	parser.add_argument('--lr_decay', type=float, default=0.99)

	parser.add_argument('--seed', type=int, default=42)
	parser.add_argument('--output_dir', type=str, default='./logs')

	return parser.parse_args()


def run(args: argparse.Namespace) -> None:
	random.seed(args.seed)
	np.random.seed(args.seed)

	torch.manual_seed(args.seed)
	torch.cuda.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)

	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	method = METHODS[args.method]()

	if args.model == 'swin' or args.model == 'vit':
		base = AutoModelForImageClassification.from_pretrained(
			MODELS[args.model], num_labels=LABELS[args.task]
		).to(device)

		peft_config = LoraConfig(
			r=args.lora_rank,
			lora_alpha=args.lora_alpha,
			lora_dropout=args.lora_dropout,
			target_modules=('query', 'value'),
			bias='none',
		)

	model = get_peft_model(base, peft_config)

	method.set_target_modules(model)

	data_dir = DATASETS[args.task](
		num_clients=args.num_clients,
		dirichlet_alpha=args.dirichlet_alpha,
		seed=args.seed,
	)

	if args.model == 'swin' or args.model == 'vit':
		train_dataloaders, test_dataloader = prepare_dataloaders(
			model=MODELS[args.model],
			data_dir=data_dir,
			num_clients=args.num_clients,
			batch_size=args.batch_size,
			task=args.task,
		)

	clients = []
	for i, dataloader in enumerate(train_dataloaders):
		local_model = copy.deepcopy(model).to('cpu')
		num_examples = len(dataloader.dataset)

		params_A, params_B, params_head = [], [], []
		for name, param in local_model.named_parameters():
			if not param.requires_grad:
				continue
			if 'lora_A' in name:
				params_A.append(param)
			elif 'lora_B' in name:
				params_B.append(param)
			else:
				params_head.append(param)

		optimizer = SGD(
			[
				{'params': params_A, 'lr': args.lr_a},
				{'params': params_B, 'lr': args.lr_b},
				{'params': params_head, 'lr': args.lr_head},
			],
		)

		if args.model == 'vit':
			noise_multiplier = compute_noise_level(
				num_examples=num_examples,
				batch_size=args.batch_size,
				global_rounds=args.global_rounds,
				local_steps=args.local_steps,
				target_epsilon=args.epsilon,
				target_delta=args.delta,
			)
		elif args.model == 'swin':
			noise_multiplier = NOISE_MULTIPLIERS[args.task][args.epsilon]

		privacy_engine = PrivacyEngine(accountant="rdp")
		local_model, optimizer, dataloader = privacy_engine.make_private(
			module=local_model,
			optimizer=optimizer,
			data_loader=dataloader,
			noise_multiplier=noise_multiplier,
			max_grad_norm=args.max_grad_norm,
		)

		if args.model == 'vit':
			optimizer.clipping_strategy = 'flat'
		elif args.model == 'swin':
			optimizer.clipping_strategy = 'median'

		scheduler = ExponentialLR(optimizer, gamma=args.lr_decay)

		clients.append(
			Client(
				id=i,
				model=local_model,
				dataloader=dataloader,
				optimizer=optimizer,
				scheduler=scheduler,
				privacy_engine=privacy_engine,
				num_examples=num_examples,
				steps=args.local_steps,
				device=device,
			)
		)

	server = Server(
		model=model,
		method=method,
		rounds=args.global_rounds,
		clients=clients,
		sample_rate=args.sample_rate,
		test_dataloader=test_dataloader,
		delta=args.delta,
		device=device,
		seed=args.seed,
	)

	history = server.run()
	result = {'config': vars(args), 'history': history}

	path = (
		Path(args.output_dir)
		/ f'{args.method}_{args.model}_{args.task}_eps{args.epsilon:g}_seed{args.seed}.json'
	)

	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(result, indent=2), encoding='utf-8')


def main():
	args = parse_args()
	run(args)

if __name__ == "__main__":
	main()
