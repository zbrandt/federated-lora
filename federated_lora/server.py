from __future__ import annotations

import random
import time

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedModel

from federated_lora.client import Client
from federated_lora.eval import evaluate
from federated_lora.method import Method
from federated_lora.model import get_trainable_state, load_trainable_state


class Server:
	def __init__(
		self,
		model: PreTrainedModel,
		method: Method,
		rounds: int,
		clients: list[Client],
		sample_rate: float,
		test_dataloader: DataLoader,
		delta: float,
		device: str,
		seed: int,
	) -> None:
		self.model = model
		self.method = method
		self.rounds = rounds
		self.clients = clients
		self.sample_rate = sample_rate
		self.test_dataloader = test_dataloader
		self.delta = delta
		self.device = device
		self.seed = seed

	@staticmethod
	def fedavg(
		uploads: list[dict[str, torch.Tensor]],
		num_examples: list[int],
	) -> dict[str, torch.Tensor]:
		"""
		Calculate the weighted average of client updates to the model.

		Parameters
		----------
		uploads : list[dict[str, torch.Tensor]]
			The client trainable parameter state-dict updates to the model.
		num_examples : list[int]
			The total number of examples of each client.

		Returns
		-------
		dict[str, torch.Tensor]
			The aggregated client updates to the model.
		"""
		total = sum(num_examples)
		weights = [n / total for n in num_examples]

		return {
			key: sum(
				w * upload[key].float()
				for w, upload in zip(weights, uploads, strict=False)
			)
			for key in uploads[0]
		}

	def run(self):
		""" """
		history = []
		rng = random.Random(self.seed)

		for r in range(1, self.rounds + 1):
			# select clients
			k = max(1, round(self.sample_rate * len(self.clients)))
			indices = rng.sample(range(len(self.clients)), k)
			selected = [self.clients[i] for i in indices]

			t0 = time.perf_counter()
			uploads, losses = [], []
			for client in selected:
				# broadcast model weights
				global_state = get_trainable_state(self.model)
				load_trainable_state(client.model, global_state)

				# compute local client uploads
				upload, loss = client.local_update(round=r, method=self.method)

				uploads.append(upload)
				losses.append(loss)

			# aggregate client uploads into the new global state
			num_examples = [client.num_examples for client in selected]
			agg = self.method.aggregate(
				uploads=uploads, num_examples=num_examples
			)

			# TODO: snapshot learning rates before decay
			param_groups = selected[0].optimizer.param_groups
			lr_A = param_groups[0]['lr']
			lr_B = param_groups[1]['lr']
			lr_head = param_groups[2]['lr']

			# decay learning rates
			for client in self.clients:
				client.scheduler.step()

			# update the global model with the aggregated weights
			load_trainable_state(self.model, agg)

			t1 = time.perf_counter()
			test_loss, top1_acc, top5_acc = evaluate(
				self.model,
				self.test_dataloader,
				self.device,
			)
			t2 = time.perf_counter()

			metrics = {
				'round_index': r,
				'train_loss': sum(losses) / max(1, len(losses)),
				'test_loss': test_loss,
				'top1_acc': top1_acc,
				'top5_acc': top5_acc,
				'train_s': t1 - t0,
				'eval_s': t2 - t1,
				'selected_clients': indices,
				'num_selected': len(selected),
				'lr_A': lr_A,
				'lr_B': lr_B,
				'lr_head': lr_head,
			}

			epsilons = [
				client.privacy_engine.get_epsilon(delta=self.delta)
				for client in self.clients
				if client.optimizer.noise_multiplier > 0
				and client.privacy_engine.accountant.history
			]
			metrics['epsilon_spent'] = max(epsilons) if epsilons else None

			history.append(metrics)

			print(
				f'[{self.method.name} round {r}/{self.rounds}] '
				f'train_loss={metrics["train_loss"]:.4f} '
				f'test_loss={metrics["test_loss"]:.4f} '
				f'top1_acc={metrics["top1_acc"]:.4f} '
				f'top5_acc={metrics["top5_acc"]:.4f} '
				f'train_s={t1 - t0:.1f} eval_s={t2 - t1:.1f}',
				flush=True,
			)

		return history
