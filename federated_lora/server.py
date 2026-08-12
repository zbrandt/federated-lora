from __future__ import annotations

import random
import time
from collections import defaultdict

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
		max_grad_norm: float,
		test_dataloader: DataLoader,
		device: str,
		seed: int,
	) -> None:
		self.model = model
		self.method = method
		self.rounds = rounds
		self.clients = clients
		self.max_grad_norm = max_grad_norm
		self.test_dataloader = test_dataloader
		self.device = device
		self.seed = seed
		self.per_client_state: dict[int, dict[str, torch.Tensor]] = { client.id: get_trainable_state(client.model) for client in clients }

	@staticmethod
	def fedavg(
		uploads: dict[int, dict[str, torch.Tensor]],
		num_examples: dict[int, int],
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
		total = sum(num_examples.values())
		weights = {cid: n / total for cid, n in num_examples.items()}
		keys = next(iter(uploads.values())).keys()

		# rewrite such that we are taking the keys and the items from the client dictionray
		return {
			key: sum(
				weights[cid] * upload[key].float()
				for cid, upload in uploads.items()
			)
			for key in keys
		}

	def run(self):
		""" """
		history = []
		rng = random.Random(self.seed)

		for round in range(1, self.rounds + 1):
			# select clients samples ALL clients
			indices = rng.sample(range(len(self.clients)), len(self.clients))
			selected = [self.clients[i] for i in indices]

			t0 = time.perf_counter()
			uploads, losses = {}, []
			for client in selected:
				load_trainable_state(client.model, self.per_client_state[client.id])
				# broadcast model weights
				# global_state = get_trainable_state(self.model)
				# needs to have a rank-aware variant
				# update the local client model based on rank \
				# what does flora broadcast to all of the clients at each round?
				# expand global state to make each client key have as its value a dictionary of parameter name, parameter value

				# load_trainable_state(client.model, global_state)

				# compute local client uploads
				upload, loss = client.local_update(
					round=round, method=self.method
				)

				# storing uploads in a dictionary
				uploads[client.id] = upload
				losses.append(loss)

			# aggregate client uploads into the new global state
			# num_examples = [client.num_examples for client in selected]
			num_examples = {client.id: client.num_examples for client in selected}
			# aggregate all of the uploads, for different client states
			self.per_client_state, eval_state = self.method.aggregate(uploads=uploads, num_examples=num_examples)
			# agg = self.method.aggregate(
			# 	uploads=uploads, num_examples=num_examples, round=round
			# )

			# update the global model with the aggregated weights
			load_trainable_state(self.model, eval_state)

			t1 = time.perf_counter()
			test_loss, top1_acc, top5_acc = evaluate(
				self.model,
				self.test_dataloader,
				self.device,
			)
			t2 = time.perf_counter()

			metrics = {
				'round_index': round,
				'train_loss': sum(losses) / max(1, len(losses)),
				'test_loss': test_loss,
				'top1_acc': top1_acc,
				'top5_acc': top5_acc,
			}

			history.append(metrics)

			print(
				f'[{self.method.name} round {round}/{self.rounds}] '
				f'train_loss={metrics["train_loss"]:.4f} '
				f'test_loss={metrics["test_loss"]:.4f} '
				f'top1_acc={metrics["top1_acc"]:.4f} '
				f'top5_acc={metrics["top5_acc"]:.4f} '
				f'train_s={t1 - t0:.1f} eval_s={t2 - t1:.1f}',
				flush=True,
			)

		return history
