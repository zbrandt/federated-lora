from __future__ import annotations

import random
import time

from torch.utils.data import DataLoader
from transformers import PreTrainedModel

from federated_lora.client import Client
from federated_lora.eval import evaluate
from federated_lora.method import Method


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

	def run(self):
		""" """
		history = []
		rng = random.Random(self.seed)

		for round in range(1, self.rounds + 1):
			# select clients samples ALL clients
			indices = rng.sample(range(len(self.clients)), len(self.clients))
			selected = [self.clients[i] for i in indices]

			t0 = time.perf_counter()
			uploads, losses = [], []
			for client in selected:
				# TODO: move to broadcast method?
				trainable_parameters = {
					name
					for name, param in self.model.named_parameters()
					if param.requires_grad
				}
				global_state = {
					key: value.detach().cpu().clone()
					for key, value in self.model.state_dict().items()
					if key in trainable_parameters
				}
				client.model._module.load_state_dict(global_state, strict=False)

				# compute local client uploads
				upload, loss = client.update(round=round, method=self.method)

				uploads.append(upload)
				losses.append(loss)

			# aggregate client uploads into the new global state
			agg = self.method.aggregate(uploads=uploads, round=round)

			# update the global model with the aggregated weights
			self.model.load_state_dict(agg, strict=False)

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
