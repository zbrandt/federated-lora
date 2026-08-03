from __future__ import annotations

import torch
from torch import Generator
from torch.utils.data import DataLoader
from transformers import PreTrainedModel

from federated_lora.core.client import Client
from federated_lora.core.eval import evaluate


class Server:
	def __init__(
		self,
		model: PreTrainedModel,
		method,
		rounds: int,
		sample_rate: float,
		clients: list[Client],
		generator: Generator,
		max_grad_norm: float,
		test_dataloader: DataLoader,
		device: str,
		lr_decay: float = 1.0,
	) -> None:
		self.model = model
		self.method = method
		self.rounds = rounds
		self.sample_rate = sample_rate
		self.clients = clients
		self.generator = generator
		self.max_grad_norm = max_grad_norm
		self.test_dataloader = test_dataloader
		self.device = device
		self.lr_decay = lr_decay

		# trainable parameters are the LoRA A and B matrix parameters and the
		# classification head which PEFT keeps in ``modules_to_save``.
		self.trainable_parameters = {
			name
			for name, param in model.named_parameters()
			if param.requires_grad
		}
		self.global_state = {
			key: value.detach().cpu().clone()
			for key, value in model.state_dict().items()
			if key in self.trainable_parameters
		}

	def run(self):
		""" """
		history = []

		for round_index in range(1, self.rounds + 1):
			# select clients
			k = max(1, round(self.sample_rate * len(self.clients)))
			picks = torch.randperm(
				len(self.clients), generator=self.generator
			)[:k]
			selected_clients = [self.clients[i] for i in picks.tolist()]

			client_uploads, losses = [], []
			for client in selected_clients:
				# broadcast global model to client
				self.model.load_state_dict(self.global_state, strict=False)

				# compute local client uploads
				_, loss = self.method.local_update(
					model=self.model,
					optimizer=client.optimizer,
					train_dataloader=client.dataloader,
					noise_multiplier=client.noise_multiplier,
					max_grad_norm=self.max_grad_norm,
					local_steps=client.steps,
					device=self.device,
				)

				# snapshot this client's trained parameters before next client
				# overwrites the shared model in place
				upload = {
					key: value.detach().cpu().clone()
					for key, value in self.model.state_dict().items()
					if key in self.trainable_parameters
				}
				client_uploads.append(upload)
				losses.append(loss)

			# aggregate client uploads into the new global state
			self.global_state = self.method.aggregate(client_uploads)

			# update the global model with the aggregated weights
			self.model.load_state_dict(self.global_state, strict=False)

			eval_loss, accuracy = evaluate(
				self.model,
				self.test_dataloader,
				self.device,
			)

			metrics = {
				'round_index': round_index,
				'train_loss': sum(losses) / max(1, len(losses)),
				'eval_loss': eval_loss,
				'accuracy': accuracy,
			}

			history.append(metrics)

			print(
				f'[round {round_index}/{self.rounds}] '
				f'train_loss={metrics["train_loss"]:.4f} '
				f'eval_loss={metrics["eval_loss"]:.4f} '
				f'acc={metrics["accuracy"]:.4f}',
				flush=True,
			)

			# decay learning rate for all clients
			if self.lr_decay != 1.0:
				for client in self.clients:
					for group in client.optimizer.param_groups:
						group['lr'] *= self.lr_decay

		return history
