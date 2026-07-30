from __future__ import annotations

from dataclasses import dataclass

from federated_lora.config import Config
from federated_lora.core.broadcast import broadcast
from federated_lora.core.eval import evaluate_accuracy
from federated_lora.core.selection import select_clients


@dataclass(slots=True)
class RoundMetrics:
	round_index: int
	train_loss: float
	eval_loss: float
	accuracy: float


class Server:
	def __init__(
		self,
		config: Config,
		model,
		method,
		clients,
		eval_dataset,
		tokenizer,
		device,
		generator,
	) -> None:
		self.config = config
		self.model = model
		self.method = method
		self.clients = clients
		self.eval_dataset = eval_dataset
		self.tokenizer = tokenizer
		self.device = device
		self.generator = generator
		self.global_state = {
			key: value.detach().cpu().clone()
			for key, value in model.state_dict().items()
			if 'lora_' in key
		}

	def run(self) -> list[RoundMetrics]:
		""" """
		history: list[RoundMetrics] = []

		for round_index in range(1, self.config.rounds + 1):
			selected_clients = select_clients(
				self.clients, self.config.client_sample_rate, self.generator
			)

			# Broadcast
			broadcast(self.model, self.global_state)

			# Client computation
			client_uploads, losses = [], []
			for client in selected_clients:
				upload, loss = self.method.local_update(
					self.device,
					self.model,
					client.train_dataloader,
					self.config.privacy.noise_multiplier,
					self.config.privacy.clip_norm,
					self.config.local_steps,
				)
				client_uploads.append(upload)
				losses.append(loss)

			# Aggregation
			global_dict = self.method.aggregate(self.model, client_uploads)

			# TODO: model update
			self.model.load_state_dict(global_dict, strict=False)

			eval_loss, accuracy = evaluate_accuracy(
				self.model,
				self.eval_dataset,
				self.tokenizer,
				self.config,
				self.device,
			)

			metrics = RoundMetrics(
				round_index=round_index,
				train_loss=sum(losses) / max(1, len(losses)),
				eval_loss=eval_loss,
				accuracy=accuracy,
			)
			history.append(metrics)

			print(
				f'[round {round_index}/{self.config.rounds}] '
				f'train_loss={metrics.train_loss:.4f} '
				f'eval_loss={metrics.eval_loss:.4f} '
				f'acc={metrics.accuracy:.4f}',
				flush=True,
			)

		return history
