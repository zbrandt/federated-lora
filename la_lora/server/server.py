from __future__ import annotations

from dataclasses import dataclass

from la_lora.config import Config
from la_lora.eval import evaluate_accuracy
from la_lora.server.aggregate import aggregate
from la_lora.server.broadcast import broadcast
from la_lora.server.computation import client_computation
from la_lora.server.selection import select_clients
from la_lora.server.update import update_model


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
		clients,
		eval_dataset,
		tokenizer,
		device,
		generator,
	) -> None:
		self.config = config
		self.model = model
		self.clients = clients
		self.eval_dataset = eval_dataset
		self.tokenizer = tokenizer
		self.device = device
		self.generator = generator
		self.global_state = {
			key: value.detach().cpu().clone()
			for key, value in model.state_dict().items()
			if 'lora_' in key or 'modules_to_save' in key
		}

	def run(self) -> list[RoundMetrics]:
		""" """
		history: list[RoundMetrics] = []

		for round_index in range(1, self.config.rounds + 1):
			# Client selection
			selected_clients = select_clients(
				self.clients, self.config.client_sample_rate, self.generator
			)

			# Broadcast
			broadcast(self.model, self.global_state)

			# Client computation
			states, losses = client_computation(
				selected_clients, self.model, self.global_state
			)

			# Aggregation
			aggregated = aggregate(states)

			# Model update
			self.global_state = update_model(aggregated)

			self.model.load_state_dict(
				self.global_state, strict=False
			)  # TODO: figure out what this does

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
