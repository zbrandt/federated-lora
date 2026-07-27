from __future__ import annotations

from dataclasses import dataclass

from flora.config import Config
from flora.eval import evaluate_accuracy
from flora.server.selection import select_clients
from flora.server.broadcast import broadcast
from flora.server.aggregate import aggregate
from flora.server.computation import client_computation
from flora.server.update import update_model


@dataclass(slots=True)
class RoundMetrics:
	round_index: int
	train_loss: float
	eval_loss: float
	accuracy: float
	uploaded_bytes: int
	cumulative_uploaded_bytes: int
	epsilon_spent: float | None


class Server:
	def __init__(self, config: Config, model, clients, eval_dataset, tokenizer, device, generator) -> None:
		self.config = config
		self.model = model
		self.clients = clients
		self.eval_dataset = eval_dataset
		self.tokenizer = tokenizer
		self.device = device
		self.generator = generator
		self.scaling = config.lora_alpha / config.lora_rank
		self.global_state = {
			key: value.detach().cpu().clone()
			for key, value in model.state_dict().items()
			if "lora_" in key or "modules_to_save" in key
		}

	def run(self) -> list[RoundMetrics]:
		history: list[RoundMetrics] = []
		cumulative = 0

		for round_index in range(1, self.config.rounds + 1):
			# Client selection
			selected_clients = select_clients(self.clients, self.config.client_sample_rate, self.generator)

			# Broadcast
			broadcast(self.model, self.global_state)

			# Client computation
			states, losses, uploaded, epsilons = client_computation(selected_clients, self.model, self.global_state)

			# Aggregation (FLoRA: exact stacking, not FedAvg)
			aggregated = aggregate(states, self.scaling)

			# Model update: fold the stacked delta into the base weights, reinit fresh adapters
			self.global_state = update_model(self.model, aggregated, self.generator)

			self.model.load_state_dict(self.global_state, strict=False)

			eval_loss, accuracy = evaluate_accuracy(
				self.model,
				self.eval_dataset,
				self.tokenizer,
				self.config,
				self.device,
			)

			cumulative += uploaded
			round_epsilon = max((e for e in epsilons if e is not None), default=None)
			metrics = RoundMetrics(
				round_index=round_index,
				train_loss=sum(losses) / max(1, len(losses)),
				eval_loss=eval_loss,
				accuracy=accuracy,
				uploaded_bytes=uploaded,
				cumulative_uploaded_bytes=cumulative,
				epsilon_spent=round_epsilon,
			)
			history.append(metrics)

			eps_str = f" eps={round_epsilon:.4f}" if round_epsilon is not None else ""
			print(
				f"[round {round_index}/{self.config.rounds}] "
				f"train_loss={metrics.train_loss:.4f} "
				f"eval_loss={metrics.eval_loss:.4f} "
				f"acc={metrics.accuracy:.4f}"
				f"{eps_str}",
				flush=True,
			)

		return history
