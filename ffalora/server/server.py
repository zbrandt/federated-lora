"""
The Server runs the federated learning loop: each round it picks some
clients, sends them the current model, collects their local updates,
averages them together, and evaluates the result.
"""

from __future__ import annotations

from dataclasses import dataclass

from ffalora.config import Config
from ffalora.eval import evaluate_accuracy
from ffalora.server.selection import select_clients
from ffalora.server.broadcast import broadcast
from ffalora.server.aggregate import aggregate
from ffalora.server.computation import client_computation


# Everything logged about one round of training.
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
	# Store the model, clients, and starting state for a run.
	def __init__(self, config: Config, model, clients, eval_dataset, tokenizer, device, generator) -> None:
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
			if key.endswith(".B.weight") or "classifier" in key
		}

	# Run every round: select clients, broadcast the model, train, aggregate, and evaluate.
	def run(self) -> list[RoundMetrics]:
		history: list[RoundMetrics] = []
		cumulative = 0

		for round_index in range(1, self.config.rounds + 1):
			# Client selection
			selected_clients = select_clients(self.clients, self.config.client_sample_rate, self.generator)

			# Broadcast
			broadcast(self.model, self.global_state)

			# Client computation
			states, losses, weights, uploaded, epsilons = client_computation(selected_clients, self.model, self.global_state)

			# Aggregation: FFA-LoRA's frozen, shared A makes plain weighted averaging exact
			# (see server/aggregate.py) -- no extra merge step needed between rounds.
			self.global_state = aggregate(states, weights)

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
