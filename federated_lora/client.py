from __future__ import annotations

import torch
from opacus import PrivacyEngine
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from transformers import PreTrainedModel

from federated_lora.data import cycle
from federated_lora.methods import Method
from federated_lora.metrics import summarize_diagnostics
from federated_lora.model import get_trainable_state


class Client:
	def __init__(
		self,
		id: int,
		model: PreTrainedModel,
		dataloader: DataLoader,
		optimizer: Optimizer,
		scheduler: ExponentialLR,
		privacy_engine: PrivacyEngine,
		num_examples: float,
		steps: int,
		device: torch.device,
	) -> None:
		self.id = id
		self.model = model
		self.dataloader = dataloader
		self.optimizer = optimizer
		self.scheduler = scheduler
		self.privacy_engine = privacy_engine
		self.num_examples = num_examples
		self.steps = steps
		self.device = device

	def local_update(
		self,
		round: int,
		method: Method,
	) -> tuple[dict, float, dict]:
		"""
		Perform local training steps for one global communication round.

		Parameters
		----------
		round : int
			The current global communication round.
		method : Method
			The federated fine-tuning method.

		Returns
		-------
		tuple[dict, float, dict]
			A trainable state-dict upload for server side aggregation,
			mean local training loss, and round diagnoistics summary.
		"""
		self.model = self.model.to(self.device)
		self.model.train()

		batches = cycle(self.dataloader)

		total_loss = 0.0
		step_diags = []
		for step in range(self.steps):
			batch = next(batches)
			batch = {
				key: value.to(self.device) for key, value in batch.items()
			}
			batch_size = next(iter(batch.values())).shape[0]

			self.optimizer.zero_grad(set_to_none=True)
			out = self.model(**batch)
			loss = out.loss
			loss.backward()

			method.step(
				model=self.model,
				optimizer=self.optimizer,
				round=round,
				step=step,
				batch_size=batch_size,
			)

			# privatize() stashes this step's clipping/grad-norm diagnostics on
			# the optimizer; collect it (empty Poisson lots leave it None).
			diag = getattr(self.optimizer, 'last_diagnostics', None)
			if diag is not None:
				step_diags.append(diag)
				self.optimizer.last_diagnostics = None

			self.optimizer.zero_grad(set_to_none=True)

			total_loss += float(loss.item())

		upload = get_trainable_state(self.model)

		self.model = self.model.to('cpu')

		return (
			upload,
			total_loss / self.steps,
			summarize_diagnostics(step_diags),
		)
