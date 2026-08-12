"""
Each Client trains its own slice of data for one round.
It only ever trains the small LoRA B matrix and the classifier head, optionally
adding DP noise (via Opacus) so no single training example can be traced back.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import cycle

import torch
from datasets import Dataset
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, PreTrainedModel

from ffalora.config import Config
from ffalora.lora_layer import HeterogeneousDPLoRALayer


# Everything a client hands back to the server after one round of training.
@dataclass(slots=True)
class ClientResult:
	state_dict: dict[str, torch.Tensor]
	n_examples: int
	n_tokens: int
	average_loss: float
	uploaded_bytes: int
	epsilon_spent: float | None = None


class Client:
	# Store this client's data, device, and settings; DP state is filled in lazily on first use.
	def __init__(
			self,
			client_id: int,
			dataset: Dataset,
			config: Config,
			device: torch.device,
			collator: DataCollatorWithPadding,
			rank: int,
			target_epsilon: float | None,
		) -> None:
		self.client_id = client_id
		self.dataset = dataset
		self.config = config
		self.device = device
		self.collator = collator
		self.rank = rank                        # this client's active LoRA rank r_i (<= config.lora_rank == r_max)
		self.target_epsilon = target_epsilon    # this client's DP budget; None disables DP for this client
		self._privacy_engine = None             # lazily created once, reused every round so epsilon composes across rounds
		self._sigma = None                      # noise multiplier derived from target_epsilon; computed once, cached

	# Train this client's copy of the model for one round, with or without DP, and package up the result.
	def local_update(self, model: PreTrainedModel, adapter_state: dict[str, torch.Tensor]) -> ClientResult:
		model.load_state_dict(adapter_state, strict=False)
		model.train()

		lora_layers: list[HeterogeneousDPLoRALayer] = []
		lora_params: list[torch.nn.Parameter] = []
		for module in model.modules():
			if isinstance(module, HeterogeneousDPLoRALayer):
				module.active_rank = self.rank
				lora_layers.append(module)
				lora_params.extend(module.B.parameters())

		head = list(model.classifier.parameters()) if self.config.train_classifier_head else []

		for parameter in model.parameters():
			parameter.requires_grad = False
		for parameter in (*lora_params, *head):
			parameter.requires_grad = True

		train_dataloader = DataLoader(
			self.dataset,
			batch_size=self.config.batch_size,
			shuffle=True,
			collate_fn=self.collator,
		)

		if self.target_epsilon is not None:
			optimizer = AdamW([*lora_params, *head], lr=self.config.dp_lr)
			total_loss, epsilon_spent, steps_taken = self._local_update_dp(model, optimizer, train_dataloader)
		else:
			optimizer = AdamW([*lora_params, *head], lr=self.config.lr)
			total_loss, steps_taken = self._local_update_plain(model, optimizer, train_dataloader)
			epsilon_spent = None

		# DP noise gets added to every column of B, even the inactive ones (rank r_i..r_max),
		# so they have to be zeroed out again here before the update is sent to the server.
		with torch.no_grad():
			for layer in lora_layers:
				if self.rank < layer.r_max:
					layer.B.weight[:, self.rank:] = 0.0

		updated_state = {
			key: value.detach().cpu().clone()
			for key, value in model.state_dict().items()
			if key.endswith(".B.weight") or "classifier" in key
		}

		return ClientResult(
			state_dict=updated_state,
			n_examples=len(self.dataset),
			n_tokens=0,
			average_loss=total_loss / max(1, steps_taken),
			uploaded_bytes=0,
			epsilon_spent=epsilon_spent,
		)

	# Run ordinary (non-private) training for this client's local steps or epochs.
	def _local_update_plain(self, model: PreTrainedModel, optimizer: Optimizer, train_dataloader: DataLoader) -> tuple[float, int]:
		total_loss = 0.0
		steps_taken = 0

		if self.config.local_epochs is not None:
			for _ in range(self.config.local_epochs):
				for batch in train_dataloader:
					batch = {key: value.to(self.device) for key, value in batch.items()}

					optimizer.zero_grad(set_to_none=True)
					loss = model(**batch).loss
					loss.backward()
					optimizer.step()
					total_loss += float(loss.item())
					steps_taken += 1
		else:
			batches = cycle(train_dataloader)
			for _ in range(self.config.local_steps):
				batch = next(batches)
				batch = {key: value.to(self.device) for key, value in batch.items()}

				optimizer.zero_grad(set_to_none=True)
				loss = model(**batch).loss
				loss.backward()
				optimizer.step()
				total_loss += float(loss.item())
				steps_taken += 1

		return total_loss, steps_taken

	# Run training with DP-SGD (via Opacus): clip and noise every gradient step so no example leaks.
	def _local_update_dp(self, model: PreTrainedModel, optimizer: Optimizer, train_dataloader: DataLoader) -> tuple[float, float, int]:
		from opacus import PrivacyEngine
		from opacus.accountants.utils import get_noise_multiplier

		if self._privacy_engine is None:
			self._privacy_engine = PrivacyEngine()

		sample_rate = min(1.0, self.config.batch_size / len(self.dataset))

		if self._sigma is None:
			if self.config.local_epochs is not None:
				# Shard sizes differ per client under non-iid partitioning, so
				# "steps per epoch" has to be computed per client, not assumed fixed.
				steps_per_epoch = math.ceil(len(self.dataset) / self.config.batch_size)
				local_steps_equivalent = self.config.local_epochs * steps_per_epoch
			else:
				local_steps_equivalent = self.config.local_steps
			expected_total_steps = round(self.config.rounds * self.config.client_sample_rate) * local_steps_equivalent
			self._sigma = get_noise_multiplier(
				target_epsilon=self.target_epsilon,
				target_delta=self.config.delta,
				sample_rate=sample_rate,
				steps=max(1, expected_total_steps),
			)

		dp_model, dp_optimizer, dp_loader = self._privacy_engine.make_private(
			module=model,
			optimizer=optimizer,
			data_loader=train_dataloader,
			noise_multiplier=self._sigma,
			max_grad_norm=self.config.max_grad_norm,
		)

		total_loss = 0.0
		steps_taken = 0

		# Run one training batch and add its loss to the running total.
		def _run_batch(batch: dict) -> None:
			nonlocal total_loss, steps_taken
			if batch["input_ids"].shape[0] == 0:
				# Poisson sampling under DP can emit an empty batch; skip it.
				return
			batch = {key: value.to(self.device) for key, value in batch.items()}

			dp_optimizer.zero_grad(set_to_none=True)
			loss = dp_model(**batch).loss
			loss.backward()
			dp_optimizer.step()
			total_loss += float(loss.item())
			steps_taken += 1

		if self.config.local_epochs is not None:
			for _ in range(self.config.local_epochs):
				if self.config.max_physical_batch_size is not None:
					from opacus.utils.batch_memory_manager import BatchMemoryManager

					with BatchMemoryManager(
						data_loader=dp_loader,
						max_physical_batch_size=self.config.max_physical_batch_size,
						optimizer=dp_optimizer,
					) as memory_safe_loader:
						for batch in memory_safe_loader:
							_run_batch(batch)
				else:
					for batch in dp_loader:
						_run_batch(batch)
		else:
			batches = cycle(dp_loader)
			while steps_taken < self.config.local_steps:
				_run_batch(next(batches))

		epsilon_spent = self._privacy_engine.get_epsilon(self.config.delta)

		# Hand the shared model back in its plain (un-instrumented) form --
		# it's reused by every other client (see server/computation.py).
		if hasattr(dp_model, "to_standard_module"):
			dp_model.to_standard_module()
		elif hasattr(dp_model, "remove_hooks"):
			dp_model.remove_hooks()

		return total_loss, epsilon_spent, steps_taken
