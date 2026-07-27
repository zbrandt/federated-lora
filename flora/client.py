from __future__ import annotations

from dataclasses import dataclass
from itertools import cycle

import torch
from datasets import Dataset
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, PreTrainedModel

from flora.config import Config
from flora.lora_layer import HeterogeneousDPLoRALayer


@dataclass(slots=True)
class ClientResult:
	state_dict: dict[str, torch.Tensor]
	n_examples: int
	n_tokens: int
	average_loss: float
	uploaded_bytes: int
	epsilon_spent: float | None = None


class Client:
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

	def local_update(self, model: PreTrainedModel, adapter_state: dict[str, torch.Tensor]) -> ClientResult:
		"""
		Perform a local client update under FFA-LoRA (Frozen-A LoRA, see
		lora_layer.py): only `B` (per LoRA layer) and the classifier head are
		ever trained. The shared `A_max` is frozen and never appears here --
		not in `adapter_state`, not in the optimizer, not in DP clipping/noise.

		Parameters
		----------
		model : PreTrainedModel
			Model with HeterogeneousDPLoRALayer adapters injected.
		adapter_state : dict[str, torch.Tensor]
			The global `B` (per layer) and classifier head state to train
			from this round.

		Returns
		-------
		ClientResult
			The client's updated state plus loss/example/privacy bookkeeping.
		"""
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
			total_loss, epsilon_spent = self._local_update_dp(model, optimizer, train_dataloader)
		else:
			optimizer = AdamW([*lora_params, *head], lr=self.config.lr)
			total_loss = self._local_update_plain(model, optimizer, train_dataloader)
			epsilon_spent = None

		# See lora_layer.py's docstring: zero-padding the forward pass alone
		# is NOT sufficient to keep inactive columns (rank r_i..r_max) at
		# zero once DP noise is involved -- Opacus's noise is added to the
		# whole gradient tensor unconditionally, regardless of whether the
		# true gradient there happens to be zero. So those columns are
		# explicitly reset here, on every path (DP or not, for consistency),
		# before the state is extracted for upload.
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
			average_loss=total_loss / self.config.local_steps,
			uploaded_bytes=0,
			epsilon_spent=epsilon_spent,
		)

	def _local_update_plain(self, model: PreTrainedModel, optimizer: Optimizer, train_dataloader: DataLoader) -> float:
		batches = cycle(train_dataloader)
		total_loss = 0.0
		for _ in range(self.config.local_steps):
			batch = next(batches)
			batch = {key: value.to(self.device) for key, value in batch.items()}

			optimizer.zero_grad(set_to_none=True)
			loss = model(**batch).loss
			loss.backward()
			optimizer.step()
			total_loss += float(loss.item())

		return total_loss

	def _local_update_dp(self, model: PreTrainedModel, optimizer: Optimizer, train_dataloader: DataLoader) -> tuple[float, float]:
		"""
		Train B (+ classifier head) under Opacus per-example DP-SGD.

		Flat clipping across B + head combined (NOT per-layer clipping --
		Opacus's `DPPerLayerOptimizer` collapses a list of per-layer clip
		norms into a single *aggregate* L2 value and uses that much larger
		number to scale the noise added to every parameter, which silently
		inflates the injected noise -- a real bug hit and reverted in an
		earlier iteration of this codebase).

		`A_max` is a buffer, not a parameter, so it never appears in
		`optimizer` and is therefore never clipped, noised, or updated by
		Opacus -- only `B` and the head are. This is what keeps the noise in
		the reconstructed update `(B + noise) @ A` linear rather than
		quadratic, unlike every previous (both-A-and-B-trainable) DP-FLoRA
		attempt.

		Each client keeps ONE `PrivacyEngine` for its whole lifetime, reused
		every round so its accountant's epsilon composes correctly across
		every round this client is actually selected for.
		"""
		from opacus import PrivacyEngine
		from opacus.accountants.utils import get_noise_multiplier

		if self._privacy_engine is None:
			self._privacy_engine = PrivacyEngine()

		if self._sigma is None:
			sample_rate = min(1.0, self.config.batch_size / len(self.dataset))
			# A client's total future participation count isn't known in
			# advance (depends on random per-round selection), so calibrate
			# sigma against the EXPECTED total local steps across the whole
			# run instead. A client selected more/less than this expectation
			# will over/under-spend its nominal target_epsilon somewhat.
			expected_total_steps = round(self.config.rounds * self.config.client_sample_rate) * self.config.local_steps
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

		batches = cycle(dp_loader)
		total_loss = 0.0
		steps_taken = 0
		while steps_taken < self.config.local_steps:
			batch = next(batches)
			if batch["input_ids"].shape[0] == 0:
				# Poisson sampling under DP can emit an empty batch; skip it.
				continue
			batch = {key: value.to(self.device) for key, value in batch.items()}

			dp_optimizer.zero_grad(set_to_none=True)
			loss = dp_model(**batch).loss
			loss.backward()
			dp_optimizer.step()
			total_loss += float(loss.item())
			steps_taken += 1

		epsilon_spent = self._privacy_engine.get_epsilon(self.config.delta)

		# Hand the shared model back in its plain (un-instrumented) form --
		# it's reused by every other client (see server/computation.py).
		if hasattr(dp_model, "to_standard_module"):
			dp_model.to_standard_module()
		elif hasattr(dp_model, "remove_hooks"):
			dp_model.remove_hooks()

		return total_loss, epsilon_spent
