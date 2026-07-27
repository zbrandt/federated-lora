from __future__ import annotations

from dataclasses import dataclass
from itertools import cycle

import torch
from datasets import Dataset
from torch.optim import SGD, AdamW, Optimizer
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, PreTrainedModel

from flora.config import Config


@dataclass(slots=True)
class ClientResult:
	state_dict: dict[str, torch.Tensor]
	n_examples: int
	n_tokens: int
	average_loss: float
	uploaded_bytes: int
	epsilon_spent: float | None = None


class Client:
	def __init__(self, client_id: int, dataset: Dataset, config: Config, device: torch.device, collator: DataCollatorWithPadding) -> None:
		self.client_id = client_id
		self.dataset = dataset
		self.config = config
		self.device = device
		self.collator = collator
		self._privacy_accountant = None  # opacus.accountants.RDPAccountant; lazily created once, reused every round so epsilon composes across rounds

	def local_update(self, model: PreTrainedModel, adapter_state: dict[str, torch.Tensor]) -> ClientResult:
		"""
		Perform a local client update.

		Unlike LA-LoRA, FLoRA trains the LoRA `A` and `B` matrices jointly with a
		single optimizer -- there is no alternating schedule, since FLoRA's
		exactness comes from how the server aggregates client updates (stacking),
		not from decoupling the local gradient.

		Parameters
		----------
		model : PreTrainedModel
			Model with LoRA adapter modules to train.
		adapter_state : dict[str, torch.Tensor]
			The global adapter (and classifier head) state to train from this round.

		Returns
		-------
		ClientResult
			The client's updated adapter state plus loss/example/privacy bookkeeping.
		"""
		model.load_state_dict(adapter_state, strict=False)
		model.train()

		lora_params, head = [], []
		for name, parameter in model.named_parameters():
			if "lora_A" in name or "lora_B" in name:
				lora_params.append(parameter)
			elif "modules_to_save" in name:
				head.append(parameter)

		for name, parameter in model.named_parameters():
			parameter.requires_grad = False
		for parameter in (*lora_params, *head):
			parameter.requires_grad = True

		train_dataloader = DataLoader(
			self.dataset,
			batch_size=self.config.batch_size,
			shuffle=True,
			collate_fn=self.collator,
		)

		if self.config.use_dp:
			# Plain SGD, not AdamW -- see _local_update_dp's docstring for why.
			optimizer = SGD([*lora_params, *head], lr=self.config.dp_lr, momentum=self.config.dp_momentum)
			total_loss, epsilon_spent = self._local_update_dp(model, optimizer, train_dataloader, lora_params, head)
		else:
			optimizer = AdamW([*lora_params, *head], lr=self.config.lr)
			total_loss = self._local_update_plain(model, optimizer, train_dataloader)
			epsilon_spent = None

		updated_state = {
			key: value.detach().cpu().clone()
			for key, value in model.state_dict().items()
			if "lora_" in key or "modules_to_save" in key
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

	def _local_update_dp(
			self,
			model: PreTrainedModel,
			optimizer: Optimizer,
			train_dataloader: DataLoader,
			lora_params: list[torch.nn.Parameter],
			head: list[torch.nn.Parameter],
		) -> tuple[float, float]:
		"""
		Train under client-level DP-SGD, following Liu et al., "Differentially
		Private Low-Rank Adaptation of Large Language Model Using Federated
		Learning" (arXiv:2312.17493), Algorithm 1 -- NOT Opacus's per-example
		DP-SGD (no `GradSampleModule`, no per-sample gradient clipping). Each
		step computes an ordinary mini-batch gradient, clips it (once, not
		once per example) to `config.max_grad_norm`, adds one Gaussian noise
		draw calibrated to `noise_multiplier * max_grad_norm`, then takes a
		plain SGD step -- matching Algorithm 1 lines 13-16 exactly (plain SGD,
		no momentum, is what the paper uses; hence `dp_momentum` defaults to
		0). LoRA A/B and the classifier head are clipped/noised as two
		separate groups, mirroring the paper's independent treatment of A and
		B.

		This protects each client's entire per-step update as the unit of
		privacy (client-level DP) rather than each individual training
		example (Opacus's per-example DP). That's a weaker guarantee, but it
		doesn't depend on a large batch size to keep the noise-to-signal ratio
		usable the way per-example DP-SGD does (noise here is added once to
		the already batch-averaged gradient, not summed per-example and only
		then divided by batch size) -- and unlike three successive attempts at
		tuning Opacus's per-example pipeline for this codebase, this is the
		mechanism an actual published, validated implementation uses.

		Only Opacus's standalone `RDPAccountant` is used here (for `epsilon`
		reporting), not its optimizer/module wrapping -- it composes privacy
		spend across steps/rounds exactly like the rest of Opacus, without
		requiring per-example gradients.
		"""
		from opacus.accountants import RDPAccountant

		if self._privacy_accountant is None:
			self._privacy_accountant = RDPAccountant()

		sample_rate = min(1.0, self.config.batch_size / len(self.dataset))
		batches = cycle(train_dataloader)
		total_loss = 0.0
		for _ in range(self.config.local_steps):
			batch = next(batches)
			batch = {key: value.to(self.device) for key, value in batch.items()}

			optimizer.zero_grad(set_to_none=True)
			loss = model(**batch).loss
			loss.backward()

			for group in (lora_params, head):
				torch.nn.utils.clip_grad_norm_(group, self.config.max_grad_norm)
				for parameter in group:
					parameter.grad += torch.normal(
						mean=0.0,
						std=self.config.noise_multiplier * self.config.max_grad_norm,
						size=parameter.grad.shape,
						device=parameter.grad.device,
					)

			optimizer.step()
			total_loss += float(loss.item())
			self._privacy_accountant.step(noise_multiplier=self.config.noise_multiplier, sample_rate=sample_rate)

		epsilon_spent = self._privacy_accountant.get_epsilon(self.config.delta)

		return total_loss, epsilon_spent
