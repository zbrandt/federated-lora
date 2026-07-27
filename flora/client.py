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
		self._privacy_engine = None  # lazily created once, then reused every round so epsilon composes across rounds

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
			# Plain SGD (+momentum), not AdamW, for the DP path. DP noise adds a
			# constant bias to Adam's second-moment estimate, which miscalibrates
			# its adaptive step size specifically for small, low-variance
			# parameters like a rank-8 LoRA adapter -- Adam ends up normalizing
			# every step to ~lr regardless of whether the underlying (noised)
			# gradient carried any real signal, so it can't tell a genuine
			# (tiny) LoRA gradient apart from pure injected noise. With only
			# `local_steps` per round before the adapter is reinitialized from
			# scratch (see server/update.py), there's no time for that bias to
			# wash out either. SGD's step scales directly with the clipped,
			# noised gradient, so it doesn't have this failure mode.
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
		Train under Opacus per-example DP-SGD.

		Each client keeps ONE `PrivacyEngine` for its whole lifetime
		(`self._privacy_engine`), created lazily on first use. Its accountant
		lives on the engine, not the optimizer, so calling `make_private` again
		next round (with a fresh optimizer/loader, since the client's model
		state is reloaded from `global_state` each round) keeps accumulating
		privacy spend across every round this client has participated in --
		exactly the composition guarantee a federated DP client needs.

		Clipping is done per-parameter (`clipping="per_layer"`), not Opacus's
		default flat clipping. Flat clipping computes ONE combined per-example
		norm across every trainable tensor and applies that single clip factor
		to all of them; since the classifier head sits directly on the loss its
		per-example gradients are naturally much larger than the LoRA A/B
		gradients (attenuated by the frozen encoder), so a shared flat norm lets
		the head dictate the clip factor for the adapter too, crushing its
		already-tiny gradient before noise is even added -- the adapter then
		never receives a usable signal and only ever sees the injected noise.
		Per-layer clipping gives the adapter its own (smaller) clip norm via
		`config.lora_max_grad_norm`, independent of the head's.

		`model` is shared across all clients (see server/computation.py), so
		once training finishes we must strip Opacus's grad-sample hooks off it
		before returning -- otherwise the next client's plain forward/backward
		would carry (and be slowed by) this client's per-sample-gradient hooks.
		"""
		from opacus import PrivacyEngine

		if self._privacy_engine is None:
			self._privacy_engine = PrivacyEngine()

		per_param_max_grad_norm = (
			[self.config.lora_max_grad_norm] * len(lora_params)
			+ [self.config.max_grad_norm] * len(head)
		)

		dp_model, dp_optimizer, dp_loader = self._privacy_engine.make_private(
			module=model,
			optimizer=optimizer,
			data_loader=train_dataloader,
			noise_multiplier=self.config.noise_multiplier,
			max_grad_norm=per_param_max_grad_norm,
			clipping="per_layer",
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

		# Hand the shared model back in its plain (un-instrumented) form.
		# `to_standard_module` is Opacus's purpose-built API for this; fall
		# back to `remove_hooks` on older/newer Opacus versions where the
		# wrapper's exact surface differs -- verify against the pinned
		# `opacus` version in pyproject.toml if this ever raises.
		if hasattr(dp_model, "to_standard_module"):
			dp_model.to_standard_module()
		elif hasattr(dp_model, "remove_hooks"):
			dp_model.remove_hooks()

		return total_loss, epsilon_spent
