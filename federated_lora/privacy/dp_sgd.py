from __future__ import annotations

import torch
from torch.func import functional_call, grad
from transformers import PreTrainedModel


def privatize(
	model: PreTrainedModel,
	batch: dict[str, torch.Tensor],
	active_key: str,  # 'lora_A' on even steps, 'lora_B' on odd steps
	clip_norm: float,
	noise_multiplier: float,
	generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
	"""
	Compute the DP-SGD gradient for the active LoRA matrix via torch.func.

	Per-example gradients are clipped jointly (one norm over all active params
	per sample), summed, perturbed with Gaussian noise N(0, (C*sigma)^2), and
	divided by the batch size. Only params whose name contains ``active_key``
	are differentiated; everything else is held fixed. Returns name -> gradient
	to be assigned to ``param.grad`` before ``optimizer.step()``.
	"""
	named_params = dict(model.named_parameters())
	active = {n: p for n, p in named_params.items() if active_key in n}
	fixed = {
		n: p.detach() for n, p in named_params.items() if active_key not in n
	}
	fixed |= {n: b.detach() for n, b in model.named_buffers()}

	def sample_loss(active_p, fixed_p, sample):
		merged = {**active_p, **fixed_p}
		inputs = {k: v.unsqueeze(0) for k, v in sample.items()}
		return functional_call(model, merged, (), inputs).loss

	batch_size = next(iter(batch.values())).shape[0]
	grad_fn = grad(sample_loss)

	# Per-sample gradients via torch.func.grad in a loop rather than vmap:
	# vmap cannot trace the model's dropout randomness or the data-dependent
	# control flow in transformers' attention-mask construction. DP-SGD also
	# needs a deterministic forward, so run with dropout disabled and restore
	# the original train/eval mode afterwards.
	was_training = model.training
	model.eval()
	try:
		per_sample_grads = [
			grad_fn(active, fixed, {k: v[i] for k, v in batch.items()})
			for i in range(batch_size)
		]
	finally:
		model.train(was_training)

	# stack each param's per-sample grads into shape (batch, *param.shape)
	per_sample = {
		name: torch.stack([g[name] for g in per_sample_grads])
		for name in active
	}

	# joint per-sample L2 norm across all active params
	flat = torch.cat(
		[g.reshape(batch_size, -1) for g in per_sample.values()], dim=1
	)
	norms = flat.norm(dim=1)
	factor = (clip_norm / (norms + 1e-6)).clamp(max=1.0)  # (batch,)

	grads: dict[str, torch.Tensor] = {}
	for name, g in per_sample.items():
		scaled = g * factor.view(-1, *([1] * (g.dim() - 1)))
		summed = scaled.sum(dim=0)
		noise = clip_norm * noise_multiplier * torch.randn(
			summed.shape, generator=generator, device=summed.device
		)
		grads[name] = (summed + noise) / batch_size

	return grads