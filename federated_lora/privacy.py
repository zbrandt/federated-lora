from __future__ import annotations

import torch
import torch.nn as nn
from opacus import GradSampleModule
from opacus.accountants.utils import get_noise_multiplier


def compute_noise_level(
	num_examples: int,
	batch_size: int,
	global_rounds: int,
	local_steps: int,
	target_epsilon: float,
	target_delta: float,
) -> float:
	"""
	Compute the Gaussian noise multiplier (sigma) that reaches a total budget of
	(``target_epsilon``, ``target_delta``) over all local DP-SGD steps for one
	client, given its data size and batch size.

	Parameters
	----------
	num_examples : int
		The number of local training examples; sets the sampling rate.
	batch_size : int
		The local batch (lot) size used for Poisson/uniform sampling.
	global_rounds : int
		The number of communication rounds.
	local_steps : int
		The number of local update steps per round.
	target_epsilon : float
		The target privacy loss budget. ``0.0`` disables DP (sigma = 0).
	target_delta : float
		The target failure probability of the (epsilon, delta)-DP guarantee.

	Returns
	-------
	float
		The Gaussian noise multiplier for differential privacy.
	"""
	sample_rate = batch_size / num_examples
	if target_epsilon == 0.0:
		sigma = 0.0
	else:
		steps = global_rounds * local_steps
		sigma = get_noise_multiplier(
			target_epsilon=target_epsilon,
			target_delta=target_delta,
			sample_rate=sample_rate,
			steps=steps,
			accountant='prv',
		)

	print(f'size: {num_examples}, sample_rate: {sample_rate}, sigma: {sigma}')
	return sigma


def get_trainable_with_grad_sample(
		model: GradSampleModule
	) -> list[nn.Parameter]:
	"""
	Get trainable parameters with a per-example gradient.

	Parameters
	----------
	model : GradSampleModule
		The wrapped model for computing per-example gradients.

	Returns
	-------
	list[nn.Parameter]
		A list of parameters with per-example gradients.
	"""
	return [
		p for _, p in model.named_parameters()
		if p.requires_grad and getattr(p, 'grad_sample', None) is not None
	]


def clip_and_accumulate(
		params: list[nn.Parameter]
	) -> dict[nn.Parameter, float]:
	""" 
	Perform gradient clipping on a per-layer basis using median clipping.

	Parameters
	----------
	params : list[nn.Parameter]
		The list of parameters with per-example gradients.

	Returns
	-------
	dict[torch.Tensor, float]
		A dictionary mapping parameters to their clipping threshold.
	"""
	thresholds = {}

	for p in params:
		grad_sample = p.grad_sample

		# get per-example gradient norms
		norms = grad_sample.reshape(len(grad_sample), -1).norm(2, dim=-1)

		# set clipping threshold to median of gradient norm distribution 
		c = norms.median().clamp(min=1e-6)

		# scale down norms greater than the threshold, leave the rest alone
		factor = (c / (norms + 1e-6)).clamp(max=1.0)

		# apply factors and sum over the batch in one shot
		p.summed_grad = torch.einsum('i,i...->...', factor.to(grad_sample.dtype), grad_sample)

		thresholds[p] = float(c)

	return thresholds


def add_noise(
		thresholds: dict[nn.Parameter, float], 
		noise_multiplier: float, 
	) -> None:
	"""
	Adds noise to clipped gradients. Stores clipped and noised result in ``p.grad``

	Parameters
	----------
	thresholds : dict[nn.Parameter, float]
		A dictionary mapping parameters to their clipping threshold.
	noise_multiplier : float
		The Gaussian noise multiplier for differential privacy.
	"""
	for p, c in thresholds.items():
		if noise_multiplier > 0:
			p.summed_grad = p.summed_grad + torch.normal(
				mean=0.0, 
				std=noise_multiplier * c,
				size=p.summed_grad.shape,
				device=p.summed_grad.device, 
			)


def scale_grad(
		params: list[nn.Parameter], 
		expected_batch_size: int,
	) -> None:
	"""
	Divides gradients by ``expected_batch_size``.
	
	Parameters
	----------
	params : list[nn.Parameter]
		The list of parameters with per-example gradients.
	expected_batch_size : int
		The batch_size used for averaging gradients.
	"""
	for p in params:
		p.grad = p.summed_grad / expected_batch_size


def zero_grad(
		params: list[torch.Tensor]
	) -> None:
	"""
	Clear ``p.grad_sample`` and ``p.summed_grad`` from parameters.

	Parameters
	----------
	params : list[torch.Tensor]
		The list of parameters with per-example gradients.
	"""
	for p in params:
		p.summed_grad = None
		p.grad_sample = None


# def perform_accounting(
# 	noise_multiplier: float,
# 	sample_rate: float,
# 	steps: int,
# 	target_delta: float,
# ) -> float:
# 	"""
# 	Perform privacy accounting via Rényi Differential Privacy.

# 	Parameters
# 	----------
# 	noise_multiplier : float
# 		The Gaussian noise multiplier for differential privacy.
# 	sample_rate : float
# 		The local data sampling rate.
# 	steps : int
# 		The product of the number of communication rounds and the number of
# 		local update steps per round.
# 	target_delta : float
# 		TODO

# 	Returns
# 	-------
# 	float
# 		The privacy budget expended from training.
# 	"""
# 	accountant = RDPAccountant()

# 	for _ in range(steps):
# 		# account for the privacy cost of one training step
# 		accountant.step(
# 			noise_multiplier=noise_multiplier,
# 			sample_rate=sample_rate,
# 		)

# 	return accountant.get_epsilon(delta=target_delta)
