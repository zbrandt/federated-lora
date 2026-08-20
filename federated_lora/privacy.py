from __future__ import annotations

import torch
import torch.nn as nn
from opacus import GradSampleModule
from opacus.accountants.utils import get_noise_multiplier
from opacus.optimizers import DPOptimizer


def compute_noise_level(
	num_examples: int,
	batch_size: int,
	global_rounds: int,
	local_steps: int,
	target_epsilon: float | None,
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
	target_epsilon : float | None
		The target privacy loss budget. ``None`` disables differential privacy,
		with a Gaussian noise multiplier of ``0.0``.
	target_delta : float
		The target failure probability of the (epsilon, delta)-DP guarantee.

	Returns
	-------
	float
		The Gaussian noise multiplier for differential privacy.
	"""
	if not target_epsilon:
		return 0.0

	sample_rate = min(1.0, batch_size / max(1, num_examples))
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


def privatize(
	model: GradSampleModule,
	optimizer: DPOptimizer,
) -> list[nn.Parameter]:
	"""
	Privatize model training parameters.

	Parameters
	----------
	model : GradSampleModule
		Wrapped model to compute per-sample gradients.
	optimizer : DPOptimizer
		Wrapped optimizer to clip per-sample gradients and add Gaussian noise.

	Returns
	-------
	list[nn.Parameter] | None
		A list of privatized model training parameters.
	"""
	params = []
	for _, p in model.named_parameters():
		if p.requires_grad and getattr(p, 'grad_sample', None) is not None:
			params.append(p)

	sensitivity = clip_and_accumulate(
		params=params,
		clipping_strategy=optimizer.clipping_strategy,
		max_grad_norm=optimizer.max_grad_norm,
	)
	print(sensitivity)

	add_noise(params, optimizer.noise_multiplier, sensitivity)
	scale_grad(params, optimizer.expected_batch_size)

	return params


def clip_and_accumulate(
	params: list[nn.Parameter],
	clipping_strategy: str,
	max_grad_norm: float,
) -> float:
	"""
	Clip and accumulate per-example gradients.

	Three clipping strategies are supported:
	- ``'median'``: each layer is clipped independently to the median of its own
		per-example gradient-norm distribution.
	- ``'flat'``: the global per-example gradient norm (across all layers) is
		clipped to the single bound ``max_grad_norm``.
	- ``'none'``: no clipping, the per-example gradients are summed as-is.

	Parameters
	----------
	params : list[nn.Parameter]
		The list of parameters with per-example gradients.
	strategy : str
		The clipping strategy (``'median'``, ``'flat'``, or ``'none'``),
		defaults to ``'median'``.
	max_grad_norm : float
		The global clipping bound used by the ``'flat'`` strategy.

	Returns
	-------
	float
		TODO
	"""
	if clipping_strategy == 'none':
		for p in params:
			p.summed_grad = p.grad_sample.sum(dim=0)
		return 0.0

	if clipping_strategy == 'flat':
		per_example_norms = torch.stack(
			[
				p.grad_sample.reshape(len(p.grad_sample), -1).norm(2, dim=-1)
				for p in params
			],
			dim=0,
		).norm(2, dim=0)

		factor = (max_grad_norm / (per_example_norms + 1e-6)).clamp(max=1.0)

		for p in params:
			grad_sample = p.grad_sample
			p.summed_grad = torch.einsum(
				'i,i...->...', factor.to(grad_sample.dtype), grad_sample
			)

		return float(max_grad_norm)

	if clipping_strategy == 'median':
		thresholds = []
		for p in params:
			per_example_norms = p.grad_sample.reshape(
				len(p.grad_sample), -1
			).norm(2, dim=-1)

			c = per_example_norms.median().clamp(min=1e-6)
			factor = (c / (per_example_norms + 1e-6)).clamp(max=1.0)

			p.summed_grad = torch.einsum(
				'i,i...->...',
				factor.to(p.grad_sample.dtype),
				p.grad_sample,
			)
			thresholds.append(float(c))

		return sum(c**2 for c in thresholds) ** 0.5


def add_noise(
	params: list[nn.Parameter], noise_multiplier: float, sensitivity: float
) -> None:
	"""
	Adds noise to clipped gradients.

	Parameters
	----------
	noise_multiplier : float
		The Gaussian noise multiplier for differential privacy.
	sensitivity : float
		TODO
	"""
	if noise_multiplier <= 0 or sensitivity == 0.0:
		return
	for p in params:
		if noise_multiplier > 0:
			p.summed_grad = p.summed_grad + torch.normal(
				mean=0.0,
				std=noise_multiplier * sensitivity,
				size=p.summed_grad.shape,
				device=p.summed_grad.device,
			)


def scale_grad(
	params: list[nn.Parameter],
	expected_batch_size: int,
	# accumulated_iterations: int,
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


def zero_grad(params: list[torch.Tensor]) -> None:
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
