from __future__ import annotations

import torch
import torch.nn as nn
from opacus import GradSampleModule
from opacus.accountants.utils import get_noise_multiplier
from opacus.optimizers import DPOptimizer

from federated_lora.metrics import pre_clip_norm_stats


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
) -> list[nn.Parameter] | None:
	"""
	Clip per sample gradients and add Gaussian noise.

	Aggregate ``p.grad_sample`` over all parameters to calculate per sample
	norms. Clip ``p.grad_sample`` according to the optimizer's clipping strategy
	(``'median'``, ``'flat'``, or ``'none'``). Aggregate clipped per sample
	gradients into ``p.summed_grad``. Add Gaussian noise to ``p.summed_grad``
	calibrated to the noise multiplier and the clipping threshold. Divide
	gradients by ``expected_batch_size`` into ``p.grad``.

	Parameters
	----------
	model : GradSampleModule
		The wrapped model carrying per-example gradients.
	optimizer : DPOptimizer
		The wrapped optimizer carrying the noise multiplier and expected batch
		size.

	Returns
	-------
	list[nn.Parameter] | None
		A list of the privatized parameters or ``None`` for an empty batch.
	"""
	named = {}
	for n, p in model.named_parameters():
		if p.requires_grad and getattr(p, 'grad_sample', None) is not None:
			named[n] = p

	if not named:
		optimizer.last_diagnostics = None
		optimizer.zero_grad()
		return

	strategy = getattr(optimizer, 'clipping_strategy', 'median')

	params = list(named.values())
	norm_mean, norm_median = pre_clip_norm_stats(params)
	thresholds = clip_and_accumulate(params, strategy, optimizer.max_grad_norm)
	add_noise(thresholds, optimizer.noise_multiplier)
	scale_grad(params, optimizer.expected_batch_size)

	optimizer.last_diagnostics = {
		'strategy': strategy,
		'grad_norm_mean': norm_mean,
		'grad_norm_median': norm_median,
		'clip_thresholds': {n: thresholds[p] for n, p in named.items()},
	}

	return params


def clip_and_accumulate(
	params: list[nn.Parameter],
	strategy: str = 'median',
	max_grad_norm: float = 1.0,
) -> dict[nn.Parameter, float]:
	"""
	Clip per-example gradients and sum them into ``p.summed_grad``.

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
		The clipping strategy (``'median'``, ``'flat'``, or ``'none'``).
		Defaults to ``'median'``.
	max_grad_norm : float
		The global clipping bound used by the ``'flat'`` strategy.

	Returns
	-------
	dict[nn.Parameter, float]
		A dictionary mapping each parameter to the clipping threshold applied
		to it. ``add_noise`` uses these to calibrate the per-layer noise scale.
	"""
	thresholds: dict[nn.Parameter, float] = {}

	if strategy == 'flat':
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
			thresholds[p] = float(max_grad_norm)
		return thresholds

	for p in params:
		grad_sample = p.grad_sample

		if strategy == 'none':
			p.summed_grad = grad_sample.sum(dim=0)
			thresholds[p] = float('inf')
			continue

		# get per-example gradient norms
		norms = grad_sample.reshape(len(grad_sample), -1).norm(2, dim=-1)

		# set clipping threshold to median of gradient norm distribution
		c = norms.median().clamp(min=1e-6)

		# scale down norms greater than the threshold, leave the rest alone
		factor = (c / (norms + 1e-6)).clamp(max=1.0)

		# apply factors and sum over the batch in one shot
		p.summed_grad = torch.einsum(
			'i,i...->...', factor.to(grad_sample.dtype), grad_sample
		)

		thresholds[p] = float(c)

	return thresholds


def add_noise(
	thresholds: dict[nn.Parameter, float],
	noise_multiplier: float,
) -> None:
	"""
	Adds noise to clipped gradients. Stores clipped and noised result in
	``p.summed_grad``.

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
				std=noise_multiplier
				* c,  # calibrate with the clipping threshold
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
