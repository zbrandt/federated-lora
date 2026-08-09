from __future__ import annotations

from pathlib import Path

import numpy as np
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier


def compute_noise_level(
	client_data: dict,
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
	client_data : dict
		TODO
	batch_size : int
		TODO
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
	sample_rate = batch_size / len(client_data['labels'])
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

	print(
		f'size: {len(client_data["labels"])}, '
		f'sample_rate: {sample_rate}, sigma: {sigma}'
	)
	return sigma


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
