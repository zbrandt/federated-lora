from __future__ import annotations

from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier


def compute_noise_level(
	target_epsilon: float, target_delta: float, sample_rate: float, steps: int
) -> float:
	"""
	Compute the noise level sigma to reach a total budget of
	(``target_epsilon``, ``target_delta``) at the end of steps, with a given
	``sample_rate``.

	Parameters
	----------
	target_epsilon : float
		The target privacy loss budget.
	target_delta : float
		TODO
	sample_rate : float
		The local data sampling rate.
	steps : int
		The product of the number of communication rounds and the number of
		local update steps per round.
	Returns
	-------
	float
		The Gaussian noise multiplier for differential privacy.
	"""
	if target_epsilon == 0.0:
		noise_multiplier = 0.0
	else:
		noise_multiplier = get_noise_multiplier(
			target_epsilon=target_epsilon,
			target_delta=target_delta,
			sample_rate=sample_rate,
			steps=steps,
		)

	return noise_multiplier


def perform_accounting(
	noise_multiplier: float,
	sample_rate: float,
	steps: int,
	target_delta: float,
) -> float:
	"""
	Perform privacy accounting via Rényi Differential Privacy.

	Parameters
	----------
	noise_multiplier : float
		The Gaussian noise multiplier for differential privacy.
	sample_rate : float
		The local data sampling rate.
	steps : int
		The product of the number of communication rounds and the number of
		local update steps per round.
	target_delta : float
		TODO

	Returns
	-------
	float
		The privacy budget expended from training.
	"""
	accountant = RDPAccountant()

	for _ in range(steps):
		# account for the privacy cost of one training step
		accountant.step(
			noise_multiplier=noise_multiplier,
			sample_rate=sample_rate,
		)

	return accountant.get_epsilon(delta=target_delta)
