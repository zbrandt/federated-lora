from __future__ import annotations

from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier

def calibrate_noise_multiplier(
	target_epsilon: float,
	target_delta: float,
	sample_rate: float,
	steps: int,
) -> float:
	"""
	Solve for the Gaussian noise multiplier sigma that spends exactly
	``target_epsilon`` over ``steps`` subsampled-Gaussian releases.

	steps is the number of noised gradient releases that touch data — one per
	local step, i.e. ``rounds * local_steps`` (both A- and B-steps consume a
	batch, so both count).
	"""
	return get_noise_multiplier(
		target_epsilon=target_epsilon,
		target_delta=target_delta,
		sample_rate=sample_rate,
		steps=steps,
	)


def achieved_epsilon(
	noise_multiplier: float,
	sample_rate: float,
	steps: int,
	delta: float,
) -> float:
	"""
	Report the epsilon actually spent by ``steps`` releases at the given
	sigma — written into the run JSON so accuracy can be plotted against it.
	"""
	accountant = RDPAccountant()
	for _ in range(steps):
		accountant.step(
			noise_multiplier=noise_multiplier, sample_rate=sample_rate
		)
	return accountant.get_epsilon(delta=delta)