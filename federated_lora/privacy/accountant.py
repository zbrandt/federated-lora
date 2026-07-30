from __future__ import annotations

from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier


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
