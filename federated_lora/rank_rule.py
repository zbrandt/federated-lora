"""
A simple rule for picking the smallest LoRA rank that still gets good accuracy
under a given privacy budget. Bigger ranks have more parameters that need noise
added to them, so they lose more accuracy as the privacy budget gets tighter.

ACC_NODP_BY_RANK and KAPPA are fit by `python -m rankrule.fit_rank_rule` against
results/sweep/rank_sweep_r*_epsnone_seed*.json (no-DP calibration) and
results/sweep/grid_sweep_r*_eps*_seed*.json (DP grid) -- rerun and re-paste the
printed values below whenever more of either sweep lands.
"""
from __future__ import annotations

from federated_lora.config import Config
from federated_lora.privacy import compute_noise_level

# Rough per-client shard size by task, used when an exact per-client count
# isn't available (the non-iid Dirichlet split makes it vary client to
# client; this is just the dataset's train-split size divided evenly).
TRAIN_SIZE_BY_TASK: dict[str, int] = {'cifar100': 50000}

# accuracy(rank, sigma) = ACC_NODP_BY_RANK[rank] - KAPPA * rank * sigma^2
# Fit 2026-08-20 over results/sweep's rank_sweep + grid_sweep data (seeds
# 42-46, cifar100): 20 (rank, epsilon) cells across 6 ranks, R^2=0.940. See
# `python -m rankrule.fit_rank_rule` output for per-point residuals.
ACC_NODP_BY_RANK: dict[int, float] = {
	2: 0.8687,
	4: 0.8764,
	6: 0.8926,
	8: 0.8954,
	16: 0.9376,
	32: 0.9271,
}
KAPPA: float = 0.001432


def shard_size(config: Config) -> float:
	return TRAIN_SIZE_BY_TASK[config.task] / config.num_clients


def noise_multiplier(config: Config, target_epsilon: float | None, size: float) -> float:
	"""Sigma Opacus needs to hit target_epsilon over this run's rounds/steps -- same accounting as the real training path (federated_lora.privacy.compute_noise_level)."""
	if target_epsilon is None:
		return 0.0
	return compute_noise_level(
		num_examples=size,
		batch_size=config.batch_size,
		global_rounds=config.global_rounds,
		local_steps=config.local_steps,
		target_epsilon=target_epsilon,
		target_delta=config.target_delta,
	)


def modeled_accuracy(
	config: Config,
	rank: int,
	target_epsilon: float | None,
	size: float,
) -> float:
	"""Predict a rank's accuracy at a given epsilon using the fitted rule above."""
	if target_epsilon is None:
		return ACC_NODP_BY_RANK[rank]
	sigma = noise_multiplier(config, target_epsilon, size)
	return ACC_NODP_BY_RANK[rank] - KAPPA * rank * sigma ** 2


def recommended_rank(
	config: Config,
	target_epsilon: float | None,
	size: float,
	candidate_ranks: tuple[int, ...] = (2, 8, 16, 32),
) -> int:
	"""Pick the rank (from candidate_ranks) the rule predicts will score best at target_epsilon."""
	missing = [r for r in candidate_ranks if r not in ACC_NODP_BY_RANK]
	if missing:
		raise ValueError(
			f'no acc_nodp calibration for rank(s) {missing}; add a rank_sweep '
			f'data point for each before including it as a candidate'
		)
	return max(candidate_ranks, key=lambda r: modeled_accuracy(config, r, target_epsilon, size))
