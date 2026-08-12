"""
A simple rule for picking the smallest LoRA rank that still gets good accuracy
under a given privacy budget. Bigger ranks have more parameters that need noise
added to them, so they lose more accuracy as the privacy budget gets tighter.
"""
from __future__ import annotations

import math

from ffalora.config import Config

# Mean accuracy over the last 5 rounds, seeds 42-46, no DP (results/sweep/rank_sweep_r{r}_epsnone_seed*.json).
ACC_NODP_BY_RANK: dict[int, float] = {2: 0.8308, 4: 0.8754, 6: 0.8589, 8: 0.8529, 16: 0.9278, 32: 0.9271}

# One measured (rank, epsilon, accuracy) point used to calibrate kappa below.
_ANCHOR_RANK = 8
_ANCHOR_EPSILON = 1.0
_ANCHOR_ACC_DP = 0.6254

# Rough size of each client's training data per GLUE task, used when an exact shard size isn't known yet.
GLUE_TRAIN_SIZES: dict[str, int] = {"sst2": 67349, "qnli": 104743, "qqp": 363846, "mnli": 392702}


# Estimate how many training steps a client will actually take, for calibrating the DP noise level.
def expected_dp_steps(config: Config, shard_size: float) -> tuple[float, int]:
    sample_rate = min(1.0, config.batch_size / shard_size)

    if config.local_epochs is not None:
        steps_per_epoch = math.ceil(shard_size / config.batch_size)
        local_steps_equivalent = config.local_epochs * steps_per_epoch
    else:
        local_steps_equivalent = config.local_steps

    expected_total_steps = round(config.rounds * config.client_sample_rate) * local_steps_equivalent
    return sample_rate, max(1, expected_total_steps)


# Look up how much noise Opacus needs to add to hit a target epsilon.
def noise_multiplier(config: Config, target_epsilon: float, shard_size: float) -> float:
    from opacus.accountants.utils import get_noise_multiplier

    sample_rate, steps = expected_dp_steps(config, shard_size)
    return get_noise_multiplier(
        target_epsilon=target_epsilon,
        target_delta=config.delta,
        sample_rate=sample_rate,
        steps=steps,
    )


# The single coefficient linking rank x noise to accuracy lost, fit from the one anchor point above.
def kappa(config: Config, shard_size: float) -> float:
    sigma_anchor = noise_multiplier(config, _ANCHOR_EPSILON, shard_size)
    return (ACC_NODP_BY_RANK[_ANCHOR_RANK] - _ANCHOR_ACC_DP) / (_ANCHOR_RANK * sigma_anchor ** 2)


# Predict a rank's accuracy at a given epsilon using the rule above.
def modeled_accuracy(
    config: Config,
    rank: int,
    target_epsilon: float | None,
    shard_size: float,
    kappa_value: float | None = None,
) -> float:
    if target_epsilon is None:
        return ACC_NODP_BY_RANK[rank]
    if kappa_value is None:
        kappa_value = kappa(config, shard_size)
    sigma_query = noise_multiplier(config, target_epsilon, shard_size)
    return ACC_NODP_BY_RANK[rank] - kappa_value * rank * sigma_query ** 2


# Pick the rank (from a list of candidates) that the rule predicts will score best at a given epsilon.
def recommended_rank(
    config: Config,
    target_epsilon: float | None,
    shard_size: float,
    candidate_ranks: tuple[int, ...] = (2, 4, 6, 8),
) -> int:
    missing = [r for r in candidate_ranks if r not in ACC_NODP_BY_RANK]
    if missing:
        raise ValueError(
            f"no acc_nodp calibration for rank(s) {missing}; add a rank_sweep "
            f"data point for each before including it as a candidate"
        )

    kappa_value = kappa(config, shard_size) if target_epsilon is not None else None
    return max(candidate_ranks, key=lambda r: modeled_accuracy(config, r, target_epsilon, shard_size, kappa_value))
