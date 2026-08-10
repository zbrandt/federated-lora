"""
Under construction. 
This module contains functions for calibrating the rank of FFA-LoRA layers in a federated learning setting, particularly when differential privacy (DP) is applied.
The goal is to select a rank that maximizes model accuracy while accounting for the noise introduced by DP.
The calibration is based on empirical results from prior experiments, which provide a mapping from rank to expected accuracy without DP, and a method to estimate the noise multiplier for a given DP budget. 
The recommended rank is chosen to maximize a modeled accuracy function that combines the expected accuracy without DP and a penalty term for the noise introduced by DP, which is proportional to the rank and the square of the noise multiplier.
"""
from __future__ import annotations

import math

from flora.config import Config

# Static calibration data from prior experiments, mapping rank to expected accuracy without DP. 
# This is used to estimate the accuracy loss due to DP noise for different ranks.
ACC_NODP_BY_RANK: dict[int, float] = {2: 0.8308, 4: 0.8754, 6: 0.8589, 8: 0.8529}

# Calibration anchor: mean of the last 5 rounds, seeds 42-44.
_ANCHOR_RANK = 8
_ANCHOR_EPSILON = 1.0
_ANCHOR_ACC_DP = 0.6254

# Average per-client shard sizes for GLUE tasks.
GLUE_TRAIN_SIZES: dict[str, int] = {"sst2": 67349, "qnli": 104743, "qqp": 363846, "mnli": 392702}

# Computes the expected sample rate and total number of local update steps for a client given the configuration and its shard size.
# parameters: config: Config - The hyperparameter configuration from config.py. shard_size: float - The number of training examples in the client's local dataset shard.
# returns: tuple[float, int] - A tuple containing the expected sample rate and the total number of local update steps for the client.
def expected_dp_steps(config: Config, shard_size: float) -> tuple[float, int]:
    sample_rate = min(1.0, config.batch_size / shard_size)

    if config.local_epochs is not None:
        steps_per_epoch = math.ceil(shard_size / config.batch_size)
        local_steps_equivalent = config.local_epochs * steps_per_epoch
    else:
        local_steps_equivalent = config.local_steps

    expected_total_steps = round(config.rounds * config.client_sample_rate) * local_steps_equivalent
    return sample_rate, max(1, expected_total_steps)

# Computes the noise multiplier (sigma) for a client given the configuration, its target epsilon, and its shard size. This uses Opacus's method for calibrating sigma to achieve the desired DP guarantee.
# parameters: config: Config - The hyperparameter configuration from config.py. target_epsilon: float - The target epsilon for differential privacy for the client. shard_size: float - The number of training examples in the client's local dataset shard.
# returns: float - The calibrated noise multiplier (sigma) for the client to achieve the target epsilon under DP-SGD.
def noise_multiplier(config: Config, target_epsilon: float, shard_size: float) -> float:
    """Opacus's calibrated sigma for one client at `target_epsilon`, using the same expected-steps math as client.py."""
    from opacus.accountants.utils import get_noise_multiplier

    sample_rate, steps = expected_dp_steps(config, shard_size)
    return get_noise_multiplier(
        target_epsilon=target_epsilon,
        target_delta=config.delta,
        sample_rate=sample_rate,
        steps=steps,
    )

# Recommends a LoRA rank for a client given the configuration, its target epsilon, and its shard size. It selects the rank that maximizes a modeled accuracy function that combines the expected accuracy without DP and a penalty term for the noise introduced by DP.
# parameters: config: Config - The hyperparameter configuration from config.py. target_epsilon: float | None - The target epsilon for differential privacy for the client. If None, the client is non-private. shard_size: float - The number of training examples in the client's local dataset shard.
# candidate_ranks: tuple[int, ...] - A tuple of candidate ranks to consider for selection. Defaults to (2, 4, 6, 8).
# returns: int - The recommended LoRA rank for the client that maximizes the modeled accuracy function. 
def recommended_rank(
    config: Config,
    target_epsilon: float | None,
    shard_size: float,
    candidate_ranks: tuple[int, ...] = (2, 4, 6, 8),
) -> int:
   # Validates that the candidate ranks have corresponding accuracy calibration data. Raises a ValueError if any candidate rank is missing calibration data.
    missing = [r for r in candidate_ranks if r not in ACC_NODP_BY_RANK]
    if missing:
        raise ValueError(
            f"no acc_nodp calibration for rank(s) {missing}; add a rank_sweep "
            f"data point for each before including it as a candidate"
        )

    if target_epsilon is None:
        return max(candidate_ranks, key=lambda r: ACC_NODP_BY_RANK[r])

    sigma_anchor = noise_multiplier(config, _ANCHOR_EPSILON, shard_size)
    kappa = (ACC_NODP_BY_RANK[_ANCHOR_RANK] - _ANCHOR_ACC_DP) / (_ANCHOR_RANK * sigma_anchor ** 2)

    sigma_query = noise_multiplier(config, target_epsilon, shard_size)

    def score(r: int) -> float:
        return ACC_NODP_BY_RANK[r] - kappa * r * sigma_query ** 2

    return max(candidate_ranks, key=score)
