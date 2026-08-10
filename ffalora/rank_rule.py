"""
Rule for picking a single global LoRA rank (`Config.lora_rank`) from a
fixed, shared per-client DP epsilon -- the "choose rank based on DP"
direction, not the reverse: `client_epsilons` is normally an
externally-imposed privacy budget, `lora_rank` is the knob actually under
our control, so epsilon is the input and rank is the output.

Model
-----
Under FFA-LoRA (see lora_layer.py) only `B` is ever trained or noised, and
`B`'s parameter count scales linearly with rank while Opacus's per-step
noise variance per parameter -- sigma(eps)**2 * max_grad_norm**2 -- does
NOT depend on rank. So the *total* noise energy injected into B scales
linearly with rank, while the clipped per-example signal norm is capped at
max_grad_norm regardless of rank: the same signal budget gets spread
across more parameters as rank grows. Modeling final accuracy as

    accuracy(r, eps) ~= acc_nodp(r) - kappa * r * sigma(eps)**2

`acc_nodp(r)` is read off the no-DP rank sweep (mean of the last 5 rounds,
seeds 42-46 -- results/sweep/rank_sweep_r{r}_epsnone_seed*.json), which is
close to flat across r in {2,4,6,8} on SST-2/RoBERTa-base (0.831 / 0.875 /
0.859 / 0.853): this task needs almost no LoRA capacity, so there is
essentially no accuracy cost to picking a small rank absent noise. `kappa`
is calibrated from the one pair of runs that isolates the noise effect at
matched epsilon: rank_sweep's r=8 no-DP baseline (0.8529) against
dp_sweep's r=8, eps=1 run (0.6254) -- a drop of 0.2275. This is
corroborated by the grid sweep's r=2, eps=1 point (0.8318, vs the r=2
no-DP baseline of 0.8308 -- essentially zero drop), matching the model's
prediction that a small rank is nearly immune to this same eps=1 noise
because it has 4x fewer noised B parameters than r=8.

The grid sweep (results/sweep/grid_sweep_*) is still running the full
rank x epsilon interaction as of this writing; as more of it lands, use it
to replace ACC_NODP_BY_RANK / the anchor point below with a real fitted
surface instead of this one-anchor linear model.

Calling `get_noise_multiplier` requires Opacus (and, transitively, torch)
to import -- this module cannot be exercised in an environment where torch
itself can't load, same as client.py's `_local_update_dp`.
"""
from __future__ import annotations

import math

from ffalora.config import Config

# Mean of the last 5 rounds, seeds 42-46, no DP.
# results/sweep/rank_sweep_r{r}_epsnone_seed*.json
ACC_NODP_BY_RANK: dict[int, float] = {2: 0.8308, 4: 0.8754, 6: 0.8589, 8: 0.8529, 16: 0.9278, 32: 0.9271}

# Calibration anchor: mean of the last 5 rounds, seeds 42-44.
# results/sweep/dp_sweep_rank8_eps1_seed*.json
_ANCHOR_RANK = 8
_ANCHOR_EPSILON = 1.0
_ANCHOR_ACC_DP = 0.6254

# Static GLUE train-split sizes (nyu-mll/glue), used only to estimate an
# average per-client shard size when the caller doesn't have one on hand
# (e.g. picking a rank before the dataset has been partitioned). Pass an
# actual shard size instead whenever one is available.
GLUE_TRAIN_SIZES: dict[str, int] = {"sst2": 67349, "qnli": 104743, "qqp": 363846, "mnli": 392702}


def expected_dp_steps(config: Config, shard_size: float) -> tuple[float, int]:
    """
    Reproduce client.py's `_local_update_dp` calibration inputs (sample_rate,
    expected_total_steps) from a config and an assumed per-client shard size,
    without needing an actual partitioned Dataset. Pass the client's real
    local example count if known, or `GLUE_TRAIN_SIZES[task] / num_clients`
    as an average-case estimate when picking a rank before partitioning.
    """
    sample_rate = min(1.0, config.batch_size / shard_size)

    if config.local_epochs is not None:
        steps_per_epoch = math.ceil(shard_size / config.batch_size)
        local_steps_equivalent = config.local_epochs * steps_per_epoch
    else:
        local_steps_equivalent = config.local_steps

    expected_total_steps = round(config.rounds * config.client_sample_rate) * local_steps_equivalent
    return sample_rate, max(1, expected_total_steps)


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


def recommended_rank(
    config: Config,
    target_epsilon: float | None,
    shard_size: float,
    candidate_ranks: tuple[int, ...] = (2, 4, 6, 8),
) -> int:
    """
    Pick the rank in `candidate_ranks` that maximizes the modeled
    accuracy(r, eps) = acc_nodp(r) - kappa * r * sigma(eps)**2 (see module
    docstring). With no DP (`target_epsilon is None`) the noise term drops
    out, so this just returns argmax(ACC_NODP_BY_RANK) over the candidates.
    """
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
