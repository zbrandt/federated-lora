from __future__ import annotations

from typing import Callable

CouplingFn = Callable[..., int]


def _snap(value: float, candidate_ranks: tuple[int, ...]) -> int:
    """Round to the nearest available candidate rank."""
    return min(candidate_ranks, key=lambda r: abs(r - value))


def linear_coupling(
    epsilon: float,
    candidate_ranks: tuple[int, ...],
    eps_min: float = 0.5,
    eps_max: float = 8.0,
) -> int:
    """
    Rank scales linearly with epsilon between eps_min and eps_max, snapped
    to the nearest candidate rank.

    Direction: stricter privacy (smaller epsilon, larger DP noise
    multiplier) gets a *smaller* rank. Under DP-SGD, more trainable
    parameters means more capacity that has to be learned through the same
    fixed noise budget, which degrades the effective signal-to-noise ratio
    for a fixed epsilon -- so a client with a tight privacy budget is better
    served by a smaller adapter it can actually learn despite the noise,
    while a client with a loose budget can afford more capacity. epsilon
    outside [eps_min, eps_max] clamps to the nearest end.
    """
    rank_min, rank_max = min(candidate_ranks), max(candidate_ranks)
    t = (epsilon - eps_min) / (eps_max - eps_min)
    t = max(0.0, min(1.0, t))
    target = rank_min + t * (rank_max - rank_min)
    return _snap(target, candidate_ranks)


def threshold_coupling(
    epsilon: float,
    candidate_ranks: tuple[int, ...],
    thresholds: tuple[float, ...] = (1.0, 2.5, 5.0),
) -> int:
    """
    Bucket epsilon into discrete rank tiers by threshold crossing, rather
    than linear_coupling's continuous interpolation. Needs
    len(candidate_ranks) - 1 thresholds (candidate_ranks sorted ascending);
    the default (1.0, 2.5, 5.0) matches the project's standard 4-rank grid
    (2, 8, 16, 32) against the standard epsilon range [0.5, 8.0]. Same
    directionality as linear_coupling: stricter privacy -> smaller rank tier.
    """
    ranks = sorted(candidate_ranks)
    if len(thresholds) != len(ranks) - 1:
        raise ValueError(
            f'need {len(ranks) - 1} thresholds for {len(ranks)} candidate '
            f'ranks, got {len(thresholds)}'
        )
    for rank, threshold in zip(ranks, thresholds):
        if epsilon < threshold:
            return rank
    return ranks[-1]


def client_ranks_from_epsilons(
    client_epsilons: list[float],
    coupling_fn: CouplingFn,
    candidate_ranks: tuple[int, ...],
    **kwargs,
) -> list[int]:
    """Apply a coupling function to every client's epsilon to get its assigned rank."""
    return [coupling_fn(eps, candidate_ranks, **kwargs) for eps in client_epsilons]
