"""Aggregation strategies for federated LoRA.

Registry-based so new methods drop in without touching the training loop:

    @register("my_method")
    def my_method(client_states, weights=None, **kw): ...

THE CENTRAL ISSUE. Ideal aggregation wants mean(B_k A_k). Vanilla FedAvg over
the factors computes mean(B_k) @ mean(A_k). Expanding two clients:

    (1/2)(B1+B2) @ (1/2)(A1+A2)
        = 1/4 (B1A1 + B1A2 + B2A1 + B2A2)

The cross terms B1A2 and B2A1 are pure noise -- they pair one client's
projection with another client's reconstruction. This inexact-aggregation
error is the seed of both the DP noise-amplification problem and the
rank-heterogeneity problem, and it is what the stacking/SVD methods below
are designed to remove.
"""

from typing import Callable, Dict, List, Optional

import torch

AGGREGATORS: Dict[str, Callable] = {}


def register(name: str):
    def deco(fn):
        AGGREGATORS[name] = fn
        return fn

    return deco


def get_aggregator(name: str) -> Callable:
    if name not in AGGREGATORS:
        raise KeyError(f"Unknown aggregator '{name}'. Available: {sorted(AGGREGATORS)}")
    return AGGREGATORS[name]


def _normalise(weights: Optional[List[float]], n: int) -> List[float]:
    if weights is None:
        return [1.0 / n] * n
    total = float(sum(weights))
    return [w / total for w in weights]


@register("fedavg")
def fedavg(
    client_states: List[Dict[str, torch.Tensor]],
    weights: Optional[List[float]] = None,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """Weighted average of each LoRA tensor independently.

    This is the FedIT baseline. It is INEXACT (see module docstring) and it
    requires every client to share the same rank, since tensors of different
    shapes cannot be stacked. Correct as a baseline; not a target method.
    """
    w = _normalise(weights, len(client_states))
    keys = client_states[0].keys()

    out = {}
    for key in keys:
        shapes = {tuple(s[key].shape) for s in client_states}
        if len(shapes) > 1:
            raise ValueError(
                f"fedavg requires homogeneous ranks; got shapes {shapes} for '{key}'. "
                "Use a rank-heterogeneous aggregator instead."
            )
        stacked = torch.stack([s[key].float() for s in client_states])
        coef = torch.tensor(w, dtype=stacked.dtype).view(-1, *([1] * (stacked.dim() - 1)))
        out[key] = (stacked * coef).sum(dim=0)
    return out


# ---------------------------------------------------------------------------
# Skeletons for the rank-heterogeneous methods. Fill these in as the project
# moves to Phase 2/3 -- each has a different failure mode worth measuring.
# ---------------------------------------------------------------------------


@register("zero_pad")
def zero_pad(client_states, weights=None, max_rank: int = None, **kwargs):
    """HetLoRA-style: zero-pad every client up to max_rank, average, truncate.

    TODO. Watch for: padding introduces aggregation bias, and under DP the
    padded coordinates carry noise but no signal, so the truncation step
    discards a rank-varying noise-contaminated subspace.
    """
    raise NotImplementedError("zero_pad aggregation not implemented yet")


@register("stack")
def stack(client_states, weights=None, **kwargs):
    """FLoRA-style: concatenate A's and B's along the rank dimension.

    Exact -- B_cat @ A_cat reproduces sum_k B_k A_k with no cross terms, and
    heterogeneous ranks concatenate natively. Cost: the aggregate rank grows
    with the number of clients, hence the re-compression step below.

    TODO.
    """
    raise NotImplementedError("stack aggregation not implemented yet")


@register("stack_svd")
def stack_svd(client_states, weights=None, target_rank: int = None, **kwargs):
    """Stack, then truncated-SVD back down to target_rank (SDFLoRA-style).

    This is the operator whose interaction with DP noise is the open question:
    how Gaussian noise redistributes across singular directions depends on the
    spectrum, which differs per client and per round. No sensitivity analysis
    for it exists in the literature.

    TODO.
    """
    raise NotImplementedError("stack_svd aggregation not implemented yet")
