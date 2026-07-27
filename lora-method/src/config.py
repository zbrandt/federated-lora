"""Experiment configuration.

Every knob lives here so a run is fully described by one object, and a sweep
is just a loop over Configs. `to_flat_dict()` is what gets written to CSV.
"""

from dataclasses import dataclass, asdict, field
from typing import Optional, Tuple


@dataclass
class Config:
    # ---------------- model ----------------
    model_name: str = "roberta-base"
    rank: int = 8
    alpha: Optional[int] = None  # defaults to rank -> scaling s = alpha/rank = 1
    lora_dropout: float = 0.0  # keep 0: a second noise source confounds the rank sweep
    target_modules: Tuple[str, ...] = ("query", "value")
    simple_head: bool = True
    # simple_head=True swaps RoBERTa's 592k-param classification head for a
    # single Linear(768, 2) (~1.5k). Under DP every trainable parameter gets
    # noised, so the stock head would dominate the noise budget and mask the
    # effect of rank -- which is the whole variable of interest.

    # ---------------- data ----------------
    dataset_name: str = "sst2"
    max_length: int = 128
    n_train: Optional[int] = None  # subsample train set (None = full). Use for debugging.
    n_val: Optional[int] = None

    # ---------------- optimisation ----------------
    lr: float = 2e-4
    epochs: int = 3
    batch_size: int = 512  # LOGICAL batch: what the privacy accounting sees
    eval_batch_size: int = 64
    max_physical_batch_size: int = 16  # chunk size actually sent to the GPU under DP
    weight_decay: float = 0.0

    # ---------------- privacy ----------------
    epsilon: Optional[float] = None
    delta: float = 1e-5
    max_grad_norm: float = 1.0
    noise_multiplier: Optional[float] = None
    # Exactly one of epsilon / noise_multiplier should normally be set.
    #   epsilon=3.0                       -> Opacus solves for sigma
    #   noise_multiplier=0.0              -> SNR DIAGNOSTIC: full DP pipeline
    #                                        (Poisson sampling, per-sample grads,
    #                                        clipping) with no noise added
    #   both None                         -> non-private baseline

    # ---------------- bookkeeping ----------------
    seed: int = 0
    device: str = "auto"
    out_dir: str = "results"
    run_name: str = ""
    log_every: int = 50

    def __post_init__(self):
        if self.alpha is None:
            self.alpha = self.rank
        if self.epsilon is not None and self.noise_multiplier is not None:
            raise ValueError(
                "Set epsilon OR noise_multiplier, not both. "
                "noise_multiplier is for the sigma=0 diagnostic."
            )

    @property
    def use_dp(self) -> bool:
        """True whenever the Opacus pipeline should be engaged at all."""
        return self.epsilon is not None or self.noise_multiplier is not None

    @property
    def accounts_privacy(self) -> bool:
        """True when a meaningful (eps, delta) can be reported."""
        return self.epsilon is not None or (
            self.noise_multiplier is not None and self.noise_multiplier > 0
        )

    def to_flat_dict(self) -> dict:
        d = asdict(self)
        d["target_modules"] = "+".join(self.target_modules)
        return d


@dataclass
class FLConfig(Config):
    """Federated variant. Inherits every field above."""

    num_clients: int = 5
    rounds: int = 10
    local_epochs: int = 1
    clients_per_round: Optional[int] = None  # None = all clients every round
    aggregator: str = "fedavg"
    split: str = "iid"  # "iid" | "dirichlet"
    dirichlet_beta: float = 0.5

    # --- coupled rank/privacy heterogeneity (the research contribution) ---
    # Leave both None for the homogeneous baseline. Populate with one entry per
    # client to run the heterogeneous setting.
    client_ranks: Optional[Tuple[int, ...]] = None
    client_epsilons: Optional[Tuple[float, ...]] = None

    def __post_init__(self):
        super().__post_init__()
        if self.clients_per_round is None:
            self.clients_per_round = self.num_clients
        for name in ("client_ranks", "client_epsilons"):
            v = getattr(self, name)
            if v is not None and len(v) != self.num_clients:
                raise ValueError(f"{name} must have length num_clients={self.num_clients}")
