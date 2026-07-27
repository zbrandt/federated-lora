from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HeterogeneousDPLoRALayer(nn.Module):
    """
    LoRA adapter with a frozen, shared `A_max` and a rank-heterogeneous,
    DP-SGD-trainable `B` -- FFA-LoRA (Frozen-A LoRA).

    Only `B` is ever trained or noised. `A_max` is a fixed, orthogonal-init
    buffer: no gradient, never touched by the optimizer or by DP noise. This
    is what keeps DP noise linear in the reconstructed update
    `(B + noise) @ A` rather than quadratic `(B + noise)(A + noise)` -- the
    mechanism that made every previous (both-A-and-B-trainable) DP-FLoRA
    attempt fail. It also makes plain weighted FedAvg on `B` exact: since
    `A` is identical and frozen across every client, `mean(B_k) @ A =
    mean(B_k @ A)`, with none of the cross-term aggregation noise that
    motivated FLoRA's stacking trick in the first place.

    Rank heterogeneity: a client assigned rank `r_i` (<= `r_max`) only has
    "access" to the first `r_i` rows of `A_max`. This is implemented by
    zero-padding the projection through `A_max` up to `r_max` before feeding
    it into `B` -- NOT by reshaping `B` itself. `B` stays a real
    `nn.Linear(r_max, out_features, bias=False)` at all times so Opacus's
    built-in per-example gradient hooks for `nn.Linear` work unmodified; a
    raw, manually-sliced `nn.Parameter` wouldn't have a registered grad
    sampler. Since the padded entries are exactly 0 for every example, the
    *true* gradient for `B`'s columns `>= r_i` is exactly 0 too.

    IMPORTANT: under DP-SGD this padding trick alone is NOT sufficient to
    keep those columns at zero. Opacus's noise is added to the whole
    gradient tensor unconditionally, regardless of whether the true
    gradient there happens to be zero -- so client code MUST ALSO explicitly
    zero `B.weight[:, r_i:]` after local training finishes, before the
    state is sent to the server (see `client.py`'s `local_update`). This
    layer only implements the forward/active-rank mechanics; it does not
    (and cannot, from inside a single forward pass) prevent that
    post-training noise leak on its own.
    """

    def __init__(self, base_layer: nn.Linear, r_max: int, lora_alpha: float, lora_dropout: float, seed: int) -> None:
        super().__init__()
        self.base_layer = base_layer
        for parameter in self.base_layer.parameters():
            parameter.requires_grad = False

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        self.r_max = r_max
        self.active_rank = r_max
        self.scaling = lora_alpha / r_max
        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

        # Orthogonal rows so that ANY prefix of r_i < r_max rows is still a
        # well-conditioned rank-r_i projection, not just an arbitrary subset
        # of a Gaussian matrix.
        a_max = torch.empty(r_max, in_features)
        generator = torch.Generator().manual_seed(seed)
        try:
            nn.init.orthogonal_(a_max, generator=generator)
        except TypeError:
            # Older torch without a `generator` kwarg on orthogonal_ -- fall
            # back to temporarily seeding the global RNG so init is at least
            # reproducible run-to-run, even though it's no longer isolated
            # from other torch.* random draws elsewhere in the program.
            rng_state = torch.get_rng_state()
            torch.manual_seed(seed)
            nn.init.orthogonal_(a_max)
            torch.set_rng_state(rng_state)
        self.register_buffer("A_max", a_max, persistent=False)

        self.B = nn.Linear(r_max, out_features, bias=False)
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)

        rank = self.active_rank
        a_active = self.A_max[:rank, :]
        projected = F.linear(self.dropout(x), a_active)
        if rank < self.r_max:
            projected = F.pad(projected, (0, self.r_max - rank))
        lora_out = self.B(projected)

        return base_out + lora_out * self.scaling


def inject_heterogeneous_lora(model: nn.Module, config) -> None:
    """
    Walk `model` and replace every submodule whose attribute name matches
    `config.target_modules` (e.g. "query", "value") with a
    `HeterogeneousDPLoRALayer` wrapping the original `nn.Linear`, in place.

    There is only one shared physical `model` instance in this simulator
    (every client trains it in turn -- see `client_computation` in
    `server/computation.py`), so "A_max shared across all clients" is
    automatic: there's exactly one `A_max` buffer per wrapped layer, not one
    per client, with no synchronization needed.
    """
    target_names = set(config.target_modules)

    for module in model.modules():
        for child_name, child in list(module.named_children()):
            if child_name not in target_names or not isinstance(child, nn.Linear):
                continue

            layer = HeterogeneousDPLoRALayer(
                base_layer=child,
                r_max=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                seed=config.seed,
            )
            layer = layer.to(device=child.weight.device, dtype=child.weight.dtype)
            setattr(module, child_name, layer)
