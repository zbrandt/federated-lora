"""
Plain FLoRA: both A and B are trainable per client. Aggregation uses the
stacking trick instead of naively averaging A and B separately, then
compresses back down to the original rank via truncated SVD.
"""

from __future__ import annotations

from typing import Protocol

import torch
import torch.nn as nn
from opacus.optimizers.optimizer import DPOptimizer

from flora.config import Config
from flora.lora_layer import inject_lora


class Method(Protocol):
    """
    A federated fine-tuning method.
    """

    name: str

    def set_target_modules(self, model: nn.Module) -> None:
        """
        Configure which modules of the model to adapt.

        Parameters
        ----------
        model : nn.Module
            The PEFT model.
        """
        ...

    def step(
        self, model: nn.Module, optimizer: DPOptimizer, round: int, step: int
    ) -> None:
        """
        Perform a single optimization step to update trainable parameters.

        Parameters
        ----------
        model : nn.Module
            The local PEFT model.
        optimizer : DPOptimizer
            The wrapper that adds additional functionality to clip per sample
            gradients and add Gaussian noise.
        round : int
            The current global communication round.
        step : int
            The current local step.
        """
        ...

    def aggregate(
        self,
        uploads: list[dict[str, torch.Tensor]],
        num_examples: list[int],
    ) -> dict[str, torch.Tensor]:
        """
        Aggregate client trainable parameter state-dict updates to the model.

        Parameters
        ----------
        uploads : list[dict[str, torch.Tensor]]
            The client trainable parameter state-dict updates to the model.
        num_examples : list[int]
            The total number of examples of each client.

        Returns
        -------
        dict[str, torch.Tensor]
            The aggregated client updates to the model.
        """
        ...


class FLoRA:
    name = "flora"

    def __init__(self, config: Config) -> None:
        self.config = config

    def set_target_modules(self, model: nn.Module) -> None:
        inject_lora(model, self.config)

    def step(self, model: nn.Module, optimizer: DPOptimizer, round: int, step: int) -> None:
        optimizer.step()

    def aggregate(
        self,
        uploads: list[dict[str, torch.Tensor]],
        num_examples: list[int],
    ) -> dict[str, torch.Tensor]:
        total = float(sum(num_examples))
        weights = [n / total for n in num_examples]
        keys = uploads[0].keys()

        a_suffix, b_suffix = ".A.weight", ".B.weight"
        prefixes = {key[: -len(a_suffix)] for key in keys if key.endswith(a_suffix)}
        aggregated: dict[str, torch.Tensor] = {}

        for prefix in prefixes:
            a_key, b_key = prefix + a_suffix, prefix + b_suffix
            rank = uploads[0][a_key].shape[0]

            a_stack = torch.cat([upload[a_key].to(torch.float32) for upload in uploads], dim=0)
            b_stack = torch.cat(
                [upload[b_key].to(torch.float32) * weight for upload, weight in zip(uploads, weights)], dim=1
            )
            delta_w = b_stack @ a_stack

            U, S, Vt = torch.linalg.svd(delta_w, full_matrices=False)
            sqrt_s = S[:rank].clamp_min(0).sqrt()
            new_b = U[:, :rank] * sqrt_s.unsqueeze(0)
            new_a = sqrt_s.unsqueeze(1) * Vt[:rank, :]

            ref_a, ref_b = uploads[0][a_key], uploads[0][b_key]
            aggregated[a_key] = new_a.to(dtype=ref_a.dtype, device=ref_a.device)
            aggregated[b_key] = new_b.to(dtype=ref_b.dtype, device=ref_b.device)

        for key in keys:
            if key.endswith(a_suffix) or key.endswith(b_suffix):
                continue
            reference = uploads[0][key]
            accumulator = torch.zeros_like(reference, dtype=torch.float32)
            for upload, weight in zip(uploads, weights):
                accumulator += upload[key].to(torch.float32) * weight
            aggregated[key] = accumulator.to(dtype=reference.dtype, device=reference.device)

        return aggregated
