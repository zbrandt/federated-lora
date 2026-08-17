from __future__ import annotations

import torch
import torch.nn as nn
from opacus.optimizers.optimizer import DPOptimizer


class RBLoRA:
    name = 'rblora'

    def set_target_modules(self, model: nn.Module) -> None:
        for name, parameter in model.named_parameters():
            if 'lora_' in name or 'classifier' in name:
                parameter.requires_grad = True
            else:
                parameter.requires_grad = False

    def step(
        self, model: nn.Module, optimizer: DPOptimizer, round: int, step: int
    ) -> None:
        optimizer.step()
        optimizer.zero_grad()

    def aggregate(
        self,
        uploads: dict[int, dict[str, torch.Tensor]],
        num_examples: dict[int, int],
    ) -> tuple[dict[int, dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
        total = sum(num_examples.values())
        weights = {cid: n / total for cid, n in num_examples.items()}
        reference = next(iter(uploads.values()))
        a_keys = [k for k in reference if 'lora_A' in k]
        per_client: dict[int, dict[str, torch.Tensor]] = {cid: {} for cid in uploads}
        eval_state: dict[str, torch.Tensor] = {}

        for a_key in a_keys:

            b_key = a_key.replace('lora_A', 'lora_B')

            r_max = max(upload[a_key].shape[0] for upload in uploads.values())
            n = reference[a_key].shape[1]
            m = reference[b_key].shape[0]

            a_sum = torch.zeros(r_max, n)
            b_sum = torch.zeros(m, r_max)
            slot_weight = torch.zeros(r_max)  # sum of w_i where delta_i,r == 1

            for cid, upload in uploads.items():
                r_i = upload[a_key].shape[0]
                w = weights[cid]
                a_sum[:r_i] += w * upload[a_key].float()
                b_sum[:, :r_i] += w * upload[b_key].float()
                slot_weight[:r_i] += w

            a_global = a_sum / slot_weight.clamp_min(1e-12)[:, None]
            b_global = b_sum / slot_weight.clamp_min(1e-12)[None, :]

            for cid, upload in uploads.items():
                r_i = upload[a_key].shape[0]
                per_client[cid][a_key] = a_global[:r_i].to(upload[a_key].dtype)
                per_client[cid][b_key] = b_global[:, :r_i].to(upload[b_key].dtype)

            eval_state[a_key] = a_global.to(reference[a_key].dtype)
            eval_state[b_key] = b_global.to(reference[b_key].dtype)

        for key in reference:
            if 'lora_' in key:
                continue
            acc = sum(weights[cid] * uploads[cid][key].float() for cid in uploads)
            acc = acc.to(reference[key].dtype)
            eval_state[key] = acc
            for cid in per_client:
                per_client[cid][key] = acc

        return per_client, eval_state
