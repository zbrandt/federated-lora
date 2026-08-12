from __future__ import annotations

import torch

def broadcast(model, global_state: dict[str, torch.Tensor]) -> None:
    model.load_state_dict(global_state, strict=False)
