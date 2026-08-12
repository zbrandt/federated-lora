from __future__ import annotations

import torch

# Copy the current global B/head weights into the shared model before training starts.
def broadcast(model, global_state: dict[str, torch.Tensor]) -> None:
    model.load_state_dict(global_state, strict=False)
