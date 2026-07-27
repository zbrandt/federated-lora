from __future__ import annotations

import torch

def broadcast(model, global_state: dict[str, torch.Tensor]) -> None:
    """
    Load the global adapter (and classifier head) state into the shared model
    before this round's client computation begins. Base weights are untouched
    here -- they were already folded in permanently by the previous round's
    `update_model` (see server/update.py).
    """
    model.load_state_dict(global_state, strict=False)
