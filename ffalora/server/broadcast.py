from __future__ import annotations

import torch

def broadcast(model, global_state: dict[str, torch.Tensor]) -> None:
    """
    Load the global `B` (per LoRA layer) and classifier head state into the
    shared model before this round's client computation begins. The base
    weights and the frozen, shared `A_max` (see lora_layer.py) are never
    touched here -- under FFA-LoRA, only `B` and the head ever change.
    """
    model.load_state_dict(global_state, strict=False)
