"""Model construction: backbone + LoRA adapters + (optionally) a slim head."""

from typing import Dict

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification

from .data import get_num_labels
from .utils import get_logger


class SlimClassificationHead(nn.Module):
    """A single Linear on the <s> token, replacing RoBERTa's two-layer head.

    Why this matters for DP: the stock RobertaClassificationHead is
        dense    : 768 x 768 + 768 = 590,592 params
        out_proj : 768 x   2 +   2 =   1,538 params
    i.e. ~592k trainable parameters, versus ~295k for rank-8 LoRA. Under
    DP-SGD every trainable parameter receives Gaussian noise, so two-thirds of
    the noise budget would be spent on a randomly-initialised head rather than
    on the adapters -- and the noised-parameter count would barely move as rank
    varies, washing out the exact effect the rank sweep is trying to measure.

    Replacing only `out_proj` does NOT help: it was already tiny. The 590k
    `dense` layer is the one that has to go.
    """

    def __init__(self, hidden_size: int, num_labels: int):
        super().__init__()
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, features, **kwargs):
        return self.out_proj(features[:, 0, :])  # <s> / [CLS] token


def count_trainable(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def param_breakdown(model) -> Dict[str, int]:
    """Split trainable params into adapter vs head. Log this every run."""
    lora, head, other = 0, 0, 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "lora" in name:
            lora += p.numel()
        elif "classifier" in name or "score" in name:
            head += p.numel()
        else:
            other += p.numel()
    return {"lora": lora, "head": head, "other": other, "total": lora + head + other}


def build_model(cfg, rank: int = None):
    """Build a LoRA-adapted classifier.

    `rank` overrides cfg.rank, which is what lets different federated clients
    hold different-rank adapters.
    """
    log = get_logger()
    r = cfg.rank if rank is None else rank
    alpha = r  # keep scaling s = alpha/r = 1 so rank is the ONLY thing varying

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=get_num_labels(cfg.dataset_name),
    )

    # Must happen BEFORE get_peft_model, so PEFT wraps the head we actually want.
    if cfg.simple_head:
        model.classifier = SlimClassificationHead(
            model.config.hidden_size,
            get_num_labels(cfg.dataset_name),
        )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=r,
        lora_alpha=alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.target_modules),
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    bd = param_breakdown(model)
    log.info(
        "rank=%d | trainable %s (lora=%s, head=%s, other=%s)",
        r,
        f"{bd['total']:,}",
        f"{bd['lora']:,}",
        f"{bd['head']:,}",
        f"{bd['other']:,}",
    )
    return model


def get_lora_state(model, to_cpu: bool = True) -> Dict[str, torch.Tensor]:
    """Extract LoRA A/B tensors -- the only thing clients transmit."""
    state = {}
    for name, p in model.named_parameters():
        if "lora" in name:
            t = p.detach()
            state[name] = t.cpu().clone() if to_cpu else t.clone()
    return state


def set_lora_state(model, state: Dict[str, torch.Tensor]) -> None:
    """Copy LoRA tensors back into a model in place (no deepcopy needed)."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in state:
                p.copy_(state[name].to(p.device))


def get_head_state(model, to_cpu: bool = True) -> Dict[str, torch.Tensor]:
    state = {}
    for name, p in model.named_parameters():
        if p.requires_grad and ("classifier" in name or "score" in name):
            t = p.detach()
            state[name] = t.cpu().clone() if to_cpu else t.clone()
    return state


def set_head_state(model, state: Dict[str, torch.Tensor]) -> None:
    set_lora_state(model, state)  # same copy-by-name logic
