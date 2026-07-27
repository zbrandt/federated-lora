"""Training / evaluation engine for a single (centralised) client."""

import time
from contextlib import nullcontext
from typing import Dict, Optional

import torch
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.validators import ModuleValidator

from .data import get_dataloaders, get_dataset
from .model import build_model
from .utils import cuda_mem_gb, get_device, get_logger, set_seed

BATCH_KEYS = ("input_ids", "attention_mask", "token_type_ids", "labels")


def _to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items() if k in BATCH_KEYS}


@torch.no_grad()
def evaluate_model(model, loader, device) -> Dict[str, float]:
    """Accuracy + mean loss. Computed directly in torch so no network call
    (the `evaluate` library hits the hub, which many clusters block)."""
    model.eval()
    correct, total, loss_sum, n_batches = 0, 0, 0.0, 0

    for batch in loader:
        batch = _to_device(batch, device)
        out = model(**batch)
        preds = out.logits.argmax(dim=-1)
        correct += (preds == batch["labels"]).sum().item()
        total += batch["labels"].numel()
        loss_sum += out.loss.item()
        n_batches += 1

    model.train()
    return {
        "accuracy": correct / max(total, 1),
        "eval_loss": loss_sum / max(n_batches, 1),
    }


def attach_privacy(model, optimizer, train_loader, cfg):
    """Wrap model/optimizer/loader in the Opacus DP pipeline.

    Returns (model, optimizer, train_loader, privacy_engine). All three
    returned objects are new -- the originals must not be used afterwards.
    """
    log = get_logger()

    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        log.warning("ModuleValidator flagged %d issue(s): %s", len(errors), errors[:3])

    privacy_engine = PrivacyEngine()

    if cfg.noise_multiplier is not None:
        # Diagnostic path: everything DP does EXCEPT adding noise.
        # noise_multiplier=0.0 isolates "is the pipeline wired correctly?"
        # from "is the signal-to-noise ratio too low?".
        log.info("DP pipeline with noise_multiplier=%.3f (no epsilon target)", cfg.noise_multiplier)
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=cfg.noise_multiplier,
            max_grad_norm=cfg.max_grad_norm,
        )
    else:
        log.info("DP-SGD targeting (eps=%.2f, delta=%.0e)", cfg.epsilon, cfg.delta)
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            target_epsilon=cfg.epsilon,
            target_delta=cfg.delta,
            epochs=cfg.epochs,
            max_grad_norm=cfg.max_grad_norm,
        )
        log.info("Solved noise_multiplier sigma = %.4f", optimizer.noise_multiplier)

    return model, optimizer, train_loader, privacy_engine


def train(model, train_loader, val_loader, cfg) -> Dict:
    """Train one model. Returns a metrics dict."""
    log = get_logger()
    device = get_device(cfg.device)
    model.to(device)
    model.train()

    # Filter to trainable params: passing all 125M would allocate optimiser
    # state for the frozen backbone.
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    privacy_engine = None
    if cfg.use_dp:
        model, optimizer, train_loader, privacy_engine = attach_privacy(
            model, optimizer, train_loader, cfg
        )

    history = []
    t0 = time.time()

    for epoch in range(cfg.epochs):
        running_loss, step_count = 0.0, 0

        # A fresh BatchMemoryManager each epoch: it is a single-use context.
        # It keeps the LOGICAL batch (what the accounting sees) at
        # cfg.batch_size while only cfg.max_physical_batch_size examples are
        # resident on the GPU at once.
        if cfg.use_dp:
            batch_context = BatchMemoryManager(
                data_loader=train_loader,
                max_physical_batch_size=cfg.max_physical_batch_size,
                optimizer=optimizer,
            )
        else:
            batch_context = nullcontext(train_loader)

        with batch_context as loader:
            for batch in loader:
                # Poisson sampling under DP can emit an empty batch.
                if batch["input_ids"].shape[0] == 0:
                    continue

                batch = _to_device(batch, device)
                out = model(**batch)
                loss = out.loss

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                running_loss += loss.item()
                step_count += 1

                if cfg.log_every and step_count % cfg.log_every == 0:
                    log.debug(
                        "  epoch %d step %d loss %.4f (mem %.1fGB)",
                        epoch + 1,
                        step_count,
                        running_loss / step_count,
                        cuda_mem_gb(),
                    )

        # NOTE: divide by step_count, not len(train_loader). Under
        # BatchMemoryManager these differ by ~batch_size/max_physical_batch_size
        # (e.g. 32x), which silently inflates the reported loss.
        train_loss = running_loss / max(step_count, 1)
        metrics = evaluate_model(model, val_loader, device)

        log.info(
            "epoch %d/%d | train_loss %.4f | val_acc %.4f | val_loss %.4f",
            epoch + 1,
            cfg.epochs,
            train_loss,
            metrics["accuracy"],
            metrics["eval_loss"],
        )
        history.append({"epoch": epoch + 1, "train_loss": train_loss, **metrics})

    eps_spent = None
    if privacy_engine is not None and cfg.accounts_privacy:
        eps_spent = privacy_engine.get_epsilon(cfg.delta)
        log.info("epsilon spent: %.4f (target %.2f)", eps_spent, cfg.epsilon or float("nan"))

    return {
        "accuracy": history[-1]["accuracy"],
        "eval_loss": history[-1]["eval_loss"],
        "train_loss": history[-1]["train_loss"],
        "epsilon_spent": eps_spent,
        "runtime_s": round(time.time() - t0, 1),
        "history": history,
    }


def run_single(cfg) -> Dict:
    """End-to-end: build everything, train, return a flat result row."""
    set_seed(cfg.seed)
    dataset, tokenizer = get_dataset(cfg)
    train_loader, val_loader = get_dataloaders(cfg, dataset, tokenizer)
    model = build_model(cfg)

    from .model import param_breakdown

    bd = param_breakdown(model)
    results = train(model, train_loader, val_loader, cfg)
    history = results.pop("history")

    row = {
        **cfg.to_flat_dict(),
        **results,
        "trainable_params": bd["total"],
        "lora_params": bd["lora"],
        "head_params": bd["head"],
    }
    return {"row": row, "history": history, "model": model}
