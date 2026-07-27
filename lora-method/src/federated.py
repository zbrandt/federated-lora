"""Federated training loop.

Memory note: clients share ONE model instance. Each client loads the global
LoRA state, trains, and hands back its own LoRA state. Deep-copying the full
125M-parameter model per client per round (the obvious approach) is both slow
and a fast route to OOM once client counts grow.
"""

import time
from typing import Dict

import numpy as np
import torch

from .aggregators import get_aggregator
from .data import get_dataloaders, get_dataset, make_client_loader, split_clients
from .engine import evaluate_model
from .model import build_model, get_lora_state, set_lora_state
from .utils import free_memory, get_device, get_logger, set_seed


def client_config(fl_cfg, client_id: int):
    """Per-client view of the config.

    This is where coupled rank/privacy heterogeneity enters: client k trains
    with rank r_k under budget eps_k. With both lists None you get the
    homogeneous baseline.
    """
    from dataclasses import replace

    overrides = {"epochs": fl_cfg.local_epochs}
    if fl_cfg.client_ranks is not None:
        overrides["rank"] = fl_cfg.client_ranks[client_id]
        overrides["alpha"] = fl_cfg.client_ranks[client_id]
    if fl_cfg.client_epsilons is not None:
        overrides["epsilon"] = fl_cfg.client_epsilons[client_id]
    return replace(fl_cfg, **overrides)


def local_train(model, loader, cfg, device) -> None:
    """Train the shared model in place on one client's data.

    Non-private path only. For per-client DP, route through engine.train so
    each client gets its own accountant (each client accounts to its OWN
    epsilon -- privacy is self-contained per client, which is what keeps the
    coupled setting tractable without a new global accounting theorem).
    """
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    for _ in range(cfg.local_epochs):
        for batch in loader:
            batch = {
                k: v.to(device)
                for k, v in batch.items()
                if k in ("input_ids", "attention_mask", "token_type_ids", "labels")
            }
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


def federated_train(fl_cfg) -> Dict:
    """Run the full federated experiment. Returns metrics + per-round history."""
    log = get_logger()
    set_seed(fl_cfg.seed)
    device = get_device(fl_cfg.device)

    dataset, tokenizer = get_dataset(fl_cfg)
    client_datasets = split_clients(fl_cfg, dataset)
    _, val_loader = get_dataloaders(fl_cfg, dataset, tokenizer)

    if fl_cfg.client_ranks is not None:
        log.warning(
            "Heterogeneous ranks %s require a rank-aware aggregator; "
            "'fedavg' will raise on shape mismatch.",
            fl_cfg.client_ranks,
        )

    model = build_model(fl_cfg)
    model.to(device)
    global_state = get_lora_state(model)

    # HARD GUARD: local_train() has no DP path yet (roadmap step 6). Failing
    # loudly beats silently reporting non-private results as private.
    if fl_cfg.epsilon is not None or fl_cfg.client_epsilons is not None:
        raise NotImplementedError(
            "Per-client DP in the federated loop is not implemented yet "
            "(roadmap step 6). local_train() runs WITHOUT differential privacy, "
            "so any epsilon passed here would be ignored. Wire each client "
            "through its own PrivacyEngine/accountant before using this path."
        )

    aggregate = get_aggregator(fl_cfg.aggregator)
    sizes = [len(d) for d in client_datasets]
    rng = np.random.default_rng(fl_cfg.seed)

    history = []
    t0 = time.time()

    for rnd in range(fl_cfg.rounds):
        selected = (
            list(range(fl_cfg.num_clients))
            if fl_cfg.clients_per_round >= fl_cfg.num_clients
            else sorted(
                rng.choice(fl_cfg.num_clients, fl_cfg.clients_per_round, replace=False).tolist()
            )
        )

        client_states, weights = [], []
        for cid in selected:
            c_cfg = client_config(fl_cfg, cid)
            loader = make_client_loader(c_cfg, client_datasets[cid], tokenizer)

            set_lora_state(model, global_state)  # reset to the global model
            local_train(model, loader, c_cfg, device)

            client_states.append(get_lora_state(model))
            weights.append(sizes[cid])

        global_state = aggregate(client_states, weights=weights)
        set_lora_state(model, global_state)

        metrics = evaluate_model(model, val_loader, device)
        log.info(
            "round %d/%d | clients %s | val_acc %.4f",
            rnd + 1,
            fl_cfg.rounds,
            selected,
            metrics["accuracy"],
        )
        history.append({"round": rnd + 1, "clients": selected, **metrics})

        free_memory()

    final = history[-1]
    return {
        "accuracy": final["accuracy"],
        "eval_loss": final["eval_loss"],
        "runtime_s": round(time.time() - t0, 1),
        "history": history,
        "model": model,
    }
