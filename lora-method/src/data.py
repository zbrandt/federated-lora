"""GLUE data loading, tokenisation and federated splitting.

Tokenisation is cached in-process so a sweep of 50 runs doesn't re-tokenise
67k examples 50 times.
"""

from typing import Dict, List, Tuple

import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding

from .utils import get_logger

# Which column(s) hold the text, and which split to evaluate on.
TASK_INFO = {
    "sst2": {"keys": ("sentence", None), "val": "validation", "num_labels": 2},
    "qnli": {"keys": ("question", "sentence"), "val": "validation", "num_labels": 2},
    "qqp": {"keys": ("question1", "question2"), "val": "validation", "num_labels": 2},
    "mnli": {"keys": ("premise", "hypothesis"), "val": "validation_matched", "num_labels": 3},
}

_CACHE: Dict[tuple, tuple] = {}


def get_num_labels(dataset_name: str) -> int:
    return TASK_INFO[dataset_name]["num_labels"]


def get_dataset(cfg):
    """Return (dataset_dict, tokenizer). Cached on (task, model, max_length)."""
    key = (cfg.dataset_name, cfg.model_name, cfg.max_length)
    if key in _CACHE:
        return _CACHE[key]

    log = get_logger()
    info = TASK_INFO[cfg.dataset_name]
    k1, k2 = info["keys"]

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    raw = load_dataset("glue", cfg.dataset_name)

    def tokenize(batch):
        args = (batch[k1],) if k2 is None else (batch[k1], batch[k2])
        return tokenizer(*args, truncation=True, max_length=cfg.max_length)

    ds = raw.map(tokenize, batched=True)

    drop = [c for c in ("idx", k1, k2) if c and c in ds["train"].column_names]
    ds = ds.remove_columns(drop)
    # HF models only compute loss internally when the column is called "labels".
    if "label" in ds["train"].column_names:
        ds = ds.rename_column("label", "labels")
    ds.set_format("torch")

    log.info(
        "Loaded %s: %d train / %d val",
        cfg.dataset_name,
        len(ds["train"]),
        len(ds[info["val"]]),
    )
    _CACHE[key] = (ds, tokenizer)
    return ds, tokenizer


def get_splits(cfg, dataset):
    """Return (train_split, val_split), applying optional subsampling."""
    info = TASK_INFO[cfg.dataset_name]
    train = dataset["train"]
    val = dataset[info["val"]]

    if cfg.n_train is not None and cfg.n_train < len(train):
        train = train.shuffle(seed=cfg.seed).select(range(cfg.n_train))
    if cfg.n_val is not None and cfg.n_val < len(val):
        val = val.select(range(cfg.n_val))
    return train, val


def get_dataloaders(cfg, dataset, tokenizer) -> Tuple[DataLoader, DataLoader]:
    collator = DataCollatorWithPadding(tokenizer)
    train, val = get_splits(cfg, dataset)

    train_loader = DataLoader(
        train,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,  # Opacus replaces the sampler; keep this simple
        drop_last=False,
    )
    val_loader = DataLoader(
        val,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )
    return train_loader, val_loader


# --------------------------------------------------------------------------
# Federated splitting
# --------------------------------------------------------------------------


def split_iid(train, num_clients: int, seed: int = 42) -> List:
    shuffled = train.shuffle(seed=seed)
    n = len(shuffled) // num_clients
    return [shuffled.select(range(i * n, (i + 1) * n)) for i in range(num_clients)]


def split_dirichlet(train, num_clients: int, beta: float = 0.5, seed: int = 42) -> List:
    """Label-skewed non-IID split. Smaller beta => more skew.

    This is the standard FL heterogeneity knob and is separate from the rank /
    privacy heterogeneity the project is actually about -- keep them distinct
    when reporting results.
    """
    rng = np.random.default_rng(seed)
    labels = np.array(train["labels"])
    classes = np.unique(labels)

    idx_per_client: List[List[int]] = [[] for _ in range(num_clients)]
    for c in classes:
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        props = rng.dirichlet(np.repeat(beta, num_clients))
        cuts = (np.cumsum(props) * len(idx_c)).astype(int)[:-1]
        for cid, part in enumerate(np.split(idx_c, cuts)):
            idx_per_client[cid].extend(part.tolist())

    out = []
    for idx in idx_per_client:
        rng.shuffle(idx)
        out.append(train.select(idx))
    return out


def split_clients(cfg, dataset) -> List:
    train, _ = get_splits(cfg, dataset)
    if cfg.split == "dirichlet":
        parts = split_dirichlet(train, cfg.num_clients, cfg.dirichlet_beta, cfg.seed)
    else:
        parts = split_iid(train, cfg.num_clients, cfg.seed)

    get_logger().info(
        "Split into %d clients (%s): sizes %s",
        cfg.num_clients,
        cfg.split,
        [len(p) for p in parts],
    )
    return parts


def make_client_loader(cfg, client_dataset, tokenizer) -> DataLoader:
    return DataLoader(
        client_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=DataCollatorWithPadding(tokenizer),
        num_workers=0,
    )
