"""Seeding, device selection, logging and results persistence."""

import gc
import json
import logging
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

LOGGER_NAME = "fedlora"


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(spec: str = "auto") -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def free_memory(*objs) -> None:
    """Drop references and empty the CUDA cache.

    Important in sweeps: Opacus attaches hooks and per-sample gradient buffers
    to the model, and leftover references will OOM the next run.
    """
    for o in objs:
        del o
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def cuda_mem_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / 1e9


def append_result(row: dict, out_dir: str, filename: str = "results.csv") -> Path:
    """Append one flat result row to a CSV, creating it if needed.

    Appending (rather than rewriting) means a job killed mid-sweep keeps
    everything it already finished.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / filename

    df = pd.DataFrame([row])
    if path.exists():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)
    return path


def load_results(out_dir: str, filename: str = "results.csv") -> pd.DataFrame:
    path = Path(out_dir) / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def already_done(df: pd.DataFrame, keys: dict) -> bool:
    """True if a row matching every key/value pair already exists.

    Lets a sweep be re-launched after a timeout and pick up where it stopped.
    """
    if df.empty:
        return False
    mask = pd.Series(True, index=df.index)
    for k, v in keys.items():
        if k not in df.columns:
            return False
        if v is None:
            mask &= df[k].isna()
        else:
            mask &= df[k] == v
    return bool(mask.any())


def save_json(obj, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
