"""Small shared helpers: config loading, logging, seeding, device/dtype detection, JSON IO.

torch is imported lazily inside functions so that light modules (metrics, data split
logic) and their tests do not need torch at import time.
"""
from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Iterable

import yaml

logger = logging.getLogger("docvqa")


# --------------------------------------------------------------------------- config
def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into a copy of `base`."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | os.PathLike, overrides: dict | None = None) -> dict:
    """Load a YAML config. A top-level `inherit: other.yaml` key loads that file first
    (path relative to this file) and merges this file on top of it."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path.resolve()}")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("inherit", None)
    if parent:
        cfg = deep_merge(load_config(path.parent / parent), cfg)
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg


def parse_set_args(items: Iterable[str] | None) -> dict:
    """Turn ["training.num_train_epochs=1", "batch_size=2"] into a nested dict.
    Values are parsed with YAML, so numbers/bools/lists work."""
    result: dict = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got: {item!r}")
        key, raw = item.split("=", 1)
        node = result
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return result


# --------------------------------------------------------------------------- logging / seeds
def setup_logging(log_file: str | os.PathLike | None = None, level: int = logging.INFO) -> logging.Logger:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# --------------------------------------------------------------------------- device / dtype
def bf16_supported() -> bool:
    """True only for GPUs with NATIVE bf16 (Ampere, compute capability >= 8.0).

    NOTE: torch.cuda.is_bf16_supported() can return True on a T4 because it counts
    *emulated* bf16, which is very slow. We therefore check the compute capability.
    """
    import torch

    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 8


def get_device_and_dtype(device: str = "auto"):
    """Return (device_str, torch_dtype). CUDA -> bf16 if native else fp16. CPU -> fp32."""
    import torch

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device=cuda requested but no CUDA GPU is available.")
        dtype = torch.bfloat16 if bf16_supported() else torch.float16
        logger.info(
            "GPU: %s | capability %s | dtype %s",
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_capability(0),
            dtype,
        )
        return device, dtype
    return "cpu", torch.float32


# --------------------------------------------------------------------------- IO
def read_jsonl(path: str | os.PathLike, limit: int | None = None) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def write_jsonl(path: str | os.PathLike, rows: Iterable[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(path: str | os.PathLike, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def load_json(path: str | os.PathLike) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
