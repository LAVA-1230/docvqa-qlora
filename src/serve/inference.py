"""Model engine used by the API. Loaded ONCE at startup; requests are serialized with a lock
so concurrent calls cannot run two generate() calls at once and OOM the GPU."""
from __future__ import annotations

import logging
import threading
import types

import torch
import torch.nn.functional as F
from PIL import Image

from src.evaluate import generate_answers
from src.prompting import load_processor

logger = logging.getLogger("docvqa")


def upcast_linear_matmul(model: torch.nn.Module) -> int:
    """CPU trick: keep weights in bf16 (half the RAM) but run every Linear matmul in fp32.

    Many CPUs have no fast bf16/fp16 matmul kernels (10-20x slower than fp32), while
    upcasting one weight matrix at a time is cheap. Returns the number of patched layers.
    """
    def forward(self, x):
        bias = None if self.bias is None else self.bias.float()
        return F.linear(x.float(), self.weight.float(), bias).to(x.dtype)

    n = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            module.forward = types.MethodType(forward, module)
            n += 1
    return n


class DocVQAEngine:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model = None
        self.processor = None
        self.device = "unloaded"
        self.lock = threading.Lock()
        self.max_pixels = cfg["max_pixels"]

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        from src.model.load_model import load_adapter, load_base_model
        from src.utils import get_device_and_dtype

        cfg = self.cfg
        device, dtype = get_device_and_dtype(cfg.get("device", "auto"))
        min_px, max_px = cfg["min_pixels"], cfg["max_pixels"]
        if device == "cpu":
            cpu = cfg["cpu"]
            dtype = getattr(torch, cpu.get("dtype", "bfloat16"))
            min_px, max_px = cpu["min_pixels"], cpu["max_pixels"]
            if cpu.get("num_threads"):
                torch.set_num_threads(int(cpu["num_threads"]))
            logger.warning("No GPU: CPU mode, dtype=%s, max_pixels=%d (answers are slower and may be less "
                           "accurate than at the GPU resolution)", dtype, max_px)
        self.max_pixels = max_px

        mode = cfg["mode"]
        if mode == "merged":
            path = cfg["merged_model_path"]
            self.processor = load_processor(path, min_px, max_px)
            self.model = load_base_model(path, cfg, load_in_4bit=False, device=device, dtype=dtype)
        elif mode in ("adapter", "base"):
            self.processor = load_processor(cfg["model_name"], min_px, max_px)
            four_bit = mode == "adapter" and device == "cuda" and cfg.get("adapter_load_in_4bit", True)
            model = load_base_model(cfg["model_name"], cfg, load_in_4bit=four_bit, device=device, dtype=dtype)
            self.model = load_adapter(model, cfg["adapter_path"]) if mode == "adapter" else model
        else:
            raise ValueError(f"Unknown serve mode {mode!r} (expected merged | adapter | base | mock)")

        if device == "cpu" and cfg["cpu"].get("upcast_matmul", True):
            logger.info("Patched %d Linear layers to compute in fp32", upcast_linear_matmul(self.model))
        self.model.eval()
        self.device = device
        logger.info("Engine ready: mode=%s device=%s dtype=%s", mode, device, dtype)

    def predict(self, images: list[Image.Image], questions: list[str]) -> list[str]:
        if not self.loaded:
            raise RuntimeError("model is not loaded")
        with self.lock:
            return generate_answers(self.model, self.processor, images, questions,
                                    self.cfg["system_prompt"], self.cfg["generation"]["max_new_tokens"])


class MockEngine:
    """Deterministic fake model for tests and for checking the Docker/API plumbing without weights."""

    def __init__(self, cfg: dict | None = None, answer: str = "mock answer"):
        self.cfg = cfg or {}
        self.answer = answer
        self.device = "mock"
        self.max_pixels = (cfg or {}).get("max_pixels", 1003520)
        self.loaded = False
        self.calls: list[tuple[tuple[int, int], str]] = []

    def load(self) -> None:
        self.loaded = True

    def predict(self, images: list[Image.Image], questions: list[str]) -> list[str]:
        self.calls.extend((im.size, q) for im, q in zip(images, questions))
        return [self.answer for _ in questions]
