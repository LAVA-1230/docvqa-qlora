"""ONE place that defines the chat prompt and the processor's pixel limits.

Training (collator), evaluation (evaluate.py) and serving (serve/inference.py) all call
these functions, which guarantees the model sees exactly the same prompt format and
image resolution everywhere.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("docvqa")


def build_messages(question: str, system_prompt: str, answer: str | None = None) -> list[dict]:
    """Chat in Qwen2.5-VL format: system, user [image + question], optional assistant answer.

    The {"type": "image"} entry is a placeholder; the chat template turns it into
    <|vision_start|><|image_pad|><|vision_end|> and the processor expands <|image_pad|>
    into one token per 28x28 image patch.
    """
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question.strip()},
            ],
        },
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer.strip()}]})
    return messages


def load_processor(model_name_or_path: str, min_pixels: int, max_pixels: int):
    """AutoProcessor with explicit pixel limits and LEFT padding.

    Left padding is required for batched generation (new tokens are appended on the
    right, so all prompts must end at the same position). It is also fine for training
    because the attention mask and the -100 labels ignore the pad tokens.
    """
    from transformers import AutoProcessor

    if min_pixels > max_pixels:
        raise ValueError(f"min_pixels ({min_pixels}) > max_pixels ({max_pixels})")
    processor = AutoProcessor.from_pretrained(
        model_name_or_path, min_pixels=min_pixels, max_pixels=max_pixels
    )
    # Set every representation explicitly: depending on the transformers version the
    # image processor reads either .min_pixels/.max_pixels or .size[...].
    ip = processor.image_processor
    ip.min_pixels = min_pixels
    ip.max_pixels = max_pixels
    ip.size = {"shortest_edge": min_pixels, "longest_edge": max_pixels}
    processor.tokenizer.padding_side = "left"
    logger.info(
        "Processor loaded from %s | min_pixels=%d (%d tokens) max_pixels=%d (%d tokens) | padding=left",
        model_name_or_path, min_pixels, min_pixels // (28 * 28), max_pixels, max_pixels // (28 * 28),
    )
    return processor


def postprocess_answer(text: str) -> str:
    """Keep the first non-empty line; the prompt asks for the answer only."""
    for line in text.strip().splitlines():
        if line.strip():
            return line.strip()
    return ""
