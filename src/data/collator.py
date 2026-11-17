"""Dataset + training data collator with assistant-only label masking.

For each example we build the FULL chat
    <|im_start|>system ... <|im_end|>
    <|im_start|>user <|vision_start|><|image_pad|>*N<|vision_end|> question <|im_end|>
    <|im_start|>assistant\n ANSWER <|im_end|>\n
and tokenize it with the processor (which also produces pixel_values + image_grid_thw).

labels = input_ids, but EVERYTHING is set to -100 (ignored by the loss) except the
answer tokens and the closing <|im_end|>. So the model is only trained to produce the
answer (and to stop), not to reproduce the prompt or the hundreds of image tokens.
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from src.prompting import build_messages
from src.utils import read_jsonl

logger = logging.getLogger("docvqa")
IGNORE_INDEX = -100


def load_image(path: str | Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


class DocVQADataset(Dataset):
    """Reads {split}.jsonl from data_dir; images are loaded lazily in the collator."""

    def __init__(self, data_dir: str | Path, split: str, limit: int | None = None):
        self.data_dir = Path(data_dir)
        path = self.data_dir / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found - run src.data.prepare_dataset first (or fix --data_dir)")
        self.rows = read_jsonl(path, limit=limit)
        for row in self.rows:
            row["image_path"] = str(self.data_dir / row["image"])
        logger.info("Loaded %d rows from %s", len(self.rows), path)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        return self.rows[idx]


def _find_last_subsequence(seq: list[int], sub: list[int]) -> int:
    for start in range(len(seq) - len(sub), -1, -1):
        if seq[start:start + len(sub)] == sub:
            return start
    return -1


class DocVQACollator:
    def __init__(self, processor, system_prompt: str):
        self.processor = processor
        self.system_prompt = system_prompt
        tok = processor.tokenizer
        self.assistant_prefix = tok.encode("<|im_start|>assistant\n", add_special_tokens=False)
        self.im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
        assert self.im_end_id is not None and self.im_end_id != tok.unk_token_id, "<|im_end|> token not found"

    def build_text(self, question: str, answer: str) -> str:
        messages = build_messages(question, self.system_prompt, answer=answer)
        return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    def __call__(self, examples: list[dict]) -> dict:
        images = [load_image(ex["image_path"]) for ex in examples]
        texts = [self.build_text(ex["question"], ex["answers"][0]) for ex in examples]
        batch = self.processor(text=texts, images=images, padding=True, return_tensors="pt")

        input_ids = batch["input_ids"]
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        for i, ids in enumerate(input_ids.tolist()):
            start = _find_last_subsequence(ids, self.assistant_prefix)
            assert start >= 0, "assistant prefix '<|im_start|>assistant\\n' not found in the tokenized chat"
            answer_start = start + len(self.assistant_prefix)
            try:
                answer_end = ids.index(self.im_end_id, answer_start)  # first <|im_end|> after the answer
            except ValueError as e:
                raise AssertionError("closing <|im_end|> of the assistant turn not found") from e
            assert answer_end > answer_start, f"empty answer for example {examples[i].get('id')}"
            # keep answer tokens + the <|im_end|> so the model learns to stop
            labels[i, answer_start:answer_end + 1] = input_ids[i, answer_start:answer_end + 1]

        labels[batch["attention_mask"] == 0] = IGNORE_INDEX  # padding never contributes to the loss
        batch["labels"] = labels
        return batch
