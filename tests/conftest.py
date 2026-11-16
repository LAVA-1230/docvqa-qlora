"""Shared fixtures. Tests that need the Qwen processor/config download a few MB from the
Hugging Face Hub once (cached afterwards) and are skipped automatically when offline."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
SYSTEM_PROMPT = "Answer the question using only text from the document. Reply with the answer only."
TINY_MIN_PIXELS = 4 * 28 * 28
TINY_MAX_PIXELS = 16 * 28 * 28


def make_document_image(text: str = "TOTAL: $1,250.00", size=(400, 300)) -> Image.Image:
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, size[0] - 10, size[1] - 10], outline="black")
    draw.text((30, 40), "INVOICE", fill="black")
    draw.text((30, 90), text, fill="black")
    return img


@pytest.fixture(scope="session")
def processor():
    from src.prompting import load_processor

    try:
        return load_processor(MODEL_NAME, TINY_MIN_PIXELS, TINY_MAX_PIXELS)
    except Exception as e:  # offline / hub unavailable
        pytest.skip(f"cannot load processor from the Hub: {e}")


@pytest.fixture(scope="session")
def qwen_config():
    from transformers import AutoConfig

    try:
        return AutoConfig.from_pretrained(MODEL_NAME)
    except Exception as e:
        pytest.skip(f"cannot load config from the Hub: {e}")


@pytest.fixture
def tiny_data_dir(tmp_path):
    """A 4-question / 4-document dataset in the same format prepare_dataset.py writes."""
    import json

    (tmp_path / "images").mkdir()
    answers = ["$1,250.00", "1/8/93", "John Smith", "42"]
    for split in ("train", "val", "test"):
        with open(tmp_path / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for i, ans in enumerate(answers):
                doc = f"{split}{i}"
                make_document_image(f"VALUE: {ans}").save(tmp_path / "images" / f"{doc}.png")
                f.write(json.dumps({
                    "id": hash(doc) % 10**6, "question": f"What is value {i}?", "answers": [ans],
                    "question_types": ["form"], "doc_id": doc, "ucsf_document_id": doc, "page_no": "1",
                    "image": f"images/{doc}.png", "width": 400, "height": 300,
                }) + "\n")
    return tmp_path
