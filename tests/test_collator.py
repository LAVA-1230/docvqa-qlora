"""Label-masking tests with the REAL Qwen2.5-VL processor (only the processor is loaded, no weights)."""
import torch

from src.data.collator import IGNORE_INDEX, DocVQACollator
from src.prompting import build_messages
from tests.conftest import SYSTEM_PROMPT, TINY_MAX_PIXELS, make_document_image


def _example(tmp_path, name, question, answer, size=(400, 300)):
    path = tmp_path / f"{name}.png"
    make_document_image(size=size).save(path)
    return {"id": name, "image_path": str(path), "question": question, "answers": [answer, "other gold"]}


def _unmasked(batch, i):
    keep = batch["labels"][i] != IGNORE_INDEX
    return batch["input_ids"][i][keep]


def test_only_answer_and_im_end_are_unmasked(processor, tmp_path):
    collator = DocVQACollator(processor, SYSTEM_PROMPT)
    answer = "$1,250.00"
    batch = collator([_example(tmp_path, "a", "What is the total?", answer)])

    kept = _unmasked(batch, 0)
    tok = processor.tokenizer
    assert tok.decode(kept, skip_special_tokens=True) == answer          # first gold answer only
    assert kept[-1].item() == tok.convert_tokens_to_ids("<|im_end|>")   # model learns to stop
    assert len(kept) == len(tok.encode(answer, add_special_tokens=False)) + 1
    for key in ("pixel_values", "image_grid_thw", "attention_mask", "labels"):
        assert key in batch


def test_image_and_prompt_tokens_are_masked(processor, tmp_path):
    collator = DocVQACollator(processor, SYSTEM_PROMPT)
    batch = collator([_example(tmp_path, "a", "What is the date?", "1/8/93")])
    image_pad = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    is_image = batch["input_ids"][0] == image_pad
    n_image_tokens = int(batch["image_grid_thw"][0].prod().item()) // 4  # 2x2 patch merge
    assert is_image.sum().item() == n_image_tokens > 0
    assert (batch["labels"][0][is_image] == IGNORE_INDEX).all()
    # the question text must not be in the labels
    kept_text = processor.tokenizer.decode(_unmasked(batch, 0), skip_special_tokens=True)
    assert "date" not in kept_text


def test_batch_with_padding(processor, tmp_path):
    collator = DocVQACollator(processor, SYSTEM_PROMPT)
    exs = [_example(tmp_path, "short", "Total?", "42", size=(200, 200)),
           _example(tmp_path, "long", "What is the full name written in the signature field at the bottom?",
                    "Dr. John A. Smith Jr.", size=(500, 300))]
    batch = collator(exs)
    assert batch["input_ids"].shape[0] == 2
    assert (batch["labels"][batch["attention_mask"] == 0] == IGNORE_INDEX).all()
    assert (batch["attention_mask"] == 0).any(), "expected padding in a batch of unequal lengths"
    for i, ex in enumerate(exs):
        assert processor.tokenizer.decode(_unmasked(batch, i), skip_special_tokens=True) == ex["answers"][0]
    # left padding: pad tokens are at the start of the shorter row
    short_row = int(batch["attention_mask"].sum(1).argmin())
    assert batch["attention_mask"][short_row, 0] == 0


def test_max_pixels_is_respected(processor, tmp_path):
    collator = DocVQACollator(processor, SYSTEM_PROMPT)
    batch = collator([_example(tmp_path, "big", "Total?", "42", size=(2000, 2600))])
    n_tokens = int(batch["image_grid_thw"][0].prod().item()) // 4
    assert n_tokens <= TINY_MAX_PIXELS // (28 * 28)


def test_generation_prompt_matches_training_prompt(processor):
    gen = processor.apply_chat_template(build_messages("Total?", SYSTEM_PROMPT), tokenize=False,
                                        add_generation_prompt=True)
    train = processor.apply_chat_template(build_messages("Total?", SYSTEM_PROMPT, answer="42"), tokenize=False,
                                          add_generation_prompt=False)
    assert gen.endswith("<|im_start|>assistant\n")
    assert train.startswith(gen), "inference prompt must be an exact prefix of the training text"
    assert SYSTEM_PROMPT in gen and "<|vision_start|><|image_pad|><|vision_end|>" in gen
    assert torch.__version__  # keep torch import used
