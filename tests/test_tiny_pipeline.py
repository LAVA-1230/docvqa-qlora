"""End-to-end CPU test of the GPU code paths on a TINY random Qwen2.5-VL (2 layers, same
architecture + real processor): LoRA attach, Trainer epoch with the val-ANLS callback,
checkpoint/resume discovery, batched generation, and merge_and_unload equivalence.
Only 4-bit/bitsandbytes and fp16 cannot be exercised on CPU."""
import copy

import pytest
import torch
from transformers.trainer_utils import get_last_checkpoint

from src.data.collator import DocVQACollator, DocVQADataset
from src.evaluate import generate_answers, run_eval
from src.merge_adapter import merge
from src.model.load_model import attach_lora, load_adapter, trainable_parameter_report
from src.train import build_trainer
from tests.conftest import SYSTEM_PROMPT, TINY_MAX_PIXELS, TINY_MIN_PIXELS, make_document_image

CFG = {
    "seed": 0,
    "system_prompt": SYSTEM_PROMPT,
    "min_pixels": TINY_MIN_PIXELS,
    "max_pixels": TINY_MAX_PIXELS,
    "generation": {"max_new_tokens": 4},
    "lora": {"r": 4, "lora_alpha": 8, "lora_dropout": 0.0, "bias": "none",
             "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]},
    "training": {"num_train_epochs": 1, "per_device_train_batch_size": 1, "gradient_accumulation_steps": 2,
                 "learning_rate": 0.001, "lr_scheduler_type": "cosine", "warmup_ratio": 0.0,
                 "optim": "adamw_torch", "logging_steps": 1, "save_strategy": "no", "eval_strategy": "epoch",
                 "per_device_eval_batch_size": 1, "dataloader_num_workers": 0,
                 "gradient_checkpointing": True, "report_to": "none"},
    "val_anls": {"enabled": True, "limit": 2, "batch_size": 2},
}


def make_tiny_config(config):
    cfg = copy.deepcopy(config)
    cfg.hidden_size, cfg.intermediate_size = 64, 128
    cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads = 2, 4, 2
    cfg.max_window_layers = 2
    rope = dict(cfg.rope_scaling)
    rope["mrope_section"] = [2, 3, 3]  # must sum to head_dim/2 = (64/4)/2 = 8
    cfg.rope_scaling = rope
    v = cfg.vision_config
    v.depth, v.hidden_size, v.intermediate_size, v.num_heads = 2, 32, 64, 2
    v.out_hidden_size, v.fullatt_block_indexes = 64, [1]
    return cfg


@pytest.fixture(scope="module")
def tiny_base_dir(qwen_config, tmp_path_factory):
    from transformers import Qwen2_5_VLForConditionalGeneration

    torch.manual_seed(0)
    model = Qwen2_5_VLForConditionalGeneration(make_tiny_config(qwen_config))
    path = tmp_path_factory.mktemp("tiny_base")
    model.save_pretrained(path)
    return path


def _load_tiny(path):
    from transformers import Qwen2_5_VLForConditionalGeneration

    return Qwen2_5_VLForConditionalGeneration.from_pretrained(path, torch_dtype=torch.float32,
                                                              attn_implementation="sdpa")


def test_batched_generation_returns_one_string_per_question(tiny_base_dir, processor):
    model = _load_tiny(tiny_base_dir).eval()
    images = [make_document_image(size=(300, 200)), make_document_image(size=(500, 400))]
    out = generate_answers(model, processor, images, ["Total?", "What is the date on the form?"],
                           SYSTEM_PROMPT, max_new_tokens=4)
    assert len(out) == 2 and all(isinstance(a, str) for a in out)


def test_lora_attach_train_checkpoint_and_merge(tiny_base_dir, processor, tiny_data_dir, tmp_path):
    model = attach_lora(_load_tiny(tiny_base_dir), CFG["lora"], quantized=False, gradient_checkpointing=True)
    report = trainable_parameter_report(model)
    assert 0 < report["trainable_percent"] < 100
    lora_names = [n for n, p in model.named_parameters() if p.requires_grad]
    assert len(lora_names) == 2 * 7 * 2  # layers * target modules * (A, B)
    assert not any("visual" in n for n in lora_names)

    train_ds = DocVQADataset(tiny_data_dir, "train", limit=4)
    val_ds = DocVQADataset(tiny_data_dir, "val", limit=2)
    out_dir, res_dir = tmp_path / "out", tmp_path / "results"
    trainer = build_trainer(model, processor, train_ds, val_ds, CFG, out_dir, res_dir, "cpu", torch.float32,
                            val_ds.rows)
    before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    result = trainer.train()

    assert torch.isfinite(torch.tensor(result.training_loss))
    changed = sum(not torch.equal(before[n], p) for n, p in model.named_parameters() if p.requires_grad)
    assert changed > 0, "LoRA weights did not change after training"
    assert any("eval_loss" in h for h in trainer.state.log_history)
    assert (res_dir / "val_anls_per_epoch.json").exists()
    assert (out_dir / "best_adapter" / "adapter_config.json").exists()
    assert get_last_checkpoint(str(out_dir)) is not None, "epoch-end checkpoint missing (resume would fail)"

    # --- merge: base + adapter (unmerged) must equal the merged model
    peft_model = load_adapter(_load_tiny(tiny_base_dir), str(out_dir / "best_adapter")).eval()
    collator = DocVQACollator(processor, SYSTEM_PROMPT)
    batch = collator([val_ds.rows[0]])
    batch.pop("labels")
    with torch.no_grad():
        logits_unmerged = peft_model(**batch).logits
        merged = merge(peft_model).eval()
        logits_merged = merged(**batch).logits
    assert not any("lora_" in n for n, _ in merged.named_parameters())
    torch.testing.assert_close(logits_merged, logits_unmerged, atol=1e-4, rtol=1e-4)

    # --- run_eval produces the result structure evaluate.py writes
    res = run_eval(merged, processor, val_ds.rows, CFG, batch_size=2)
    assert res["summary"]["n"] == 2 and 0.0 <= res["summary"]["anls"] <= 1.0
    assert {"id", "pred", "anls", "exact_match", "latency_ms"} <= set(res["predictions"][0])
