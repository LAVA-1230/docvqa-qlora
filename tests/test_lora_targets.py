"""The LoRA regex must hit all LLM linear layers of the REAL 3B architecture and nothing in the
vision tower. The model is built on the 'meta' device (no weights, no RAM)."""
import re

from src.model.load_model import lora_target_regex, matched_module_names

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _meta_model(config):
    from accelerate import init_empty_weights
    from transformers import Qwen2_5_VLForConditionalGeneration

    with init_empty_weights():
        return Qwen2_5_VLForConditionalGeneration._from_config(config)


def test_regex_targets_llm_layers_only(qwen_config):
    model = _meta_model(qwen_config)
    names = matched_module_names(model, lora_target_regex(TARGETS))
    assert len(names) == qwen_config.num_hidden_layers * len(TARGETS)  # 36 * 7 = 252
    assert not any("visual" in n for n in names)


def test_naive_target_list_would_hit_vision_tower(qwen_config):
    """Documents WHY the regex is needed: the ViT MLP also uses gate/up/down_proj names."""
    model = _meta_model(qwen_config)
    naive = [n for n, _ in model.named_modules() if n.split(".")[-1] in TARGETS]
    assert any(n.startswith("visual.") for n in naive)


def test_regex_on_plain_names():
    rx = re.compile(lora_target_regex(TARGETS))
    assert rx.fullmatch("model.layers.0.self_attn.q_proj")
    assert rx.fullmatch("base_model.model.model.layers.35.mlp.down_proj")
    assert not rx.fullmatch("visual.blocks.0.mlp.gate_proj")
    assert not rx.fullmatch("model.visual.blocks.3.mlp.down_proj")
    assert not rx.fullmatch("visual.blocks.0.attn.qkv")
    assert not rx.fullmatch("lm_head")
