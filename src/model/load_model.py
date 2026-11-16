"""Model loading: base model (fp16/bf16 or 4-bit NF4), LoRA attachment, adapters.

Module layout of Qwen2_5_VLForConditionalGeneration (transformers 4.51):
    visual.*        vision tower (ViT, ~670M params) + merger      -> frozen, NOT quantized
    model.layers.*  Qwen2.5 LLM decoder (36 layers, ~3B params)     -> 4-bit + LoRA
    lm_head         output projection                               -> NOT quantized
"""
from __future__ import annotations

import logging
import re

import torch

logger = logging.getLogger("docvqa")


# --------------------------------------------------------------------------- helpers
def get_visual(model):
    """Return the vision tower for plain, PEFT-wrapped and newer-transformers layouts."""
    m = model
    for _ in range(4):
        if hasattr(m, "visual"):
            return m.visual
        if hasattr(m, "model") and hasattr(m.model, "visual"):
            return m.model.visual
        m = getattr(m, "base_model", None) or getattr(m, "model", None)
        if m is None:
            break
    raise AttributeError("Could not find the vision tower ('visual') on this model")


def build_bnb_config(qcfg: dict, compute_dtype: torch.dtype):
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=qcfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=qcfg.get("bnb_4bit_use_double_quant", True),
        bnb_4bit_compute_dtype=compute_dtype,
        # In transformers this list is also honoured for 4-bit: these modules stay in
        # compute dtype. "visual" keeps the whole vision tower un-quantized.
        llm_int8_skip_modules=list(qcfg.get("skip_modules", ["visual", "lm_head"])),
    )


def load_base_model(
    model_name_or_path: str,
    cfg: dict,
    load_in_4bit: bool = False,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
):
    """Load Qwen2.5-VL. On CUDA everything goes on GPU 0 (no silent CPU offload)."""
    from transformers import Qwen2_5_VLForConditionalGeneration

    kwargs: dict = {"torch_dtype": dtype, "attn_implementation": cfg.get("attn_implementation", "sdpa")}
    if load_in_4bit:
        if device != "cuda":
            raise RuntimeError("4-bit (bitsandbytes) loading needs a CUDA GPU. Use load_in_4bit=false on CPU.")
        kwargs["quantization_config"] = build_bnb_config(cfg.get("quantization", {}), dtype)
    if device == "cuda":
        # {"": 0} instead of "auto": "auto" may silently offload layers to CPU when VRAM
        # is tight, which makes everything 10x slower and breaks 4-bit training.
        kwargs["device_map"] = {"": 0}

    logger.info("Loading %s | 4bit=%s | dtype=%s | device=%s", model_name_or_path, load_in_4bit, dtype, device)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name_or_path, **kwargs)

    if device == "cuda" and dtype == torch.float16 and cfg.get("vision_fp32_when_fp16", True):
        # fp16 has a max value of 65504; activations inside the ViT can overflow on T4 and
        # produce NaN / garbage ("!!!!") answers. The vision tower is small, so run it in
        # fp32. The model casts image embeddings back to the LLM dtype before merging.
        get_visual(model).float()
        logger.info("Vision tower cast to fp32 (fp16-only GPU)")

    if device == "cuda":
        logger.info("GPU memory after load: %.2f GB", torch.cuda.memory_allocated() / 1e9)
    return model


# --------------------------------------------------------------------------- LoRA
def lora_target_regex(target_modules: list[str]) -> str:
    """Regex matching LLM linear layers only.

    peft uses re.fullmatch(regex, module_name). The negative lookahead (?!.*visual)
    rejects every module whose name contains "visual". This matters because the
    Qwen2.5-VL vision MLP ALSO has gate_proj/up_proj/down_proj layers; a plain list
    ["gate_proj", ...] would silently add LoRA to the vision tower.
    """
    names = "|".join(re.escape(m) for m in target_modules)
    return rf"^(?!.*visual).*\.({names})$"


def matched_module_names(model, regex: str) -> list[str]:
    pattern = re.compile(regex)
    return [name for name, _ in model.named_modules() if pattern.fullmatch(name)]


def trainable_parameter_report(model) -> dict:
    if hasattr(model, "get_nb_trainable_parameters"):  # peft: counts 4-bit params correctly
        trainable, total = model.get_nb_trainable_parameters()
    else:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
    return {"trainable_params": int(trainable), "total_params": int(total),
            "trainable_percent": round(100 * trainable / total, 4)}


def attach_lora(model, lora_cfg: dict, quantized: bool, gradient_checkpointing: bool = True):
    """prepare (k-bit) -> add LoRA to LLM layers only -> verify nothing in the vision tower trains."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    # 1. freeze everything (the vision encoder stays frozen for the whole run)
    for p in model.parameters():
        p.requires_grad = False

    if quantized:
        # upcasts non-quantized params (norms, vision tower, embeddings) to fp32 for
        # stability, enables input grads and gradient checkpointing.
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    elif gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()

    regex = lora_target_regex(lora_cfg["target_modules"])
    targets = matched_module_names(model, regex)
    assert targets, f"LoRA regex {regex} matched no modules"
    assert not any("visual" in t for t in targets), "LoRA regex matched vision-tower modules!"
    logger.info("LoRA will wrap %d linear layers (e.g. %s)", len(targets), targets[:2])

    config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg.get("bias", "none"),
        target_modules=regex,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.config.use_cache = False  # incompatible with gradient checkpointing during training

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable, "no trainable parameters after get_peft_model"
    assert all("lora_" in n for n in trainable), f"non-LoRA params are trainable: {[n for n in trainable if 'lora_' not in n][:5]}"
    assert not any("visual" in n for n in trainable), "vision-tower params are trainable!"
    bad_dtype = [n for n, p in model.named_parameters() if p.requires_grad and p.dtype != torch.float32]
    assert not bad_dtype, f"trainable params must be fp32 for fp16 mixed precision, got: {bad_dtype[:3]}"

    model.print_trainable_parameters()
    return model


def load_adapter(model, adapter_path: str, is_trainable: bool = False):
    from peft import PeftModel

    logger.info("Loading LoRA adapter from %s", adapter_path)
    return PeftModel.from_pretrained(model, adapter_path, is_trainable=is_trainable)


def load_model_for_inference(
    cfg: dict,
    model_path: str | None = None,
    adapter_path: str | None = None,
    load_in_4bit: bool = False,
    device: str = "auto",
    dtype: torch.dtype | None = None,
):
    """Base model (or merged model at model_path) + optional adapter, in eval mode."""
    from src.utils import get_device_and_dtype

    device, auto_dtype = get_device_and_dtype(device)
    dtype = dtype or auto_dtype
    model = load_base_model(model_path or cfg["model_name"], cfg, load_in_4bit=load_in_4bit, device=device, dtype=dtype)
    if adapter_path:
        model = load_adapter(model, adapter_path)
    model.eval()
    return model, device, dtype
