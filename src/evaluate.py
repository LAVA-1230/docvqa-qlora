"""Phase 3 / 5 - evaluate a model on a split with ANLS, exact match and latency.

The SAME prompt (common.yaml), pixel limits and greedy decoding are used for every
model, so baseline vs fine-tuned numbers are directly comparable.

Usage
    # baseline (base model, fp16/bf16)
    python -m src.evaluate --config configs/eval.yaml --split test --output results/baseline.json
    # fine-tuned (4-bit base + best LoRA adapter, exactly as trained)
    python -m src.evaluate --config configs/eval.yaml --split test --adapter outputs/qlora/best_adapter \
        --load_in_4bit --output results/finetuned.json
    # merged model
    python -m src.evaluate --config configs/eval.yaml --split test --model_path models/merged --output ...
    # smoke test
    python -m src.evaluate --config configs/eval.yaml --split test --limit 20 --output results/smoke.json
"""
from __future__ import annotations

import argparse
import logging
import platform
import time
from contextlib import nullcontext
from datetime import datetime, timezone

import torch

from src.data.collator import load_image
from src.metrics import anls, exact_match
from src.prompting import build_messages, load_processor, postprocess_answer
from src.utils import load_config, parse_set_args, read_jsonl, save_json, set_seed, setup_logging

logger = logging.getLogger("docvqa")


@torch.inference_mode()
def generate_answers(model, processor, images, questions, system_prompt: str, max_new_tokens: int,
                     autocast_dtype: torch.dtype | None = None) -> list[str]:
    """Batched greedy generation. The processor must use LEFT padding."""
    assert processor.tokenizer.padding_side == "left", "batched generation needs left padding"
    texts = [
        processor.apply_chat_template(build_messages(q, system_prompt), tokenize=False, add_generation_prompt=True)
        for q in questions
    ]
    inputs = processor(text=texts, images=images, padding=True, return_tensors="pt").to(model.device)
    ctx = torch.autocast("cuda", dtype=autocast_dtype) if autocast_dtype is not None else nullcontext()
    with ctx:
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # greedy -> deterministic
            num_beams=1,
            temperature=None,         # unset sampling params from generation_config (avoids warnings)
            top_p=None,
            top_k=None,
            use_cache=True,           # training sets config.use_cache=False; generation needs the KV cache
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    new_tokens = out[:, inputs["input_ids"].shape[1]:]
    decoded = processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return [postprocess_answer(t) for t in decoded]


def run_eval(model, processor, rows: list[dict], cfg: dict, batch_size: int,
             autocast_dtype: torch.dtype | None = None, log_every: int = 10) -> dict:
    """Generate predictions for `rows` (dicts with image_path/question/answers) and score them."""
    preds: list[dict] = []
    is_cuda = torch.cuda.is_available() and next(model.parameters()).is_cuda
    if is_cuda:
        torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    n_batches = (len(rows) + batch_size - 1) // batch_size
    for b in range(n_batches):
        batch = rows[b * batch_size:(b + 1) * batch_size]
        images = [load_image(r["image_path"]) for r in batch]
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        answers = generate_answers(model, processor, images, [r["question"] for r in batch],
                                   cfg["system_prompt"], cfg["generation"]["max_new_tokens"], autocast_dtype)
        if is_cuda:
            torch.cuda.synchronize()
        per_sample_ms = (time.perf_counter() - t0) * 1000 / len(batch)
        for r, pred in zip(batch, answers):
            preds.append({
                "id": r["id"],
                "question": r["question"],
                "answers": r["answers"],
                "pred": pred,
                "anls": anls(pred, r["answers"]),
                "exact_match": exact_match(pred, r["answers"]),
                "question_types": r.get("question_types", []),
                "latency_ms": round(per_sample_ms, 1),
            })
        if (b + 1) % log_every == 0 or b + 1 == n_batches:
            running = sum(p["anls"] for p in preds) / len(preds)
            logger.info("  batch %d/%d | running ANLS %.4f | last: %r (gold %r)",
                        b + 1, n_batches, running, preds[-1]["pred"], preds[-1]["answers"][0])

    n = len(preds)
    summary = {
        "n": n,
        "anls": round(sum(p["anls"] for p in preds) / n, 4),
        "exact_match": round(sum(p["exact_match"] for p in preds) / n, 4),
        "avg_latency_ms": round(sum(p["latency_ms"] for p in preds) / n, 1),
        "total_seconds": round(time.time() - t_start, 1),
        "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if is_cuda else None,
    }
    empty = sum(1 for p in preds if not p["pred"].strip("! "))
    if empty > 0.5 * n:
        logger.error("More than half of the predictions are empty or '!!!'. This usually means fp16 "
                     "overflow/NaN. Check vision_fp32_when_fp16 and try --load_in_4bit.")
    return {"summary": summary, "predictions": preds}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--model_path", default=None, help="merged model dir (default: base model_name)")
    parser.add_argument("--adapter", default=None, help="LoRA adapter dir")
    parser.add_argument("--load_in_4bit", action="store_true", help="load the base model in 4-bit NF4")
    parser.add_argument("--output", required=True, help="where to write the results JSON")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--set", nargs="*", default=[], help="config overrides key=value")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config, parse_set_args(args.set))
    set_seed(cfg["seed"])
    data_dir = args.data_dir or cfg["data_dir"]
    batch_size = args.batch_size or cfg["batch_size"]
    load_in_4bit = args.load_in_4bit or cfg.get("load_in_4bit", False)

    from src.data.collator import DocVQADataset
    from src.model.load_model import load_model_for_inference

    rows = DocVQADataset(data_dir, args.split, limit=args.limit).rows
    processor = load_processor(args.model_path or cfg["model_name"], cfg["min_pixels"], cfg["max_pixels"])
    model, device, dtype = load_model_for_inference(cfg, model_path=args.model_path, adapter_path=args.adapter,
                                                    load_in_4bit=load_in_4bit)
    if device == "cpu":
        logger.warning("Running evaluation on CPU - this is very slow; use --limit for a smoke test only.")

    logger.info("Evaluating %d samples of split '%s' (batch %d)", len(rows), args.split, batch_size)
    result = run_eval(model, processor, rows, cfg, batch_size)
    result["meta"] = {
        "split": args.split,
        "model": args.model_path or cfg["model_name"],
        "adapter": args.adapter,
        "load_in_4bit": load_in_4bit,
        "dtype": str(dtype),
        "device": torch.cuda.get_device_name(0) if device == "cuda" else platform.processor() or "cpu",
        "min_pixels": cfg["min_pixels"],
        "max_pixels": cfg["max_pixels"],
        "system_prompt": cfg["system_prompt"],
        "max_new_tokens": cfg["generation"]["max_new_tokens"],
        "batch_size": batch_size,
        "limit": args.limit,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    save_json(args.output, result)
    s = result["summary"]
    logger.info("=" * 60)
    logger.info("%s | split=%s n=%d | ANLS=%.4f | EM=%.4f | latency=%.0f ms/sample",
                "adapter" if args.adapter else "model", args.split, s["n"], s["anls"], s["exact_match"],
                s["avg_latency_ms"])
    logger.info("Saved -> %s", args.output)


if __name__ == "__main__":
    main()
