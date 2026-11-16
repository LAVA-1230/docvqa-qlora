"""Phase 6 - merge the LoRA adapter into the base weights for simple, fast serving.

The base is reloaded in fp16/bf16 (NOT 4-bit: you cannot cleanly merge into 4-bit
weights), the adapter is applied, merge_and_unload() folds W + (alpha/r)*B@A into W,
and the result is saved with its processor.

Verification: on N val samples we compare ANLS of (fp16 base + adapter, unmerged)
vs (merged). Merging is mathematically exact up to fp16 rounding, so the scores should
match within the tolerance.

Usage
    python -m src.merge_adapter --config configs/eval.yaml --adapter outputs/qlora/best_adapter \
        --output models/merged --verify_limit 100
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from src.data.collator import DocVQADataset
from src.evaluate import run_eval
from src.model.load_model import get_visual, load_adapter, load_base_model
from src.prompting import load_processor
from src.utils import get_device_and_dtype, load_config, parse_set_args, save_json, set_seed, setup_logging

logger = logging.getLogger("docvqa")


def merge(model):
    merged = model.merge_and_unload()
    leftover = [n for n, _ in merged.named_parameters() if "lora_" in n]
    assert not leftover, f"LoRA params still present after merge: {leftover[:3]}"
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", default="models/merged")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--results_dir", default=None)
    parser.add_argument("--verify_limit", type=int, default=100, help="val samples for the check (0 = skip)")
    parser.add_argument("--limit", type=int, default=None, help="alias of --verify_limit for smoke tests")
    parser.add_argument("--tolerance", type=float, default=0.01, help="max allowed |ANLS diff| (0-1 scale)")
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config, parse_set_args(args.set))
    set_seed(cfg["seed"])
    verify_n = args.limit if args.limit is not None else args.verify_limit
    results_dir = Path(args.results_dir or cfg["results_dir"])

    device, dtype = get_device_and_dtype()
    processor = load_processor(cfg["model_name"], cfg["min_pixels"], cfg["max_pixels"])
    base = load_base_model(cfg["model_name"], cfg, load_in_4bit=False, device=device, dtype=dtype)
    model = load_adapter(base, args.adapter)
    model.eval()

    check: dict = {"adapter": args.adapter, "dtype": str(dtype), "verify_n": verify_n}
    rows = []
    if verify_n:
        rows = DocVQADataset(args.data_dir or cfg["data_dir"], "val", limit=verify_n).rows
        logger.info("Scoring UNMERGED (base %s + adapter) on %d val samples", dtype, len(rows))
        unmerged = run_eval(model, processor, rows, cfg, cfg.get("batch_size", 4))
        check["unmerged_anls"] = unmerged["summary"]["anls"]

    merged = merge(model)
    merged.eval()

    if verify_n:
        logger.info("Scoring MERGED model on the same samples")
        merged_res = run_eval(merged, processor, rows, cfg, cfg.get("batch_size", 4))
        check["merged_anls"] = merged_res["summary"]["anls"]
        check["abs_diff"] = round(abs(check["merged_anls"] - check["unmerged_anls"]), 4)
        check["different_predictions"] = sum(
            a["pred"] != b["pred"] for a, b in zip(unmerged["predictions"], merged_res["predictions"]))
        check["passed"] = check["abs_diff"] <= args.tolerance
        logger.info("unmerged %.4f | merged %.4f | diff %.4f | %d/%d predictions differ -> %s",
                    check["unmerged_anls"], check["merged_anls"], check["abs_diff"],
                    check["different_predictions"], len(rows), "PASS" if check["passed"] else "FAIL")

    # the vision tower may have been upcast to fp32 for fp16 inference; save everything in one dtype
    get_visual(merged).to(dtype)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out, safe_serialization=True, max_shard_size="2GB")
    processor.save_pretrained(out)
    check["output"] = str(out)
    save_json(results_dir / "merge_check.json", check)
    logger.info("Merged model + processor saved to %s", out)
    if verify_n and not check["passed"]:
        raise SystemExit(f"Merge verification FAILED: |diff| {check['abs_diff']} > tolerance {args.tolerance}")


if __name__ == "__main__":
    main()
