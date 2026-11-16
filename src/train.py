"""Phase 4 - QLoRA fine-tuning of Qwen2.5-VL-3B with transformers.Trainer.

* 4-bit NF4 base (frozen) + LoRA r=16 on the LLM linear layers only.
* custom collator -> loss only on the answer tokens.
* checkpoints every `save_steps` AND at the end of every epoch -> safe on Colab.
* auto-resume: re-running the same command continues from the last checkpoint.
* after every epoch: generation-based val ANLS; the best adapter is copied to
  <output_dir>/best_adapter (this is the one used for the final test evaluation).

Usage
    python -m src.train --config configs/train.yaml
    python -m src.train --config configs/train.yaml --limit 32 --val_limit 8 --set training.num_train_epochs=1   # smoke
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch
from transformers import Trainer, TrainerCallback, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from src.data.collator import DocVQACollator, DocVQADataset
from src.evaluate import run_eval
from src.model.load_model import attach_lora, load_base_model, trainable_parameter_report
from src.prompting import load_processor
from src.utils import (get_device_and_dtype, load_config, load_json, parse_set_args, save_json, set_seed,
                       setup_logging)

logger = logging.getLogger("docvqa")


class ValANLSCallback(TrainerCallback):
    """At the end of every epoch: generate on val, log ANLS, keep the best adapter, force a checkpoint."""

    def __init__(self, processor, val_rows, cfg, output_dir: Path, results_dir: Path, autocast_dtype):
        self.processor = processor
        self.val_rows = val_rows
        self.cfg = cfg
        self.best_dir = output_dir / "best_adapter"
        self.history_path = results_dir / "val_anls_per_epoch.json"
        self.autocast_dtype = autocast_dtype
        self.history = load_json(self.history_path) if self.history_path.exists() else []
        best_file = self.best_dir / "best_score.json"
        self.best = load_json(best_file)["val_anls"] if best_file.exists() else -1.0  # survives resume

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if self.cfg["val_anls"].get("enabled", True) and self.val_rows:
            t0 = time.time()
            model.eval()
            res = run_eval(model, self.processor, self.val_rows, self.cfg,
                           batch_size=self.cfg["val_anls"]["batch_size"], autocast_dtype=self.autocast_dtype)
            model.train()
            score = res["summary"]["anls"]
            epoch = round(state.epoch or 0, 2)
            entry = {"epoch": epoch, "global_step": state.global_step, "val_anls": score,
                     "val_exact_match": res["summary"]["exact_match"], "n": res["summary"]["n"],
                     "seconds": round(time.time() - t0, 1)}
            self.history = [h for h in self.history if h["epoch"] != epoch] + [entry]
            save_json(self.history_path, self.history)
            state.log_history.append({"epoch": epoch, "step": state.global_step, "val_anls": score})
            logger.info("[epoch %.2f] val ANLS = %.4f (best so far %.4f)", epoch, score, max(self.best, 0))
            if score > self.best:
                self.best = score
                model.save_pretrained(self.best_dir)
                save_json(self.best_dir / "best_score.json", entry)
                logger.info("New best adapter saved to %s", self.best_dir)
        control.should_save = True  # always checkpoint at epoch end (in addition to save_steps)
        return control


def build_training_args(cfg: dict, output_dir: Path, device: str, dtype: torch.dtype, has_eval: bool) -> TrainingArguments:
    t = dict(cfg["training"])
    if not has_eval:
        t["eval_strategy"] = "no"
    return TrainingArguments(
        output_dir=str(output_dir),
        fp16=device == "cuda" and dtype == torch.float16,
        bf16=device == "cuda" and dtype == torch.bfloat16,
        remove_unused_columns=False,          # keep our raw dict rows for the collator
        label_names=["labels"],
        # eval only needs the loss. Otherwise Trainer keeps the full logits of every val sample
        # (seq_len x 152k vocab ~ 0.8 GB each) and runs out of GPU memory.
        prediction_loss_only=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=cfg["seed"],
        data_seed=cfg["seed"],
        logging_first_step=True,
        use_cpu=device == "cpu",
        **t,
    )


def build_trainer(model, processor, train_ds, eval_ds, cfg, output_dir: Path, results_dir: Path,
                  device: str, dtype: torch.dtype, val_rows: list[dict]) -> Trainer:
    args = build_training_args(cfg, output_dir, device, dtype, has_eval=eval_ds is not None and len(eval_ds) > 0)
    autocast_dtype = dtype if device == "cuda" else None
    callbacks = [ValANLSCallback(processor, val_rows, cfg, output_dir, results_dir, autocast_dtype)]
    return Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DocVQACollator(processor, cfg["system_prompt"]),
        callbacks=callbacks,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--output_dir", default=None, help="checkpoints dir (Drive on Colab)")
    parser.add_argument("--results_dir", default=None)
    parser.add_argument("--limit", type=int, default=None, help="use only N training samples (smoke test)")
    parser.add_argument("--val_limit", type=int, default=None, help="override val_anls.limit / eval_loss_limit")
    parser.add_argument("--no_resume", action="store_true", help="ignore existing checkpoints")
    parser.add_argument("--allow_cpu", action="store_true", help="allow training on CPU (tiny smoke tests only)")
    parser.add_argument("--set", nargs="*", default=[], help="config overrides key=value, e.g. training.num_train_epochs=1")
    args = parser.parse_args()

    cfg = load_config(args.config, parse_set_args(args.set))
    output_dir = Path(args.output_dir or cfg["output_dir"])
    results_dir = Path(args.results_dir or cfg["results_dir"])
    data_dir = args.data_dir or cfg["data_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir / "train.log")
    set_seed(cfg["seed"])

    device, dtype = get_device_and_dtype()
    if device != "cuda" and not args.allow_cpu:
        raise SystemExit("No CUDA GPU found. Run training on Colab (Runtime > Change runtime type > T4 GPU).")
    load_in_4bit = cfg.get("load_in_4bit", True) and device == "cuda"

    processor = load_processor(cfg["model_name"], cfg["min_pixels"], cfg["max_pixels"])
    train_ds = DocVQADataset(data_dir, "train", limit=args.limit)
    val_limit = args.val_limit or cfg.get("eval_loss_limit")
    eval_ds = DocVQADataset(data_dir, "val", limit=val_limit)
    val_rows = DocVQADataset(data_dir, "val", limit=args.val_limit or cfg["val_anls"]["limit"]).rows

    model = load_base_model(cfg["model_name"], cfg, load_in_4bit=load_in_4bit, device=device, dtype=dtype)
    model = attach_lora(model, cfg["lora"], quantized=load_in_4bit,
                        gradient_checkpointing=cfg["training"].get("gradient_checkpointing", True))
    report = trainable_parameter_report(model)
    report.update({"lora": cfg["lora"], "load_in_4bit": load_in_4bit, "dtype": str(dtype)})
    save_json(results_dir / "trainable_params.json", report)
    logger.info("Trainable params: %s / %s (%.3f%%)", f"{report['trainable_params']:,}",
                f"{report['total_params']:,}", report["trainable_percent"])

    trainer = build_trainer(model, processor, train_ds, eval_ds, cfg, output_dir, results_dir, device, dtype, val_rows)

    last_ckpt = None if args.no_resume else get_last_checkpoint(str(output_dir))
    if last_ckpt:
        logger.info("RESUMING from %s", last_ckpt)
    else:
        logger.info("Starting a fresh run (no checkpoint found in %s)", output_dir)

    eff_bs = trainer.args.per_device_train_batch_size * trainer.args.gradient_accumulation_steps
    logger.info("Train samples %d | effective batch %d | epochs %s", len(train_ds), eff_bs, trainer.args.num_train_epochs)
    t0 = time.time()
    train_result = trainer.train(resume_from_checkpoint=last_ckpt)
    minutes = (time.time() - t0) / 60

    trainer.save_model(str(output_dir / "final_adapter"))
    processor.save_pretrained(str(output_dir / "final_adapter"))
    summary = {
        "train_runtime_minutes": round(minutes, 1),
        "train_loss": train_result.training_loss,
        "global_step": trainer.state.global_step,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device == "cuda" else None,
        "resumed_from": last_ckpt,
        "log_history": trainer.state.log_history,
    }
    save_json(results_dir / "train_summary.json", summary)
    logger.info("Done in %.1f min. final_adapter -> %s | best_adapter -> %s",
                minutes, output_dir / "final_adapter", output_dir / "best_adapter")


if __name__ == "__main__":
    main()
