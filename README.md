# DocVQA-QLoRA — fine-tuning Qwen2.5-VL-3B for document question answering + a Dockerized REST API

Upload a scanned form or table, ask a question ("What is the total amount?"), and get the answer.
**Qwen/Qwen2.5-VL-3B-Instruct** is fine-tuned with **QLoRA** (4-bit NF4 base, LoRA r=16) on a
~3000-question *table/list + form* subset of DocVQA. It is evaluated with **ANLS** before and
after fine-tuning and served with **FastAPI** inside **Docker**.

> Every number in this README comes from a script in this repo and is stored as JSON in `results/`.

## Results (test split, same prompt / pixels / greedy decoding)

| model | ANLS | exact match | latency (ms/sample) |
|---|---|---|---|
| Qwen2.5-VL-3B base (fp16) | _from `results/baseline.json`_ | | |
| + QLoRA adapter (4-bit base) | _from `results/finetuned.json`_ | | |

Trainable parameters: _from `results/trainable_params.json`_ · Hardware: Colab T4 16 GB ·
Training time: _from `results/train_summary.json`_ · Val ANLS per epoch: `results/val_anls_per_epoch.json`

(Fill in after the Colab run. `python -m src.compare` prints this table.)

## Architecture

```
                 ┌──────────── training (Colab T4) ─────────────┐
DocVQA (HF) ──►  prepare_dataset ──► train/val/test.jsonl + images   (split BY DOCUMENT, no leakage)
  streaming      filter table/list+form, verify images
                         │
                         ▼
        evaluate.py (baseline) ──► results/baseline.json
                         │
   train.py: Qwen2.5-VL-3B ── 4-bit NF4 LLM (frozen) ── vision tower fp16/fp32 (frozen)
             + LoRA r=16 on q,k,v,o,gate,up,down of the LLM only
             collator: loss only on answer tokens · per-epoch val ANLS → best_adapter
                         │
        evaluate.py (adapter) ──► results/finetuned.json ──► compare.py ──► error_analysis.csv
                         │
        merge_adapter.py: fp16 base + LoRA → merge_and_unload → models/merged
                         ▼
                 ┌──────────── serving (Docker) ────────────────┐
  client ──HTTP──► FastAPI /predict ─ validate (type/size/len) ─ resize ─► lock ─► model.generate ─► {answer, latency_ms, model_version}
```

## Repo layout

```
configs/      common.yaml (prompt, pixels, decoding: shared), data/train/eval/serve.yaml
src/data/     prepare_dataset.py, collator.py
src/model/    load_model.py   (4-bit loading, LoRA regex, adapter loading)
src/          train.py, evaluate.py, metrics.py, compare.py, merge_adapter.py, prompting.py, utils.py
src/serve/    app.py, schemas.py, inference.py, static/index.html (demo page)
tests/        metrics, data split, collator masking, LoRA targets, tiny end-to-end pipeline, API, compare
notebooks/    colab_runner.ipynb, 01_eda.ipynb, 02_error_analysis.ipynb
docs/GUIDE.md full walkthrough, design decisions, interview Q&A, pitfalls
```

## Reproduce

All GPU steps run on Google Colab (T4) via `notebooks/colab_runner.ipynb`, which only installs
dependencies and calls these commands. Every script supports `--limit N` for smoke tests.

| step | command (`make` target) |
|---|---|
| tests | `python -m pytest` (`make test`) |
| data | `python -m src.data.prepare_dataset --config configs/data.yaml` (`make data`) |
| baseline | `python -m src.evaluate --config configs/eval.yaml --split test --output results/baseline.json` (`make baseline`) |
| train | `python -m src.train --config configs/train.yaml` (`make train`; re-run to resume) |
| eval | `python -m src.evaluate ... --adapter outputs/qlora/best_adapter --load_in_4bit --output results/finetuned.json` (`make eval`) |
| compare | `python -m src.compare` (`make compare`) |
| merge | `python -m src.merge_adapter --adapter outputs/qlora/best_adapter --output models/merged` (`make merge`) |
| serve | `uvicorn src.serve.app:app --port 8000` (`make serve`, `make serve-mock`) |
| docker | `make docker` / `make docker-cpu` |

Hyperparameters live only in `configs/*.yaml`. Any value can be overridden with `--set key=value`
(for example `--set training.num_train_epochs=1`).

## API

```bash
docker build -t docvqa-api .
docker run --gpus all -p 8000:8000 -v ./models:/models docvqa-api        # GPU, merged model in ./models/merged
docker run -p 8000:8000 -v ./models:/models docvqa-api-cpu               # CPU image (Dockerfile.cpu)
docker run -p 8000:8000 -e MODEL_MODE=mock docvqa-api-cpu                # no weights: checks the plumbing

curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -F "file=@invoice.png" -F "question=What is the total amount?"
# {"answer":"$1,250.00","latency_ms":412.3,"model_version":"qwen2.5-vl-3b-docvqa-qlora-v1"}
curl -X POST http://localhost:8000/predict/batch -F "files=@a.png" -F "files=@b.pdf" -F "questions=Total?" -F "questions=Date?"
```

Open `http://localhost:8000/` for a small upload page, or `/docs` for Swagger UI.

* Inputs: PNG, JPEG, or PDF (first page). Max 10 MB, question ≤ 300 characters. Images are resized to `max_pixels`.
* Errors: 400 bad/empty input · 413 too large · 415 wrong type · 422 missing field · 503 model loading or GPU OOM.
* The model is loaded once at startup. A lock serializes `generate()` so concurrent requests can't OOM the GPU.
* Modes (env `MODEL_MODE`): `merged` (default, fp16/bf16), `adapter` (4-bit base + LoRA), `base`, `mock`.
* CPU mode is automatic when there is no GPU: bf16 weights with fp32 matmuls and `max_pixels = 256·28·28`.

## Known limitations

* **Possible contamination:** Qwen2.5-VL was very likely trained on DocVQA, so the baseline is already
  strong and gains are mostly in answer *format/style*. A truly private domain set would be a better test.
* The baseline is fp16 while the fine-tuned model is 4-bit. Use `baseline_4bit.json` or the merged fp16
  model (`finetuned_merged.json`) for a same-precision comparison.
* Single page only. Multi-page PDFs use only page 1. No OCR fallback.
* 3000 samples and 2 epochs on a T4; no hyperparameter search.
* CPU serving is slow (seconds per answer) and uses a lower image resolution than evaluation.
