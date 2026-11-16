# Project guide: workflow, design decisions, Colab run-book, interview prep, pitfalls

---------------------------------------------------------------------------------------------------
## 1. The 30-second pitch

> "I fine-tuned a 3B vision-language model, Qwen2.5-VL, to answer questions about scanned business
> documents (forms and tables). I used QLoRA: the 3B LLM is frozen in 4-bit and I train only small
> LoRA adapters (~1% of the parameters), so it fits on a free 16 GB T4. I built the data pipeline
> with a document-level split to avoid leakage, measured ANLS (the official DocVQA metric) before and
> after, merged the adapter, and served it as a FastAPI service in Docker with input validation,
> health checks and a GPU lock. Every number comes from a script and is saved as JSON. The CPU-testable
> parts have pytest tests, including a tiny randomly-initialised Qwen2.5-VL that runs the whole
> training → merge pipeline on my laptop."

---------------------------------------------------------------------------------------------------
## 2. Overall workflow

```
Phase 1 data      stream DocVQA train → keep table/list + form → verify images → split by document → jsonl + images
Phase 2 metric    anls() + exact match, unit tested
Phase 3 baseline  base model, fixed prompt/pixels/greedy → results/baseline.json
Phase 4 train     4-bit base + LoRA, answer-only loss, val ANLS each epoch → best_adapter
Phase 5 eval      best adapter on TEST, same settings → finetuned.json → compare → error_analysis.csv
Phase 6 merge     fp16 base + adapter → merge_and_unload → verify ANLS on 100 val → models/merged
Phase 7 API       FastAPI /health /predict /predict/batch, validation, lock, mock-tested
Phase 8 Docker    GPU image (pytorch CUDA base) + CPU image, non-root, HEALTHCHECK, weights via volume
Phase 9 docs      README results table, Makefile, notebooks
```

Where things run:
* **Laptop (CPU):** writing code, all pytest tests, data-prep smoke test, API with mock or tiny model.
* **Colab T4:** data prep (full), baseline, training, final eval, merge, API with the real model.
* **Any NVIDIA machine with Docker:** the GPU container. **Laptop Docker:** CPU image or mock mode.

---------------------------------------------------------------------------------------------------
## 3. What each part uses and why (likely interview questions)

### Model: Qwen2.5-VL-3B-Instruct
* A ViT vision encoder (~670M params) with native dynamic resolution, plus an MLP "merger" that maps
  each 2×2 group of 14 px patches (= a 28×28 px block) to one token, plus a Qwen2.5 3B LLM.
* **Why this model:** strong OCR/document ability at 3B, fits a T4 in 4-bit, and transformers support
  it natively (`Qwen2_5_VLForConditionalGeneration`, transformers ≥ 4.49).
* **min_pixels / max_pixels:** a page is resized so that its pixel count falls in this range, and
  rounded to multiples of 28. Tokens ≈ pixels / 784. `max_pixels = 1280·28·28` → at most 1280 visual
  tokens. This is the main lever for speed and memory against accuracy on small text.

### QLoRA (bitsandbytes + peft)
* **4-bit NF4:** NormalFloat4 is a 4-bit data type whose levels match a normal distribution (that's how
  weights are distributed). **Double quantization** also quantizes the per-block scale constants,
  saving ~0.4 bits per parameter. The compute dtype is fp16 on a T4 (bf16 on Ampere+): weights are
  de-quantized on the fly for each matmul.
* **The vision tower is NOT quantized** (`llm_int8_skip_modules=["visual","lm_head"]`). It's small, it's
  frozen, and quantizing it hurts OCR-level detail.
* **LoRA:** W' = W + (α/r)·B·A with r=16, α=32 (scale 2). B starts at zero, so training starts from
  exactly the base model. Only A and B train (~1% of parameters). The adapter is ~100 MB, not 7 GB.
* **Targets:** q,k,v,o,gate,up,down of the **LLM only**. The **pitfall** is that the Qwen2.5-VL vision
  MLP *also* has layers named gate_proj/up_proj/down_proj, so a plain list would put LoRA into the
  vision tower. I use the regex `^(?!.*visual).*\.(q_proj|...)$`, and a test checks it on the real 3B
  module tree (252 layers matched, none in `visual`).
* **prepare_model_for_kbit_training:** freezes the base, upcasts norms/embeddings to fp32 for stability,
  and enables input gradients so gradient checkpointing works with a frozen embedding layer.
* **Gradient checkpointing** (`use_reentrant=False`): recomputes activations in the backward pass
  instead of storing them. This trades ~30% speed for a large memory saving and is needed at ~1300
  tokens per sample on 16 GB.
* **paged_adamw_8bit:** optimizer states in 8-bit, paged to CPU RAM on memory spikes.

### Data (datasets, streaming)
* Only the DocVQA **train** split is used, because test answers are hidden. Streaming avoids
  downloading the whole dataset (tens of GB).
* **Split by document (`ucsf_document_id`), not by question.** One page can have several questions.
  If the same page were in train and test, the model could memorize the page and the test score would
  be inflated (leakage). The script asserts zero overlap.
* Images are saved with their original bytes (no re-compression). Corrupt images are dropped after a
  full decode, because `verify()` alone misses truncated files.

### Collator: label masking
* The full chat is system → user [image + question] → assistant [answer]. Labels are a copy of
  input_ids with **everything set to -100 except the answer tokens + `<|im_end|>`**.
* Why: the loss should teach *answering*, not copying the prompt or predicting ~1000 image tokens.
  Keeping `<|im_end|>` teaches the model to **stop** after the answer.
* This is tested by decoding the unmasked tokens and asserting they equal the answer.
* **Left padding** is used everywhere, because batched generation needs prompts aligned on the right.

### Training (transformers.Trainer)
* Effective batch 16 (1 × 16 grad accumulation), lr 2e-4 (LoRA tolerates higher LRs than full
  fine-tuning), cosine schedule, 3% warmup, max_grad_norm 1.0, 2 epochs, fp16 mixed precision on T4.
* `remove_unused_columns=False`, because otherwise Trainer drops the raw fields the collator needs.
* `prediction_loss_only=True`, because otherwise eval accumulates full 152k-vocab logits per sample and OOMs.
* **Per-epoch val ANLS callback:** loss doesn't directly measure answer quality, so after each epoch
  it generates answers on val, logs ANLS, and saves the best adapter.
* **Checkpoint every 25 steps + at every epoch end, auto-resume** (`get_last_checkpoint`), because
  Colab disconnects.

### Metric: ANLS
* Normalize (lowercase, strip, collapse whitespace). NL = Levenshtein / max length. Score = 1−NL if
  NL < 0.5, else 0. Take the max over gold answers, then the mean over questions.
* Why not accuracy: "$1,250" vs "$1250" should get partial credit. Why the 0.5 cut-off: so that a wrong
  answer gets 0, not 0.3.

### Evaluation fairness
* The same prompt, `min/max_pixels`, greedy decoding and max_new_tokens=32 are used for every run.
  These come from one file (`configs/common.yaml`) and one function (`src/prompting.py`).
* The fine-tuned model runs on the 4-bit base (as trained). For a same-precision comparison, also
  report `baseline_4bit.json` or the merged fp16 model.

### Merge
* You can't cleanly add LoRA deltas into 4-bit weights, so the base is reloaded in fp16 and
  `merge_and_unload()` folds B·A into W.
* Verified by comparing ANLS of unmerged vs merged on 100 val samples. They should match up to fp16
  rounding. The tiny-model test checks that the logits match to 1e-4.
* Benefit: no peft at inference and no 4-bit de-quantization overhead, so it's simpler and usually faster.

### Serving (FastAPI + uvicorn)
* The model is loaded **once** in the `lifespan` startup hook.
* Endpoints are sync `def` (run in a thread pool) and use a **threading.Lock** around `generate()`.
  Two concurrent generations on a 16 GB GPU could OOM. Production upgrades would be a request queue
  plus dynamic batching (e.g. vLLM/TGI — excluded here on purpose).
* Validation: file type is detected from **magic bytes** (not the filename), there's a max size
  (413), a max pixel count (decompression bombs), question length limits, and PDFs are rendered to
  page 1 with pypdfium2. Error codes are 4xx with clear messages, 503 while loading or on OOM.
* `MockEngine` lets the API be tested and demoed without weights.

### Docker
* The GPU image is based on `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime` (CUDA + torch already
  matched). It runs as a **non-root** user and has a `HEALTHCHECK` on `/health` with a long start
  period while the model loads.
* **No weights or secrets in the image.** Weights are mounted at `/models`, or downloaded to an
  `HF_HOME` cache volume.
* There is one uvicorn worker, because each worker would load its own copy of the model.
* `docker-compose` reserves the GPU via `deploy.resources.reservations.devices`.
* `Dockerfile.cpu` (python:3.11-slim + CPU torch) is multi-arch, so it also runs on ARM laptops.

---------------------------------------------------------------------------------------------------
## 4. Run-book: Google Colab, step by step

**0. Put the code on GitHub (once, on the laptop):**
```powershell
cd C:\Users\lavan\OneDrive\Documents\Vision_language_doc\docvqa-qlora
git add . ; git commit -m "DocVQA QLoRA project"
# create an EMPTY repo named docvqa-qlora on github.com (public is easiest for Colab cloning), then:
git remote add origin https://github.com/<you>/docvqa-qlora.git
git branch -M main ; git push -u origin main
```
(For a private repo, clone in Colab with a token: `https://<token>@github.com/<you>/docvqa-qlora.git`,
and never commit the token.)

**1. Open `notebooks/colab_runner.ipynb` in Colab** (File → Open notebook → GitHub tab, or upload it).
Set the runtime to **T4 GPU**. Edit `REPO_URL` in cell 3.

**2. Run the cells in order:**

| cell | what it does | expected output / time |
|---|---|---|
| 1 GPU check | nvidia-smi, capability | `Tesla T4 capability 7.5 -> fp16` |
| 2 Drive | mount | asks for permission |
| 3 paths | HF_HOME on Drive | prints paths |
| 4 clone | git clone/pull | last commit |
| 5 install | pip install pinned versions | ~2 min; restart if asked, then re-run 3–4 |
| 6 tests | pytest | all passed (~1–2 min) |
| P1 list types | prints question_types | form, layout, table/list, handwritten, free_text, figure/diagram, Image/Photo, Yes/No, others |
| P1 smoke | 50 samples | `Overlap check PASSED` |
| P1 full | 3000 samples → Drive | ~10–25 min; counts ≈ 2400/300/300 |
| rsync | Drive → local disk | ~1 min |
| P3 smoke | 20 test samples | first run downloads 7.5 GB (~3–5 min) |
| P3 full | baseline on test | ~10–20 min → `baseline.json` |
| P4 smoke | 32 samples, 1 epoch | no OOM, finite loss, `val ANLS` line, checkpoint saved |
| P4 full | 2 epochs | **several hours** on a T4 (estimate: ~1.5–2.5 h/epoch); re-run to resume |
| P5 eval | adapter on test | → `finetuned.json` |
| P5 compare | prints the table | baseline vs fine-tuned ANLS / EM / latency |
| P6 merge | merge + verify | `PASS`; merged model in /content/models/merged |
| P7 API | uvicorn in background + curl | JSON answer for a test document |
| save | zip results + adapter | download to the laptop |

**3. After the run:** unzip the results into `results/` in the repo, fill in the README table, and commit.

**If Colab disconnects mid-training:** re-run cells 1–5 and the rsync cell, then the P4 full cell. The
log should say `RESUMING from .../checkpoint-N`.

**Free-tier time budget tips:** the T4 quota is a few hours a day. Options: run phases on different days
(everything persists on Drive), use `--limit 1000` for training, or lower `max_pixels` in
`configs/common.yaml` (e.g. `802816` = 1024·28·28). If you lower it, re-run the baseline so the
comparison stays fair.

---------------------------------------------------------------------------------------------------
## 5. How to test it fully

| level | what | where | command |
|---|---|---|---|
| unit | ANLS edge cases | laptop | `pytest tests/test_metrics.py` |
| unit | doc-level split, no overlap, corrupt-image detection | laptop | `pytest tests/test_prepare_dataset.py` |
| unit | label masking with the real processor (answer + `<|im_end|>` only, image tokens masked, left padding, max_pixels) | laptop | `pytest tests/test_collator.py` |
| unit | LoRA regex on the real 3B architecture (meta device, no weights) | laptop | `pytest tests/test_lora_targets.py` |
| integration | tiny random Qwen2.5-VL: LoRA → Trainer epoch → val-ANLS callback → checkpoint → merge equivalence → run_eval | laptop | `pytest tests/test_tiny_pipeline.py` |
| API | all endpoints, PNG/JPG/PDF, 400/413/415/422/500/503 | laptop | `pytest tests/test_api.py` |
| smoke | real data, 50 samples | laptop/Colab | `prepare_dataset --limit 50` |
| smoke | every GPU script with `--limit` | Colab | smoke cells in the notebook |
| e2e | API with the real model + curl | Colab | Phase 7 cells |
| e2e | Docker + curl | GPU machine / laptop (CPU image) | see README |

Laptop: `C:\Users\lavan\.venvs\docvqa\Scripts\python.exe -m pytest` → all tests pass (~40 s, the
first run downloads a few MB of processor/config files).

---------------------------------------------------------------------------------------------------
## 6. Expected results (estimates, not guarantees)

* **Baseline ANLS:** probably high, around **0.80–0.90** on this table/form subset at 1280 visual
  tokens. Qwen2.5-VL-3B reports ~94 ANLS on the official DocVQA test at higher resolution.
* **Fine-tuned:** a modest gain, typically **+1 to +5 ANLS points**. Most of it comes from *format*:
  answer-only replies, the annotators' style for dates/amounts, no extra words or trailing periods.
  Exact match usually improves more than ANLS.
* **Trainable params:** ~30M of ~3.75B (about **0.8–1%**, the exact figure is in `trainable_params.json`).
* **Latency on T4:** a few hundred ms per sample in batched eval, ~0.5–1.5 s for a single API request.
  The 4-bit model is often *slower* than fp16 (de-quantization overhead). That's a good point to raise.
* **Peak training memory:** roughly 8–13 GB on a T4 at batch 1 with gradient checkpointing.
* **Merge check:** unmerged vs merged ANLS within 0.01, only a few predictions differing.
* If fine-tuning shows **no gain or a small drop**, that's still a valid, explainable result: a strong
  base, possible pre-training contamination, small data. Show the error analysis and per-group table.
  Don't hide it.

---------------------------------------------------------------------------------------------------
## 7. How to show it to an interviewer

1. **README first** (2 min): problem, architecture diagram, results table, limitations.
2. **Live demo** (3 min): in Colab, run the Phase 7 cells, or run the Docker container. Open
   `http://localhost:8000/docs` (Swagger) or `/` (upload page). Ask a question about a form, then show
   a 415 by uploading a .txt file, to show validation.
3. **Code walk-through** (5 min), in this order:
   `configs/common.yaml` (one source of truth) → `src/data/collator.py` (masking) →
   `src/model/load_model.py` (`lora_target_regex` and why) → `src/train.py` (callback, resume) →
   `src/metrics.py` → `src/serve/app.py` (validation, lock).
4. **Evidence:** `results/*.json`, `02_error_analysis.ipynb` (10 worst failures with images), and
   `pytest` output.
5. **Honesty slide:** contamination risk, 4-bit vs fp16 comparison, what you'd do next.

**What you'd do next (good closing answer):** a private, truly in-domain dataset; a hyperparameter sweep
(r, lr, epochs, max_pixels); multi-page documents; an OCR-token-assisted prompt; vLLM/TGI with dynamic
batching for throughput; quantized serving (AWQ/GPTQ); monitoring of latency and answer-confidence drift.

---------------------------------------------------------------------------------------------------
## 8. Interview Q&A cheat-sheet

**Q: Why QLoRA instead of full fine-tuning?** Full fine-tuning of 3.75B params in mixed precision with
Adam needs ~16 bytes/param ≈ 60 GB. QLoRA needs the 4-bit weights (~2 GB for the LLM) plus small
adapters and optimizer states, so it fits on 16 GB. Quality is usually close to full fine-tuning for
narrow tasks.

**Q: What does r and alpha mean?** r is the rank of the update (capacity). alpha/r scales the update.
With alpha = 2r the effective LR of the adapter is stable when you change r.

**Q: Why not LoRA on the vision tower?** The task is reading text the encoder already sees well. The
bottleneck is answer formatting and reasoning in the LLM. Training the ViT costs memory, risks
overfitting on 3k samples, and the encoder is shared knowledge we don't want to damage.

**Q: Why fp16 not bf16 on a T4?** The T4 (Turing, sm_75) has no native bf16. Note that
`torch.cuda.is_bf16_supported()` can still return True there (emulation), so the code checks the
compute capability (≥ 8). fp16 has a small range (max 65504), so the vision tower runs in fp32 and
training uses loss scaling (GradScaler via `fp16=True`).

**Q: How do you know you didn't leak test data?** The split is by document id, with an assertion in
the script, a unit test, and the test split is only used for the final eval. Model selection used the
val split.

**Q: Why ANLS on val each epoch instead of only loss?** Loss counts token-level likelihood and can keep
improving while generated answers don't. ANLS is the metric we report, so we select on it.

**Q: How does batching work for generation?** Left padding plus an attention mask. Qwen's M-RoPE
position ids are computed from the attention mask, so padding doesn't shift positions.

**Q: What happens with a 4000×3000 scan?** The API resizes to max_pixels before the processor. The
processor's smart_resize keeps the aspect ratio and makes sides multiples of 28. Tiny text can become
unreadable, which is a known failure mode, visible in the error analysis.

**Q: How would you scale the API?** Several GPU replicas behind a load balancer; inside each, a queue
plus dynamic batching (vLLM supports Qwen2.5-VL); a timeout per request; autoscaling on queue length;
merged fp16 or AWQ weights; and caching image encodings for repeated documents.

**Q: Why is the merged model better for serving?** It's one set of weights, needs no peft at runtime,
and has no LoRA matmul or 4-bit de-quantization overhead. That gives lower latency and simpler deployment.

**Q: How did you test GPU code without a GPU?** Pure functions are unit-tested. The collator is tested
with the real processor. The LoRA regex is tested on the real architecture built on the meta device.
The whole train/merge/eval path is tested with a 2-layer random Qwen2.5-VL on CPU. On Colab every
script has a `--limit` smoke mode before the full run.

**Q: What is the hardest bug you hit / would hit?** Good candidates: LoRA silently attaching to vision
gate/up/down projections; eval OOM from accumulated logits (`prediction_loss_only`);
`is_bf16_supported()` returning True on a T4; fp16 overflow giving `!!!!` outputs; Colab disconnects
(solved with step checkpoints + auto-resume).

---------------------------------------------------------------------------------------------------
## 9. Common pitfalls and challenges (and what this repo does about them)

| pitfall | symptom | fix in this repo |
|---|---|---|
| LoRA regex hits the vision MLP | more trainable params than expected, slower, worse | negative-lookahead regex + assertions + test |
| Splitting by question, not document | inflated test score | `split_by_group` + `assert_no_overlap` |
| Loss on prompt/image tokens | model learns to echo, slow convergence | assistant-only masking, tested |
| Not training `<|im_end|>` | model rambles past the answer | end token included in labels |
| Right padding in batched generation | garbage for shorter prompts | left padding globally, asserted |
| bf16 on a T4 | very slow or errors | capability check → fp16 |
| fp16 overflow in the ViT | empty/"!!!!" answers, NaN loss | vision tower in fp32; guard logs an error |
| `device_map="auto"` offloading to CPU | 10× slower, 4-bit errors | `device_map={"": 0}` |
| Eval OOM inside Trainer | crash at the first epoch end | `prediction_loss_only=True` |
| `use_cache` off after training setup | very slow generation in the callback | `use_cache=True` passed to generate |
| Colab disconnects | lost hours | save_steps 25 + epoch-end save + auto-resume, outputs on Drive |
| Drive 15 GB quota | write failures | HF cache + adapter on Drive, merged model on local disk |
| Reading images from Drive | very slow dataloader | rsync dataset to /content first |
| pip versions drift | API/behaviour changes | pinned requirements, transformers 4.51.3 |
| Different prompt/pixels between evals | meaningless comparison | `common.yaml` + `prompting.py` shared everywhere |
| Merging into 4-bit weights | wrong merged model | reload base in fp16 before merging, then verify |
| Concurrent API requests | GPU OOM | lock around generate, 1 worker |
| Huge/malicious uploads | memory blow-up | size, pixel and type checks (magic bytes) |
| Baking weights/tokens into the image | huge image, leaked secrets | volume mount / HF cache volume |
| Windows laptop: no `make`, symlink warnings, AppControl blocking DLLs | local tooling hiccups | separate venv, PowerShell commands in this guide |
