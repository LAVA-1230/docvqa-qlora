"""Phase 7 - FastAPI REST API.

Endpoints
    GET  /              tiny HTML demo page (upload + question)
    GET  /health        liveness / readiness
    POST /predict       multipart: file (png/jpg/pdf) + question  ->  {answer, latency_ms, model_version}
    POST /predict/batch multipart: files[] + questions[] (same length)

Run
    uvicorn src.serve.app:app --host 0.0.0.0 --port 8000
Config: configs/serve.yaml (path via SERVE_CONFIG). Env overrides: MODEL_MODE, MERGED_MODEL_PATH,
ADAPTER_PATH, BASE_MODEL, DEVICE, MODEL_VERSION. MODEL_MODE=mock serves a fake model.
"""
from __future__ import annotations

import io
import logging
import math
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image

from src.serve.schemas import BatchItem, BatchPredictResponse, HealthResponse, PredictResponse
from src.utils import load_config, setup_logging

logger = logging.getLogger("docvqa")
ROOT = Path(__file__).resolve().parents[2]

ENV_OVERRIDES = {
    "MODEL_MODE": "mode",
    "MERGED_MODEL_PATH": "merged_model_path",
    "ADAPTER_PATH": "adapter_path",
    "BASE_MODEL": "model_name",
    "DEVICE": "device",
    "MODEL_VERSION": "model_version",
}


def load_serve_config() -> dict:
    cfg = load_config(os.environ.get("SERVE_CONFIG", ROOT / "configs" / "serve.yaml"))
    for env, key in ENV_OVERRIDES.items():
        if os.environ.get(env):
            cfg[key] = os.environ[env]
    return cfg


# --------------------------------------------------------------------------- input validation
def detect_kind(data: bytes) -> str | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpg"
    if data.startswith(b"%PDF"):
        return "pdf"
    return None


def render_pdf_first_page(data: bytes, scale: float) -> Image.Image:
    try:
        import pypdfium2 as pdfium
    except ImportError as e:
        raise HTTPException(415, "PDF support is not installed on this server (pypdfium2)") from e
    try:
        pdf = pdfium.PdfDocument(data)
        if len(pdf) == 0:
            raise HTTPException(400, "PDF has no pages")
        image = pdf[0].render(scale=scale).to_pil()
        pdf.close()
        return image
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Could not read PDF: {e}") from e


def resize_to_max_pixels(image: Image.Image, max_pixels: int) -> Image.Image:
    w, h = image.size
    if w * h <= max_pixels:
        return image
    scale = math.sqrt(max_pixels / (w * h))
    return image.resize((max(28, int(w * scale)), max(28, int(h * scale))), Image.BICUBIC)


def read_image(data: bytes, filename: str, cfg: dict, max_pixels: int) -> Image.Image:
    limits = cfg["limits"]
    if not data:
        raise HTTPException(400, f"Uploaded file '{filename}' is empty")
    if len(data) > limits["max_upload_mb"] * 1024 * 1024:
        raise HTTPException(413, f"File '{filename}' is larger than {limits['max_upload_mb']} MB")
    kind = detect_kind(data)
    if kind is None:
        raise HTTPException(415, f"Unsupported file type for '{filename}'. Allowed: PNG, JPEG, PDF (first page)")
    if kind == "pdf":
        image = render_pdf_first_page(data, cfg.get("pdf_render_scale", 2.0))
    else:
        try:
            with Image.open(io.BytesIO(data)) as probe:
                if probe.width * probe.height > limits["max_image_pixels"]:
                    raise HTTPException(413, f"Image '{filename}' has too many pixels ({probe.width}x{probe.height})")
                probe.verify()
            image = Image.open(io.BytesIO(data))
            image.load()
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(400, f"Corrupt or unreadable image '{filename}': {e}") from e
    return resize_to_max_pixels(image.convert("RGB"), max_pixels)


def validate_question(question: str, cfg: dict) -> str:
    q = (question or "").strip()
    if not q:
        raise HTTPException(400, "question must not be empty")
    max_chars = cfg["limits"]["max_question_chars"]
    if len(q) > max_chars:
        raise HTTPException(400, f"question is too long ({len(q)} > {max_chars} characters)")
    return q


# --------------------------------------------------------------------------- app factory
def create_app(engine=None, cfg: dict | None = None) -> FastAPI:
    cfg = cfg or load_serve_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.engine is None:
            from src.serve.inference import DocVQAEngine, MockEngine

            app.state.engine = MockEngine(cfg) if cfg["mode"] == "mock" else DocVQAEngine(cfg)
        if not app.state.engine.loaded:
            t0 = time.time()
            app.state.engine.load()  # load ONCE, before the server accepts requests
            logger.info("Model loaded in %.1fs", time.time() - t0)
        yield

    app = FastAPI(title="DocVQA API", version=str(cfg.get("model_version", "dev")),
                  description="Document visual question answering with QLoRA-tuned Qwen2.5-VL-3B",
                  lifespan=lifespan)
    app.state.engine = engine
    app.state.cfg = cfg

    def get_engine():
        eng = app.state.engine
        if eng is None or not eng.loaded:
            raise HTTPException(503, "Model is still loading, try again shortly")
        return eng

    def run_model(eng, images, questions):
        import torch

        try:
            t0 = time.perf_counter()
            answers = eng.predict(images, questions)
            return answers, round((time.perf_counter() - t0) * 1000, 1)
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            raise HTTPException(503, "GPU out of memory - try a smaller image or retry") from e
        except Exception as e:
            logger.exception("Inference failed")
            raise HTTPException(500, f"Inference failed: {type(e).__name__}: {e}") from e

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index():
        return (ROOT / "src" / "serve" / "static" / "index.html").read_text(encoding="utf-8")

    @app.get("/health", response_model=HealthResponse)
    def health():
        eng = app.state.engine
        return HealthResponse(status="ok" if eng is not None and eng.loaded else "loading",
                              model_loaded=bool(eng is not None and eng.loaded),
                              model_version=cfg["model_version"],
                              device=getattr(eng, "device", "unloaded"), mode=cfg["mode"])

    # Sync (def) endpoints run in FastAPI's thread pool; the engine lock serializes GPU access.
    @app.post("/predict", response_model=PredictResponse)
    def predict(file: UploadFile = File(...), question: str = Form(...)):
        eng = get_engine()
        q = validate_question(question, cfg)
        image = read_image(file.file.read(), file.filename or "upload", cfg, eng.max_pixels)
        answers, ms = run_model(eng, [image], [q])
        return PredictResponse(answer=answers[0], latency_ms=ms, model_version=cfg["model_version"])

    @app.post("/predict/batch", response_model=BatchPredictResponse)
    def predict_batch(files: List[UploadFile] = File(...), questions: List[str] = Form(...)):
        eng = get_engine()
        if len(files) != len(questions):
            raise HTTPException(400, f"got {len(files)} files but {len(questions)} questions; they must match")
        if len(files) > cfg["limits"]["max_batch_size"]:
            raise HTTPException(400, f"batch too large ({len(files)} > {cfg['limits']['max_batch_size']})")
        qs = [validate_question(q, cfg) for q in questions]
        images = [read_image(f.file.read(), f.filename or f"file{i}", cfg, eng.max_pixels) for i, f in enumerate(files)]
        answers, ms = run_model(eng, images, qs)
        return BatchPredictResponse(results=[BatchItem(question=q, answer=a) for q, a in zip(qs, answers)],
                                    latency_ms=ms, model_version=cfg["model_version"])

    return app


setup_logging()
# Cheap to import: only the config is read here; the model is loaded in `lifespan` when
# uvicorn starts. Tests build their own app with create_app(engine=MockEngine(...)).
app = create_app()
