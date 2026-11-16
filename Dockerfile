# GPU image. Model weights are NOT baked in: mount them at /models (or use MODEL_MODE=base to
# download from the Hugging Face Hub into the /cache volume at startup). No secrets in the image.
#
#   docker build -t docvqa-api .
#   docker run --gpus all -p 8000:8000 -v ./models:/models docvqa-api
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    MODEL_MODE=merged \
    MERGED_MODEL_PATH=/models/merged \
    ADAPTER_PATH=/models/adapter

WORKDIR /app
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt bitsandbytes==0.45.5

# non-root user
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /models /cache/huggingface \
    && chown -R appuser:appuser /app /models /cache

COPY --chown=appuser:appuser configs ./configs
COPY --chown=appuser:appuser src ./src
USER appuser

EXPOSE 8000
# start-period is long because loading a 3B model can take minutes on first start
HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=5).status == 200 else 1)"

# 1 worker = 1 copy of the model in GPU memory; requests are serialized by a lock in the engine.
CMD ["uvicorn", "src.serve.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
