from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])
    model_loaded: bool
    model_version: str
    device: str
    mode: str


class PredictResponse(BaseModel):
    answer: str = Field(..., examples=["$1,250.00"])
    latency_ms: float = Field(..., description="model inference time for this request")
    model_version: str


class BatchItem(BaseModel):
    question: str
    answer: str


class BatchPredictResponse(BaseModel):
    results: list[BatchItem]
    latency_ms: float
    model_version: str


class ErrorResponse(BaseModel):
    detail: str
