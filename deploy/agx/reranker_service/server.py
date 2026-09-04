from __future__ import annotations

import base64
import io
import os
import sys
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

APP_ROOT = Path(os.environ.get("SIGHTINDEX_ROOT", Path(__file__).resolve().parents[3]))
MODEL_PATH = os.environ.get(
    "VLM_RERANK_MODEL",
    str(APP_ROOT / "models/modelscope/Qwen/Qwen3-VL-Reranker-2B"),
)
API_KEY = os.environ.get("VLM_RERANK_SERVICE_API_KEY", "")

if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from app.services.rerank import _cached_qwen3_vl_script_reranker  # noqa: E402

app = FastAPI(title="SightIndex Qwen3-VL Reranker", version="1.0")
_model = None


class RerankRequest(BaseModel):
    query: str
    image_base64: str | None = None
    image_url: str | None = None
    image_filename: str | None = None
    attributes: dict[str, Any] | None = Field(default_factory=dict)


def _check_auth(authorization: str | None) -> None:
    if not API_KEY:
        return
    expected = f"Bearer {API_KEY}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="invalid api key")


def _model_runtime():
    global _model
    if _model is None:
        _model = _cached_qwen3_vl_script_reranker(MODEL_PATH)
    return _model


def _decode_image(payload: RerankRequest) -> Image.Image:
    if payload.image_base64:
        try:
            raw = base64.b64decode(payload.image_base64)
            return Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"invalid image_base64: {exc}") from exc
    if payload.image_url:
        path = payload.image_url
        if path.startswith("/data/"):
            path = str(APP_ROOT / path.lstrip("/"))
        elif path.startswith("data/"):
            path = str(APP_ROOT / path)
        try:
            return Image.open(path).convert("RGB")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"invalid image_url: {exc}") from exc
    raise HTTPException(status_code=400, detail="image_base64 or image_url required")


@app.get("/health")
def health() -> dict[str, Any]:
    model = _model
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model": MODEL_PATH,
        "device": (
            str(getattr(model, "device", "not_loaded")) if model is not None else "not_loaded"
        ),
    }


@app.post("/api/rerank")
def rerank(
    payload: RerankRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _check_auth(authorization)
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="query required")
    model = _model_runtime()
    image = _decode_image(payload)
    started = time.time()
    scores = model.process({"query": {"text": payload.query}, "documents": [{"image": image}]})
    score = float(scores[0]) if scores else 0.0
    return {
        "score": score,
        "matched": score >= float(os.environ.get("VLM_RERANK_MATCH_THRESHOLD", "0.5")),
        "reason": "reranker score",
        "model": MODEL_PATH,
        "provider": "qwen3_vl_reranker_gpu",
        "device": str(getattr(model, "device", "unknown")),
        "elapsed_seconds": round(time.time() - started, 3),
    }


@app.post("/rerank")
def rerank_alias(
    payload: RerankRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    return rerank(payload, authorization)
