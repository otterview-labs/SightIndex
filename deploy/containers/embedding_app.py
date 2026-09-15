import base64
import hashlib
import importlib.util
import io
import logging
import math
import secrets
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import Field

from app.api.upload_limits import UploadSizeLimitMiddleware
from app.config.settings import Settings
from app.schemas.embeddings import VisualEmbeddingRequest, VisualEmbeddingResponse
from deploy.containers.download_embedding_model import WEIGHTS_SHA256, digest

logger = logging.getLogger(__name__)
MODEL_ID = "Qwen/Qwen3-VL-Embedding-2B"
EMBEDDING_DIM = 2048
MAX_PIXELS = 768 * 32 * 32
MAX_LENGTH = 4096
SCRIPT_SHA256 = "8ffa74a1a6bb759610c57865ea416fd4daf9936cb787520e1112a3e1d547f36a"


class EmbeddingRequest(VisualEmbeddingRequest):
    text: str | None = Field(default=None, max_length=8192)
    instruction: str | None = Field(default=None, max_length=1024)


class EmbeddingSettings(Settings):
    upload_image_max_bytes: int = Field(default=20 * 1024 * 1024, ge=1)
    upload_video_max_bytes: int = Field(default=128 * 1024 * 1024, ge=1)
    upload_image_max_pixels: int = Field(default=25_000_000, ge=1)


def load_model(model_dir: Path) -> Any:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Qwen embedding requires its assigned GPU; CPU fallback is disabled")
    script = model_dir / "scripts" / "qwen3_vl_embedding.py"
    if hashlib.sha256(script.read_bytes()).hexdigest() != SCRIPT_SHA256:
        raise RuntimeError("Unverified Qwen inference script")
    if digest(model_dir / "model.safetensors") != WEIGHTS_SHA256:
        raise RuntimeError("Unverified Qwen weights")
    specification = importlib.util.spec_from_file_location("sightindex_qwen_runtime", script)
    if specification is None or specification.loader is None:
        raise RuntimeError("Could not load the verified Qwen inference module")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(0.60)
    return module.Qwen3VLEmbedder(
        model_name_or_path=str(model_dir),
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
        max_length=MAX_LENGTH,
        max_pixels=MAX_PIXELS,
    )


def encode(model: Any, item: dict[str, Any]) -> list[float]:
    result = model.process([item])[0]
    vector = [float(value) for value in result.tolist()]
    if len(vector) != EMBEDDING_DIM or not all(math.isfinite(value) for value in vector):
        raise RuntimeError("Model returned an invalid embedding")
    norm = math.sqrt(sum(value * value for value in vector))
    if not 0.99 <= norm <= 1.01:
        raise RuntimeError("Model returned an unnormalized embedding")
    return vector


def decode_image(payload: str, settings: EmbeddingSettings) -> Image.Image:
    data = base64.b64decode(payload, validate=True)
    if len(data) > settings.upload_image_max_bytes:
        raise HTTPException(413, "Image exceeds the byte limit")
    try:
        with Image.open(io.BytesIO(data), formats=["JPEG", "PNG", "WEBP", "BMP", "GIF"]) as image:
            if image.width * image.height > settings.upload_image_max_pixels:
                raise HTTPException(413, "Image exceeds the pixel limit")
            image.load()
            with ImageOps.exif_transpose(image) as oriented:
                return oriented.convert("RGB")
    except Image.DecompressionBombError as exc:
        raise HTTPException(413, "Image exceeds the pixel limit") from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(400, "Invalid image") from exc


def create_app() -> FastAPI:
    settings = EmbeddingSettings()
    api_key = settings.visual_embedding_service_api_key
    if not api_key:
        raise RuntimeError("VISUAL_EMBEDDING_SERVICE_API_KEY is required")
    inference_slot = threading.BoundedSemaphore(1)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        model = load_model(Path(settings.visual_embedding_model))
        encode(model, {"text": "embedding readiness check"})
        with Image.new("RGB", (64, 64), (96, 96, 96)) as image:
            encode(model, {"image": image})
        app.state.model = model
        app.state.ready = True
        yield
        app.state.ready = False

    app = FastAPI(title="SightIndex Qwen Embedding", lifespan=lifespan)
    app.state.ready = False
    app.add_middleware(UploadSizeLimitMiddleware, settings=settings)

    @app.middleware("http")
    async def authentication(request: Request, call_next):
        if request.url.path != "/health":
            authorization = request.headers.get("authorization", "")
            scheme, _, bearer = authorization.partition(" ")
            token = bearer if scheme.lower() == "bearer" else request.headers.get("x-api-key", "")
            if not secrets.compare_digest(token.encode(), api_key.encode()):
                return JSONResponse({"detail": "Invalid embedding API key"}, status_code=401)
        return await call_next(request)

    @app.get("/health")
    def health() -> JSONResponse:
        return JSONResponse(
            {
                "ready": app.state.ready,
                "model": MODEL_ID,
                "dim": EMBEDDING_DIM,
                "weights_sha256": WEIGHTS_SHA256,
                "max_pixels": MAX_PIXELS,
                "max_length": MAX_LENGTH,
            },
            status_code=200 if app.state.ready else 503,
        )

    @app.post("/api/embeddings/visual", response_model=VisualEmbeddingResponse)
    def embedding(payload: EmbeddingRequest) -> VisualEmbeddingResponse:
        if not app.state.ready:
            raise HTTPException(503, "Embedding model is not ready")
        if not inference_slot.acquire(blocking=False):
            raise HTTPException(503, "Embedding service is busy; retry later")
        try:
            if payload.text:
                vector = encode(
                    app.state.model,
                    {
                        "text": payload.text,
                        "instruction": payload.instruction or settings.visual_embedding_instruction,
                    },
                )
            else:
                with decode_image(payload.image_base64 or "", settings) as image:
                    vector = encode(app.state.model, {"image": image})
            return VisualEmbeddingResponse(
                embedding=vector, dim=len(vector), model=MODEL_ID, provider="qwen3_vl"
            )
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Qwen embedding inference failed")
            raise HTTPException(503, "Embedding inference failed") from exc
        finally:
            inference_slot.release()

    return app
