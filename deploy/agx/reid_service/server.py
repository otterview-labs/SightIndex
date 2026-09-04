"""SapiensID person re-identification, served over HTTP.

Wraps the vendored SapiensID inference chain: YOLO-pose and a face detector produce the 19
keypoints the backbone needs, then a 412M ViT emits a 4096-d identity embedding. The model is
1.5GB resident, which is why this runs beside the API rather than inside it.

  POST /embed        one image  -> one vector
  POST /embed-batch  N images   -> N vectors
  GET  /health       process liveness and warmup state
  GET  /ready        loaded runtime with authoritative identity
"""

from __future__ import annotations

import base64
import hashlib
import io
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, Header, HTTPException
from PIL import Image
from pydantic import BaseModel

SERVICE_ROOT = Path(__file__).parent
VENDOR_ROOT = SERVICE_ROOT / "sapiensid"
# Upstream resolves bundled weights relative to its own tree, so the vendored copy keeps that
# layout and imports from the same directory the eval harness does.
VENDOR_SRC = VENDOR_ROOT / "tasks" / "sapiensID" / "src"
APP_ROOT = Path(os.environ.get("SIGHTINDEX_ROOT", SERVICE_ROOT.parents[2]))

CHECKPOINT_DIR = Path(
    os.environ.get("REID_CHECKPOINT_DIR", str(APP_ROOT / "data/models/sapiensid_wb12m"))
)
API_KEY = os.environ.get("REID_SERVICE_API_KEY", "")
# Bump when the eval-time preprocessing changes semantics. Vectors from different preprocessing
# are not comparable, so this participates in the client's index fingerprint.
PREPROCESS_VERSION = os.environ.get("REID_PREPROCESS_VERSION", "squarepad-v1")
MODEL_ID = os.environ.get("REID_MODEL", CHECKPOINT_DIR.name)
DEVICE = os.environ.get("REID_DEVICE", "")
BATCH_LIMIT = int(os.environ.get("REID_BATCH_LIMIT", "16"))
WARMUP_ON_START = os.environ.get("REID_WARMUP_ON_START", "true").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
ALIGNER_CONFIG = VENDOR_SRC / "aligners/configs/yolo_dfa.yaml"
DFA_CHECKPOINT = (
    VENDOR_SRC
    / "aligners/keypoint_predictor/pretrained_models/aligners/"
    "dfa_mobilenetv4_medium/mobilenetv4_Final.pth"
)
# The vendored upstream aligner resolves this exact cache path internally.
POSE_CHECKPOINT = Path.home() / ".cache/yolov8n-pose.pt"

for _entry in (VENDOR_ROOT / "tasks" / "sapiensID", VENDOR_ROOT):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

# Upstream's aligner restores NumPy aliases for mxnet ("np.float = np.float_") at import time.
# np.float_ itself was removed in NumPy 2.0, so that shim raises on a modern install and the
# model never loads. Restore the alias here instead of patching the vendored tree, which stays
# byte-identical to upstream. mxnet is never imported on the inference path.
if not hasattr(np, "float_"):
    np.float_ = np.float64  # type: ignore[attr-defined]

# The bundled aligner checkpoints were serialised on CUDA, so a CPU or MPS host cannot load them
# with torch's defaults. Route every load through the resolved device.
_TORCH_LOAD = torch.load


def _resolve_device() -> str:
    if DEVICE:
        return DEVICE
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


_DEVICE = _resolve_device()


def _patched_load(*args: Any, **kwargs: Any) -> Any:
    kwargs.setdefault("map_location", _DEVICE if _DEVICE != "mps" else "cpu")
    kwargs.setdefault("weights_only", False)
    return _TORCH_LOAD(*args, **kwargs)


torch.load = _patched_load

from omegaconf import OmegaConf  # noqa: E402
from src.aligners import get_aligner  # noqa: E402
from src.models import get_model  # noqa: E402

_runtime: dict[str, Any] | None = None
_runtime_error: str | None = None
# One model, one GPU: serialise inference rather than letting requests interleave on it.
_runtime_lock = threading.Lock()
_warmup_start_lock = threading.Lock()
_warmup_started = False


def _warm_runtime() -> None:
    global _runtime_error
    try:
        _load_runtime()
    except HTTPException as exc:
        _runtime_error = str(exc.detail)
    except Exception as exc:  # the readiness endpoint exposes a bounded summary
        _runtime_error = f"{type(exc).__name__}: {exc}"[:500]


def _start_warmup() -> None:
    """Start the single background load, including lazy readiness-triggered warmup."""

    global _warmup_started
    if _runtime is not None or _warmup_started:
        return
    with _warmup_start_lock:
        if _runtime is not None or _warmup_started:
            return
        _warmup_started = True
        threading.Thread(
            target=_warm_runtime,
            name="sightindex-reid-warmup",
            daemon=True,
        ).start()


@asynccontextmanager
async def lifespan(_: FastAPI):
    if WARMUP_ON_START:
        _start_warmup()
    yield


app = FastAPI(title="SightIndex SapiensID ReID", version="1.1", lifespan=lifespan)


class EmbedRequest(BaseModel):
    image_base64: str | None = None
    image_url: str | None = None


class EmbedBatchRequest(BaseModel):
    images: list[EmbedRequest]


def _check_auth(authorization: str | None) -> None:
    if not API_KEY:
        return
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="invalid api key")


def _file_sha256(path: Path) -> str:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise HTTPException(status_code=503, detail=f"checkpoint changed while hashing: {path}")
    return digest.hexdigest()


def _pipeline_files(model_pth: Path, model_yaml: Path) -> tuple[tuple[str, Path], ...]:
    """Every external asset that can change the emitted identity vector."""

    return (
        ("model.pth", model_pth),
        ("model.yaml", model_yaml),
        ("yolo_dfa.yaml", ALIGNER_CONFIG),
        ("mobilenetv4_Final.pth", DFA_CHECKPOINT),
        ("yolov8n-pose.pt", POSE_CHECKPOINT),
    )


def _checkpoint_revision(files: tuple[tuple[str, Path], ...]) -> str:
    """Fingerprint the complete model + pose/face alignment inference manifest."""

    manifest = "\n".join(f"{name}:{_file_sha256(path)}" for name, path in files)
    composite = hashlib.sha256(manifest.encode("ascii")).hexdigest()
    return f"sha256:{composite}"


def _asset_stats(files: tuple[tuple[str, Path], ...]) -> dict[str, tuple[int, int, int]]:
    try:
        return {
            name: (stat.st_ino, stat.st_size, stat.st_mtime_ns)
            for name, path in files
            if (stat := path.stat())
        }
    except OSError as exc:
        raise HTTPException(status_code=503, detail=f"ReID inference asset missing: {exc}") from exc


def _load_runtime() -> dict[str, Any]:
    global _runtime, _runtime_error
    if _runtime is not None:
        return _runtime
    with _runtime_lock:
        if _runtime is not None:
            return _runtime
        model_yaml = CHECKPOINT_DIR / "model.yaml"
        model_pth = CHECKPOINT_DIR / "model.pth"
        if not model_yaml.is_file() or not model_pth.is_file():
            raise HTTPException(
                status_code=503,
                detail=f"checkpoint missing under {CHECKPOINT_DIR} (need model.yaml + model.pth)",
            )
        pipeline_files = _pipeline_files(model_pth, model_yaml)
        before_stats = _asset_stats(pipeline_files)
        started = time.time()
        model_cfg = OmegaConf.load(model_yaml)
        model = get_model(model_cfg)
        model.load_state_dict_from_path(str(model_pth))
        model.eval()
        model.to(_DEVICE)

        aligner_cfg = OmegaConf.load(ALIGNER_CONFIG)
        # These interpolate from ${models.*} inside the upstream eval harness.
        aligner_cfg.rgb_mean = model_cfg.rgb_mean
        aligner_cfg.rgb_std = model_cfg.rgb_std
        aligner = get_aligner(aligner_cfg)
        # eval() here does not return self, so these cannot be chained.
        aligner.eval()
        aligner.to(_DEVICE)

        if _asset_stats(pipeline_files) != before_stats:
            raise HTTPException(
                status_code=503,
                detail="ReID inference assets changed while the runtime was loading",
            )
        # Hash after path-based loaders finish, and each file hash independently verifies that
        # size/mtime stayed stable while it was read. This closes the deployment replacement
        # window between publishing an identity and loading the corresponding bytes.
        checkpoint_revision = _checkpoint_revision(pipeline_files)
        if _asset_stats(pipeline_files) != before_stats:
            raise HTTPException(
                status_code=503,
                detail="ReID inference assets changed while the revision was calculated",
            )

        _runtime = {
            "model": model,
            "aligner": aligner,
            # The model's own eval-time transform: ToTensor, SquarePad(fill=1), Resize,
            # Normalize. Rebuilding it here would only invite drift from upstream; an earlier
            # version did, and stretched rectangular crops square instead of padding them.
            "transform": model.make_test_transform(),
            "input_size": tuple(model_cfg.input_size),
            "dim": int(model_cfg.output_dim),
            "checkpoint_revision": checkpoint_revision,
            "load_seconds": round(time.time() - started, 2),
        }
        _runtime_error = None
        return _runtime


def _decode(payload: EmbedRequest) -> Image.Image:
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


def _embed(images: list[Image.Image]) -> tuple[list[list[float]], dict[str, Any]]:
    runtime = _load_runtime()
    transform = runtime["transform"]
    batch = torch.stack([transform(image) for image in images]).to(_DEVICE)
    started = time.time()
    with _runtime_lock, torch.no_grad():
        keypoints, masks = runtime["aligner"](batch)
        output = runtime["model"](batch, foreground_masks=masks, ldmks=keypoints)
        feature = output["pooler_output"] if isinstance(output, dict) else output
        # L2-normalise here so callers can treat Milvus inner product as cosine similarity.
        feature = torch.nn.functional.normalize(feature.float(), dim=1)
    vectors = feature.cpu().tolist()
    return vectors, {
        "dim": runtime["dim"],
        "model": MODEL_ID,
        "checkpoint_revision": runtime["checkpoint_revision"],
        "preprocess_version": PREPROCESS_VERSION,
        "device": _DEVICE,
        "elapsed_seconds": round(time.time() - started, 3),
    }


@app.get("/health")
def health() -> dict[str, Any]:
    loaded = _runtime is not None
    pipeline_files = _pipeline_files(
        CHECKPOINT_DIR / "model.pth",
        CHECKPOINT_DIR / "model.yaml",
    )
    missing_assets = [name for name, path in pipeline_files if not path.is_file()]
    return {
        "status": "ok",
        "loaded": loaded,
        "device": _DEVICE,
        "checkpoint": str(CHECKPOINT_DIR),
        "checkpoint_present": (CHECKPOINT_DIR / "model.pth").is_file(),
        "config_present": (CHECKPOINT_DIR / "model.yaml").is_file(),
        "pipeline_assets_present": not missing_assets,
        "missing_assets": missing_assets,
        "warmup_error": _runtime_error,
        # Exact identity facts are authoritative only after the bytes have been loaded and
        # hashed. Liveness intentionally does not pretend an unloaded service is ready.
        "model": MODEL_ID,
        "checkpoint_revision": _runtime["checkpoint_revision"] if loaded else None,
        "embedding_dim": _runtime["dim"] if loaded else None,
        "preprocess_version": PREPROCESS_VERSION,
    }


@app.get("/ready")
def ready() -> dict[str, Any]:
    if _runtime is None:
        # With startup warmup disabled, readiness is the deliberate lazy-load trigger. This
        # avoids a configuration deadlock where clients refuse to call /embed until /ready is
        # healthy but no endpoint ever starts loading the model.
        _start_warmup()
        raise HTTPException(
            status_code=503,
            detail=_runtime_error or "ReID model is still loading",
        )
    return health()


@app.post("/embed")
def embed(
    payload: EmbedRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _check_auth(authorization)
    vectors, meta = _embed([_decode(payload)])
    return {
        "embedding": vectors[0],
        "provider": "sapiensid",
        **meta,
    }


@app.post("/embed-batch")
def embed_batch(
    payload: EmbedBatchRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _check_auth(authorization)
    if not payload.images:
        raise HTTPException(status_code=400, detail="images required")
    if len(payload.images) > BATCH_LIMIT:
        raise HTTPException(
            status_code=400,
            detail=f"at most {BATCH_LIMIT} images per request",
        )
    vectors, meta = _embed([_decode(item) for item in payload.images])
    return {
        "embeddings": vectors,
        "provider": "sapiensid",
        **meta,
    }
