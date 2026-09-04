import base64
import binascii
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.config.settings import Settings, get_settings
from app.schemas.scene_summary import SceneSummaryRequest
from app.services.vlm import VLMRuntimeError, VLMSceneSummaryService

router = APIRouter(prefix="/api/vlm", tags=["vlm"])


@router.post(
    "/scene-summary",
)
def summarize_scene(
    payload: SceneSummaryRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict[str, object]:
    try:
        image_bytes = _decode_image_base64(payload.image_base64)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    service = VLMSceneSummaryService(settings)
    suffix = _safe_suffix(payload.image_filename)
    with tempfile.TemporaryDirectory(prefix="sightindex-scene-summary-") as temp_dir:
        image_path = Path(temp_dir) / f"query{suffix}"
        image_path.write_bytes(image_bytes)
        try:
            if payload.context:
                return service.summarize_image(image_path, context=payload.context)
            return {"labels": service.summarize_labels(image_path)}
        except VLMRuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc


def _decode_image_base64(value: str) -> bytes:
    text = value.strip()
    if "," in text and text.split(",", 1)[0].lower().startswith("data:"):
        text = text.split(",", 1)[1]
    try:
        return base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("image_base64 must be valid base64") from exc


def _safe_suffix(filename: str | None) -> str:
    if not filename:
        return ".jpg"
    suffix = Path(filename).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        return suffix
    return ".jpg"
