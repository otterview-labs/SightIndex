import base64
import binascii
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Header, HTTPException

from app.api.deps import AppSettings
from app.config.settings import Settings
from app.schemas.attributes import (
    OPENAPI_VLM_STRUCTURED_RESPONSES,
    VLMStructuredAnalysisRequest,
    VLMStructuredAnalysisResponse,
)
from app.services.observation_index import attribute_labels
from app.services.vlm import VLMRuntimeError, VLMStructuredAnalysisService

router = APIRouter(prefix="/vlm", tags=["vlm"])


@router.post(
    "/structured-analysis",
    response_model=VLMStructuredAnalysisResponse,
    summary="Analyze image with VLM structured parser",
    description=(
        "Parse one person or vehicle image into normalized JSON attributes using the configured "
        "OpenAI-compatible VLM. This endpoint returns attributes only: it does not write "
        "`person_crops.attributes` and does not update indexes. Use "
        "`/api/attributes/person-crops/{crop_id}/analyze` for parse-and-persist workflows. "
        "When `VLM_SERVICE_API_KEY` is set, pass it as `Authorization: Bearer ...` or `X-API-Key`; "
        "`VLM_API_KEY` is reserved for calling the upstream VLM."
    ),
    responses=OPENAPI_VLM_STRUCTURED_RESPONSES,
)
def create_structured_analysis(
    payload: VLMStructuredAnalysisRequest,
    settings: AppSettings,
    authorization: Annotated[str | None, Header()] = None,
    x_api_key: Annotated[str | None, Header()] = None,
) -> VLMStructuredAnalysisResponse:
    _authorize_vlm_request(settings, authorization, x_api_key)
    image_base64 = _normalize_image_payload(payload.image_base64)
    suffix = _image_suffix(payload.image_filename)
    try:
        image_bytes = _decode_image_base64(image_base64)
        with tempfile.TemporaryDirectory(prefix="sightindex-vlm-structured-") as temp_dir:
            image_path = Path(temp_dir) / f"query{suffix}"
            image_path.write_bytes(image_bytes)
            attributes = VLMStructuredAnalysisService(settings).analyze_image(
                image_path,
                object_type=payload.object_type,
                bbox=payload.bbox.model_dump() if payload.bbox else None,
            )
    except VLMRuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return VLMStructuredAnalysisResponse(
        object_type=payload.object_type,
        bbox=payload.bbox,
        attributes=attributes,
        labels=attribute_labels(attributes, payload.label_language),
        label_language=payload.label_language,
        model=settings.vlm_model,
        provider=settings.vlm_provider,
    )


def _authorize_vlm_request(
    settings: Settings,
    authorization: str | None,
    x_api_key: str | None,
) -> None:
    expected = settings.vlm_service_api_key
    if not expected:
        return
    bearer_token = _bearer_token(authorization)
    if bearer_token == expected or x_api_key == expected:
        return
    raise HTTPException(status_code=401, detail="Invalid VLM API key")


def _bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    return token


def _decode_image_base64(image_base64: str) -> bytes:
    try:
        return base64.b64decode(image_base64, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise HTTPException(status_code=400, detail="image_base64 must be valid base64") from exc


def _normalize_image_payload(image_payload: str) -> str:
    if image_payload.startswith("data:"):
        _, _, image_payload = image_payload.partition(",")
    return image_payload


def _image_suffix(filename: str | None) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix in {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".webp"}:
        return suffix
    return ".jpg"
