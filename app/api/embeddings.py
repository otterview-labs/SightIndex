import base64
import binascii
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Header, HTTPException

from app.api.deps import AppSettings
from app.config.settings import Settings
from app.schemas.embeddings import (
    OPENAPI_IMAGE_VECTOR_RESPONSES,
    ImageVectorRequest,
    ImageVectorResponse,
    OpenAIEmbeddingData,
    OpenAIEmbeddingRequest,
    OpenAIEmbeddingResponse,
    VisualEmbeddingRequest,
    VisualEmbeddingResponse,
    VisualRerankRequest,
    VisualRerankResponse,
)
from app.services.embeddings import EmbeddingRuntimeError, VisualEmbeddingService
from app.services.rerank import VisualRerankerService, VLMRerankService, VLMRuntimeError

router = APIRouter(prefix="/embeddings", tags=["embeddings"])
openai_router = APIRouter(prefix="/v1", tags=["openai-compatible"])


@router.post("/visual", response_model=VisualEmbeddingResponse)
def create_visual_embedding(
    payload: VisualEmbeddingRequest,
    settings: AppSettings,
    authorization: Annotated[str | None, Header()] = None,
    x_api_key: Annotated[str | None, Header()] = None,
) -> VisualEmbeddingResponse:
    _authorize_embedding_request(settings, authorization, x_api_key)
    service_settings = settings
    if payload.instruction:
        service_settings = settings.model_copy(
            update={"visual_embedding_instruction": payload.instruction}
        )
    service = VisualEmbeddingService(service_settings)
    try:
        if payload.text:
            embedding = service.embed_text(payload.text)
        else:
            embedding = _embed_image_payload(service, payload)
    except EmbeddingRuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return VisualEmbeddingResponse(
        embedding=embedding,
        dim=len(embedding),
        model=service_settings.visual_embedding_model,
        provider=service_settings.visual_embedding_provider,
    )


@router.post(
    "/image-vector",
    response_model=ImageVectorResponse,
    summary="Create text or image embedding vector",
    description=(
        "Generate one visual retrieval embedding vector from either a text query or a base64 "
        "image. Provide exactly one of `text` or `image_base64`. This endpoint does not write "
        "Milvus and does not persist metadata. It uses the configured "
        "`VISUAL_EMBEDDING_PROVIDER`; when "
        "`VISUAL_EMBEDDING_SERVICE_API_KEY` is set, pass it as `Authorization: Bearer ...` or "
        "`X-API-Key`."
    ),
    responses=OPENAPI_IMAGE_VECTOR_RESPONSES,
)
def create_image_vector(
    payload: ImageVectorRequest,
    settings: AppSettings,
    authorization: Annotated[str | None, Header()] = None,
    x_api_key: Annotated[str | None, Header()] = None,
) -> ImageVectorResponse:
    _authorize_embedding_request(settings, authorization, x_api_key)
    service_settings = settings
    if payload.instruction:
        service_settings = settings.model_copy(
            update={"visual_embedding_instruction": payload.instruction}
        )
    service = VisualEmbeddingService(service_settings)
    try:
        if payload.text:
            vector = service.embed_text(payload.text)
            input_type = "text"
        else:
            vector = _embed_image_base64(
                service,
                payload.image_base64 or "",
                payload.image_filename,
                temp_prefix="sightindex-image-vector-",
            )
            input_type = "image"
    except EmbeddingRuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return ImageVectorResponse(
        embedding=vector,
        dim=len(vector),
        model=service_settings.visual_embedding_model,
        provider=service_settings.visual_embedding_provider,
        input_type=input_type,
        image_filename=payload.image_filename,
    )


@router.post("/rerank", response_model=VisualRerankResponse)
def create_visual_rerank_score(
    payload: VisualRerankRequest,
    settings: AppSettings,
    authorization: Annotated[str | None, Header()] = None,
    x_api_key: Annotated[str | None, Header()] = None,
) -> VisualRerankResponse:
    _authorize_embedding_request(settings, authorization, x_api_key)
    if settings.vlm_rerank_service_url:
        service = VLMRerankService(settings)
    else:
        service = VisualRerankerService(settings)
    try:
        image_base64 = _normalize_image_payload(payload.image_base64 or payload.image or "")
        suffix = _image_suffix(payload.image_filename)
        with tempfile.TemporaryDirectory(prefix="sightindex-rerank-") as temp_dir:
            image_path = Path(temp_dir) / f"query{suffix}"
            image_path.write_bytes(_decode_image_base64(image_base64))
            decision = service.rerank_image(
                payload.query,
                image_path,
                payload.attributes or {},
            )
    except VLMRuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return VisualRerankResponse(
        score=decision.score,
        matched=decision.matched,
        reason=decision.reason,
        model=settings.vlm_rerank_model,
        provider=settings.vlm_rerank_provider,
    )


@openai_router.post("/embeddings", response_model=OpenAIEmbeddingResponse)
def create_openai_compatible_embedding(
    payload: OpenAIEmbeddingRequest,
    settings: AppSettings,
    authorization: Annotated[str | None, Header()] = None,
    x_api_key: Annotated[str | None, Header()] = None,
) -> OpenAIEmbeddingResponse:
    _authorize_embedding_request(settings, authorization, x_api_key)
    service_settings = settings
    if payload.model:
        service_settings = settings.model_copy(update={"visual_embedding_model": payload.model})
    if payload.dimensions:
        service_settings = service_settings.model_copy(
            update={"visual_embedding_dim": payload.dimensions}
        )
    service = VisualEmbeddingService(service_settings)
    inputs = payload.input if isinstance(payload.input, list) else [payload.input]
    data: list[OpenAIEmbeddingData] = []
    try:
        for index, item in enumerate(inputs):
            embedding = _embed_openai_compatible_item(service, item)
            data.append(OpenAIEmbeddingData(index=index, embedding=embedding))
    except EmbeddingRuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return OpenAIEmbeddingResponse(
        data=data,
        model=service_settings.visual_embedding_model,
    )


def _authorize_embedding_request(
    settings: Settings,
    authorization: str | None,
    x_api_key: str | None,
) -> None:
    expected = settings.visual_embedding_service_api_key
    if not expected:
        return
    bearer_token = _bearer_token(authorization)
    if bearer_token == expected or x_api_key == expected:
        return
    raise HTTPException(status_code=401, detail="Invalid visual embedding service API key")


def _bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    return token


def _embed_openai_compatible_item(
    service: VisualEmbeddingService,
    item: object,
) -> list[float]:
    if isinstance(item, str):
        if item.startswith("data:image/"):
            return _embed_data_url(service, item)
        return service.embed_text(item)
    if isinstance(item, dict):
        item_type = item.get("type")
        if item_type == "text" and isinstance(item.get("text"), str):
            return service.embed_text(item["text"])
        if item_type == "image_url":
            image_url = item.get("image_url")
            if isinstance(image_url, dict) and isinstance(image_url.get("url"), str):
                return _embed_data_url(service, image_url["url"])
            if isinstance(image_url, str):
                return _embed_data_url(service, image_url)
        if isinstance(item.get("text"), str):
            return service.embed_text(item["text"])
        if isinstance(item.get("image"), str):
            return _embed_data_url(service, item["image"])
    raise ValueError("OpenAI-compatible embedding input item is not supported")


def _embed_data_url(service: VisualEmbeddingService, data_url: str) -> list[float]:
    header, separator, encoded_payload = data_url.partition(",")
    if not separator or not header.startswith("data:image/"):
        raise ValueError("image input must be a data:image/...;base64,... URL")
    try:
        image_bytes = base64.b64decode(encoded_payload, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("image input data URL must contain valid base64") from exc
    suffix = ".jpg"
    media_type = header.removeprefix("data:").split(";", 1)[0]
    if "/" in media_type:
        extension = media_type.rsplit("/", 1)[1].lower()
        if extension in {"bmp", "gif", "jpeg", "jpg", "png", "webp"}:
            suffix = f".{extension}"
    with tempfile.TemporaryDirectory(prefix="sightindex-openai-embedding-") as temp_dir:
        image_path = Path(temp_dir) / f"query{suffix}"
        image_path.write_bytes(image_bytes)
        return service.embed_image(image_path)


def _embed_image_payload(
    service: VisualEmbeddingService,
    payload: VisualEmbeddingRequest,
) -> list[float]:
    return _embed_image_base64(
        service,
        payload.image_base64 or "",
        payload.image_filename,
        temp_prefix="sightindex-embedding-",
    )


def _embed_image_base64(
    service: VisualEmbeddingService,
    image_base64: str,
    image_filename: str | None,
    *,
    temp_prefix: str,
) -> list[float]:
    suffix = _image_suffix(image_filename)
    with tempfile.TemporaryDirectory(prefix=temp_prefix) as temp_dir:
        image_path = Path(temp_dir) / f"query{suffix}"
        image_path.write_bytes(_decode_image_base64(_normalize_image_payload(image_base64)))
        return service.embed_image(image_path)


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
