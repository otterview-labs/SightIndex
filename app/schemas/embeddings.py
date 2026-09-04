import base64
import binascii
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class VisualEmbeddingRequest(BaseModel):
    text: str | None = None
    image_base64: str | None = None
    image_filename: str | None = None
    instruction: str | None = None

    @model_validator(mode="after")
    def validate_single_input(self) -> "VisualEmbeddingRequest":
        has_text = bool(self.text)
        has_image = bool(self.image_base64)
        if has_text == has_image:
            raise ValueError("Provide exactly one of text or image_base64")
        if self.image_base64 is not None:
            try:
                base64.b64decode(self.image_base64, validate=True)
            except (ValueError, binascii.Error) as exc:
                raise ValueError("image_base64 must be valid base64") from exc
        return self


class VisualEmbeddingResponse(BaseModel):
    embedding: list[float] = Field(min_length=1)
    dim: int = Field(ge=1)
    model: str
    provider: str


class OpenAIEmbeddingRequest(BaseModel):
    model: str | None = None
    input: str | list[Any]
    dimensions: int | None = Field(default=None, ge=1)
    encoding_format: str | None = None
    user: str | None = None


class OpenAIEmbeddingData(BaseModel):
    object: str = "embedding"
    index: int
    embedding: list[float] = Field(min_length=1)


class OpenAIEmbeddingUsage(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0


class OpenAIEmbeddingResponse(BaseModel):
    object: str = "list"
    data: list[OpenAIEmbeddingData]
    model: str
    usage: OpenAIEmbeddingUsage = Field(default_factory=OpenAIEmbeddingUsage)


class ImageVectorRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "text": "黑衣背包的人",
                    "instruction": "Retrieve images that match the user query.",
                },
                {
                    "image_base64": "aW1hZ2U=",
                    "image_filename": "crop.jpg",
                },
                {
                    "image_base64": "data:image/jpeg;base64,aW1hZ2U=",
                    "image_filename": "person.jpg",
                },
            ]
        }
    )

    text: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Text query to embed into the same visual retrieval vector space. Provide exactly "
            "one of `text` or `image_base64`."
        ),
        examples=["黑衣背包的人"],
    )
    image_base64: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Base64 encoded image bytes. A data URL prefix such as "
            "`data:image/jpeg;base64,` is also accepted. Provide exactly one of `text` or "
            "`image_base64`."
        ),
    )
    image_filename: str | None = Field(
        default=None,
        description=(
            "Optional original filename. The suffix is used to create the temporary image file."
        ),
        examples=["crop.jpg"],
    )
    instruction: str | None = Field(
        default=None,
        description=(
            "Optional embedding instruction for text queries. When omitted, "
            "`VISUAL_EMBEDDING_INSTRUCTION` is used."
        ),
        examples=["Retrieve images that match the user query."],
    )

    @model_validator(mode="after")
    def validate_single_input(self) -> "ImageVectorRequest":
        has_text = bool(self.text)
        has_image = bool(self.image_base64)
        if has_text == has_image:
            raise ValueError("Provide exactly one of text or image_base64")
        if self.image_base64 is not None:
            try:
                base64.b64decode(_strip_data_url_prefix(self.image_base64), validate=True)
            except (ValueError, binascii.Error) as exc:
                raise ValueError("image_base64 must be valid base64") from exc
        return self


class ImageVectorResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "embedding": [0.0123, -0.0456, 0.0789],
                    "dim": 2048,
                    "model": "Qwen3-VL-Embedding-2B",
                    "provider": "qwen3_vl_http",
                    "input_type": "text",
                    "image_filename": None,
                },
                {
                    "embedding": [0.0123, -0.0456, 0.0789],
                    "dim": 2048,
                    "model": "Qwen3-VL-Embedding-2B",
                    "provider": "qwen3_vl_http",
                    "input_type": "image",
                    "image_filename": "crop.jpg",
                },
            ]
        }
    )

    embedding: list[float] = Field(
        min_length=1,
        description="Text or image embedding vector. Its length should match `dim`.",
    )
    dim: int = Field(ge=1, description="Embedding dimension, for example 2048 for Qwen3-VL.")
    model: str = Field(description="Configured visual embedding model name or path.")
    provider: str = Field(description="Configured visual embedding provider.")
    input_type: Literal["text", "image"] = Field(description="Input modality used for embedding.")
    image_filename: str | None = Field(default=None, description="Echo of the request filename.")


class VisualRerankRequest(BaseModel):
    query: str = Field(min_length=1)
    image_base64: str | None = None
    image: str | None = None
    image_filename: str | None = None
    attributes: dict[str, object] | None = None

    @model_validator(mode="after")
    def validate_image_input(self) -> "VisualRerankRequest":
        image_payload = self.image_base64 or self.image
        if not image_payload:
            raise ValueError("Provide image_base64 or image")
        if self.image_base64 is not None:
            try:
                base64.b64decode(self.image_base64, validate=True)
            except (ValueError, binascii.Error) as exc:
                raise ValueError("image_base64 must be valid base64") from exc
        return self


class VisualRerankResponse(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    matched: bool = False
    reason: str | None = None
    model: str
    provider: str


OPENAPI_IMAGE_VECTOR_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {
        "description": "The request did not provide exactly one input or image_base64 is invalid.",
        "content": {
            "application/json": {
                "examples": {
                    "invalid_base64": {
                        "summary": "Invalid base64 image",
                        "value": {"detail": "image_base64 must be valid base64"},
                    },
                    "invalid_input_count": {
                        "summary": "Text and image missing or both present",
                        "value": {"detail": "Provide exactly one of text or image_base64"},
                    },
                }
            }
        },
    },
    401: {
        "description": "The visual embedding service API key is invalid.",
        "content": {
            "application/json": {
                "example": {"detail": "Invalid visual embedding service API key"}
            }
        },
    },
    503: {
        "description": "The configured embedding runtime or remote embedding service failed.",
        "content": {
            "application/json": {
                "example": {
                    "detail": (
                        "Qwen3-VL HTTP embedding service is busy; "
                        "waited 2.0s for a local slot"
                    )
                }
            }
        },
    },
}


def _strip_data_url_prefix(value: str) -> str:
    if value.startswith("data:"):
        _, _, value = value.partition(",")
    return value
