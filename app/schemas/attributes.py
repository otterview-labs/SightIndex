from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ObjectType = Literal["person", "vehicle"]
LabelLanguage = Literal["zh", "en"]


class AttributeBBox(BaseModel):
    box_id: str | None = Field(default=None, description="Optional detector box id.")
    x1: int = Field(description="Left coordinate of the bounding box in pixels.")
    y1: int = Field(description="Top coordinate of the bounding box in pixels.")
    x2: int = Field(description="Right coordinate of the bounding box in pixels.")
    y2: int = Field(description="Bottom coordinate of the bounding box in pixels.")
    score: float | None = Field(default=None, description="Optional detector confidence score.")
    label: str | None = Field(default=None, description="Optional detector label.")


class StructuredAttributeItem(BaseModel):
    object_type: ObjectType
    bbox: AttributeBBox | None = None
    attributes: dict[str, Any]


class StructuredAnalyzeResponse(BaseModel):
    count: int = 0
    items: list[StructuredAttributeItem] = Field(default_factory=list)


class PersonCropAttributeResponse(BaseModel):
    crop_id: str
    attributes: dict[str, Any]


class VLMStructuredAnalysisRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "image_base64": "data:image/jpeg;base64,aW1hZ2U=",
                    "image_filename": "person.jpg",
                    "object_type": "person",
                    "label_language": "zh",
                    "bbox": {
                        "box_id": "det-1",
                        "x1": 10,
                        "y1": 20,
                        "x2": 220,
                        "y2": 520,
                        "score": 0.91,
                        "label": "person",
                    },
                },
                {
                    "image_base64": "aW1hZ2U=",
                    "image_filename": "vehicle.jpg",
                    "object_type": "vehicle",
                    "label_language": "en",
                    "bbox": None,
                },
            ]
        }
    )

    image_base64: str = Field(
        min_length=1,
        description=(
            "Base64 encoded image bytes. A data URL prefix such as "
            "`data:image/jpeg;base64,` is also accepted."
        ),
    )
    image_filename: str | None = Field(
        default=None,
        description=(
            "Optional original filename. The suffix is used to create the temporary image file."
        ),
        examples=["person.jpg"],
    )
    object_type: ObjectType = Field(
        default="person",
        description="Object type to parse. Supported values are `person` and `vehicle`.",
    )
    label_language: LabelLanguage = Field(
        default="zh",
        description=(
            "Language for the convenience `labels` field in the response. Use `zh` for Chinese "
            "labels or `en` for English labels. Raw `attributes` keep their normalized values."
        ),
    )
    bbox: AttributeBBox | None = Field(
        default=None,
        description=(
            "Optional bounding box. When present, the VLM is asked to prioritize this area."
        ),
    )


class VLMStructuredAnalysisResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "object_type": "person",
                    "bbox": {
                        "box_id": "det-1",
                        "x1": 10,
                        "y1": 20,
                        "x2": 220,
                        "y2": 520,
                        "score": 0.91,
                        "label": "person",
                    },
                    "attributes": {
                        "object_type": "person",
                        "appearance": {
                            "hair": "short_hair",
                            "hat": False,
                            "glasses": False,
                            "gender": "unknown",
                            "age_group": "adult",
                        },
                        "clothing": {
                            "upper_color": "black",
                            "lower_color": "gray",
                        },
                        "objects": {
                            "backpack": True,
                            "holding_phone": False,
                            "cigarette": False,
                        },
                        "behavior": {
                            "smoking": False,
                            "looking_at_phone": False,
                            "falling": False,
                            "lying_on_ground": False,
                            "fighting": False,
                            "physical_conflict": False,
                        },
                        "confidence": 0.82,
                        "notes": "",
                    },
                    "labels": {
                        "对象类型": "人员",
                        "上衣颜色": "黑色",
                        "下装颜色": "灰色",
                        "背包": "是",
                        "帽子": "否",
                        "眼镜": "否",
                    },
                    "model": "Qwen3.6-35B-A3B",
                    "provider": "openai_compatible",
                },
                {
                    "object_type": "vehicle",
                    "bbox": None,
                    "attributes": {
                        "object_type": "vehicle",
                        "vehicle_color": "white",
                        "vehicle_type": "suv",
                        "vehicle_brand": None,
                        "plate_color": None,
                        "confidence": 0.76,
                        "notes": "",
                    },
                    "labels": {
                        "object_type": "vehicle",
                        "vehicle_color": "white",
                        "vehicle_type": "suv",
                    },
                    "model": "Qwen3.6-35B-A3B",
                    "provider": "openai_compatible",
                },
            ]
        }
    )

    object_type: ObjectType = Field(description="Parsed object type.")
    bbox: AttributeBBox | None = Field(default=None, description="Echo of the request bbox.")
    attributes: dict[str, Any] = Field(description="Normalized VLM attributes.")
    labels: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Convenience labels formatted in the requested `label_language`. Defaults to Chinese."
        ),
    )
    label_language: LabelLanguage = Field(description="Language used for the `labels` field.")
    model: str = Field(description="Configured VLM model.")
    provider: str = Field(description="Configured VLM provider.")


OPENAPI_VLM_STRUCTURED_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {
        "description": "The image payload is not valid base64.",
        "content": {
            "application/json": {
                "example": {"detail": "image_base64 must be valid base64"}
            }
        },
    },
    401: {
        "description": "The VLM service API key is invalid.",
        "content": {
            "application/json": {"example": {"detail": "Invalid VLM API key"}}
        },
    },
    422: {
        "description": "Request validation failed, for example unsupported object_type.",
    },
    503: {
        "description": "The configured VLM runtime or upstream VLM service failed.",
        "content": {
            "application/json": {
                "example": {"detail": "VLM structured analysis is not configured"}
            }
        },
    },
}
