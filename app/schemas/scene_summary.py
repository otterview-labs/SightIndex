from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SceneSummaryRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "image_base64": "data:image/jpeg;base64,aW1hZ2U=",
                    "image_filename": "traffic.jpg",
                }
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
        description="Optional original filename. The suffix is used for the temp image.",
    )
    context: str | None = Field(
        default=None,
        description="Optional business context hint, for example `traffic violation`.",
    )


class SceneTags(BaseModel):
    target: list[str] = Field(default_factory=list)
    attribute: list[str] = Field(default_factory=list)
    behavior: list[str] = Field(default_factory=list)
    status: list[str] = Field(default_factory=list)
    scene: list[str] = Field(default_factory=list)


class ScenePersonTags(BaseModel):
    attribute: list[str] = Field(default_factory=list)
    behavior: list[str] = Field(default_factory=list)
    status: list[str] = Field(default_factory=list)


class ScenePersonSummary(BaseModel):
    role: str = "unknown"
    tags: ScenePersonTags = Field(default_factory=ScenePersonTags)


class SceneCount(BaseModel):
    persons: int = Field(default=0, ge=0)
    vehicles: int = Field(default=0, ge=0)


class SceneSummaryResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "title": "白色SUV内两名男子,驾驶员未系安全带",
                    "description": (
                        "夜间道路,白色宝马SUV正前方行驶,驾驶员穿白色长袖握方向盘"
                        "未系安全带,副驾驶穿黑色上衣看手机,车灯开启"
                    ),
                    "tags": {
                        "target": ["vehicle", "suv", "bmw"],
                        "attribute": [
                            "white",
                            "long_sleeve",
                            "black_upper",
                            "headlights_on",
                            "blue_plate",
                        ],
                        "behavior": ["driving", "phone_calling"],
                        "status": ["unbelted", "front"],
                        "scene": ["road", "night"],
                    },
                    "persons": [
                        {
                            "role": "driver",
                            "tags": {
                                "attribute": ["white", "long_sleeve"],
                                "behavior": ["driving"],
                                "status": ["unbelted", "front"],
                            },
                        },
                        {
                            "role": "passenger",
                            "tags": {
                                "attribute": ["black_upper"],
                                "behavior": ["phone_calling"],
                                "status": ["front"],
                            },
                        },
                    ],
                    "count": {"persons": 2, "vehicles": 1},
                    "confidence": 0.86,
                }
            ]
        }
    )

    title: str = ""
    description: str = ""
    tags: SceneTags = Field(default_factory=SceneTags)
    persons: list[ScenePersonSummary] = Field(default_factory=list)
    count: SceneCount = Field(default_factory=SceneCount)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    model: str | None = None
    provider: str | None = None
    raw: dict[str, Any] | None = Field(
        default=None,
        description="Original normalized VLM JSON when extra fields need to be inspected.",
    )
