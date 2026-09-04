import base64
import binascii

from pydantic import BaseModel, model_validator


class SceneSummaryRequest(BaseModel):
    image_base64: str | None = None
    image_filename: str | None = None
    image_url: str | None = None

    @model_validator(mode="after")
    def validate_single_image_input(self) -> "SceneSummaryRequest":
        has_base64 = bool(self.image_base64)
        has_url = bool(self.image_url)
        if has_base64 == has_url:
            raise ValueError("Provide exactly one of image_base64 or image_url")
        if self.image_base64 is not None:
            try:
                base64.b64decode(self.image_base64, validate=True)
            except (ValueError, binascii.Error) as exc:
                raise ValueError("image_base64 must be valid base64") from exc
        return self


class SceneSummaryResponse(BaseModel):
    labels: list[str]
