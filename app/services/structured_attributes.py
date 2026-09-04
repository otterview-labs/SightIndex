import json
import tempfile
import uuid
from pathlib import Path
from typing import BinaryIO

from sqlalchemy.orm import Session

from app.config.settings import Settings
from app.models.media import PersonCrop
from app.schemas.attributes import AttributeBBox, ObjectType, StructuredAttributeItem
from app.services.observation_index import ObservationIndexService
from app.services.vlm import VLMRuntimeError, VLMStructuredAnalysisService


class StructuredAttributeService:
    def __init__(self, db: Session, settings: Settings) -> None:
        self.db = db
        self.settings = settings
        self.vlm = VLMStructuredAnalysisService(settings)

    def analyze_file(
        self,
        file: BinaryIO,
        filename: str,
        object_type: ObjectType,
        bbox: AttributeBBox | None = None,
    ) -> StructuredAttributeItem:
        suffix = Path(filename).suffix or ".jpg"
        with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
            tmp.write(file.read())
            tmp.flush()
            attributes = self.vlm.analyze_image(
                Path(tmp.name),
                object_type=object_type,
                bbox=bbox.model_dump() if bbox else None,
            )
        return StructuredAttributeItem(
            object_type=object_type,
            bbox=bbox,
            attributes=attributes,
        )

    def analyze_person_crop(
        self,
        crop: PersonCrop,
        *,
        persist: bool = True,
    ) -> dict[str, object]:
        crop_path = self._resolve_data_url(crop.crop_url)
        if crop_path is None or not crop_path.exists():
            raise VLMRuntimeError(f"Person crop image does not exist: {crop.crop_url}")
        attributes = self.vlm.analyze_image(crop_path, object_type="person", bbox=crop.bbox)
        attributes = self._merge_existing_geometry(crop.attributes, attributes)
        if persist:
            crop.attributes = attributes
            self.db.add(crop)
            ObservationIndexService(self.db, self.settings).upsert_crop(crop)
            self.db.commit()
            self.db.refresh(crop)
        return attributes

    @staticmethod
    def _merge_existing_geometry(
        existing: dict[str, object] | None,
        analyzed: dict[str, object],
    ) -> dict[str, object]:
        """Keeps deterministic pose/tone fields that the VLM did not answer.

        Qwen is authoritative for colours and carried objects when confident, while the local
        keypoint pipeline is authoritative for camera-relative stature, facing and measured
        garment length. Replacing the whole JSON loses those complementary facts.
        """

        if not isinstance(existing, dict):
            return {**analyzed, "source": "vlm"}
        merged = dict(analyzed)
        merged["source"] = "vlm"
        for field in ("stature", "facing"):
            if field not in merged and field in existing:
                merged[field] = existing[field]
        old_clothing = existing.get("clothing")
        new_clothing = merged.get("clothing")
        if isinstance(old_clothing, dict) and isinstance(new_clothing, dict):
            clothing = dict(new_clothing)
            for field in ("upper_length", "lower_length"):
                if not clothing.get(field) and old_clothing.get(field):
                    clothing[field] = old_clothing[field]
            merged["clothing"] = clothing
        return merged

    def analyze_unparsed_person_crops(
        self,
        limit: int,
        *,
        force: bool = False,
    ) -> tuple[int, int, list[str]]:
        query = self.db.query(PersonCrop).order_by(PersonCrop.created_at.desc())
        if force:
            crops = query.limit(limit).all()
        else:
            # `cv_tone` is an inexpensive local fallback with deliberately low confidence. It is
            # useful until the VLM runs, but it must still count as pending structured analysis.
            # Filter in Python so this works identically with SQLite and PostgreSQL JSON columns.
            crops = [
                crop
                for crop in query.all()
                if not isinstance(crop.attributes, dict)
                or crop.attributes.get("source") != "vlm"
            ][:limit]
        seen = 0
        updated = 0
        errors: list[str] = []
        for crop in crops:
            seen += 1
            try:
                self.analyze_person_crop(crop, persist=True)
                updated += 1
            except VLMRuntimeError as exc:
                errors.append(f"{crop.id}: {exc}")
        return seen, updated, errors

    def parse_bbox_json(self, value: str | None) -> AttributeBBox | None:
        if not value:
            return None
        try:
            payload = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"bbox_json is invalid: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("bbox_json must be a JSON object")
        return AttributeBBox(**payload)

    def _resolve_data_url(self, url: str) -> Path | None:
        prefix = "/data/"
        if not url.startswith(prefix):
            return None
        return self.settings.data_dir / url.removeprefix(prefix)


def crop_id_text(crop_id: uuid.UUID) -> str:
    return str(crop_id)
