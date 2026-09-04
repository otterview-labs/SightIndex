import logging
import math
import tempfile
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any

from fastapi import UploadFile
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config.settings import Settings
from app.face_algorithms import InsightFaceCudaRecognizer
from app.models.events import RecognitionEvent
from app.models.media import Image, PersonCrop
from app.models.persons import Person
from app.models.vectors import FaceEmbedding
from app.schemas.persons import FaceMatchItem, FaceRecognitionResponse, FaceSearchResponse
from app.services.observation_index import ObservationIndexService
from app.services.storage import StorageService
from app.services.time_utils import local_now
from app.services.vector_index import MilvusVectorIndex, VectorIndexError

try:  # numpy keeps the library scan vectorised; the pure-Python scan stays as a fallback
    import numpy as np
except ImportError:  # pragma: no cover - only hit when numpy is not installed
    np = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FaceCandidate:
    embedding: list[float]
    bbox: dict[str, float]
    quality_score: float
    model: str


@dataclass(frozen=True)
class FaceMatch:
    person: Person
    face_embedding: FaceEmbedding
    similarity: float


@dataclass(frozen=True)
class FaceCropComparison:
    similarity: float
    query_quality: float
    candidate_quality: float


@dataclass(frozen=True)
class FaceRuntimeStatus:
    ready: bool
    provider: str
    model: str
    device: str
    model_dir: str
    error: str | None = None


def face_runtime_status(settings: Settings) -> FaceRuntimeStatus:
    """Report whether configured face priority can really run, without doing inference."""

    provider = settings.face_embedding_provider.strip().lower()
    if provider != "insightface":
        return FaceRuntimeStatus(
            ready=False,
            provider=provider,
            model=provider or "unknown",
            device=settings.face_embedding_device or "unknown",
            model_dir="",
            error=f"Unsupported production face provider: {provider or 'unset'}",
        )
    recognizer = InsightFaceCudaRecognizer(
        model_name=settings.face_insightface_model,
        det_size=settings.face_insightface_det_size,
        device=settings.face_embedding_device,
        root=settings.face_insightface_root,
        allow_download=settings.face_insightface_allow_download,
    )
    info = recognizer.info()
    model_dir = recognizer.model_dir()
    required = (model_dir / "det_10g.onnx", model_dir / "w600k_r50.onnx")
    missing = [path.name for path in required if not path.is_file()]
    provider_error = None
    if info.device.startswith("cuda") and "CUDAExecutionProvider" not in info.available_providers:
        provider_error = "CUDAExecutionProvider is unavailable"
    elif info.device == "cpu" and "CPUExecutionProvider" not in info.available_providers:
        provider_error = "CPUExecutionProvider is unavailable"
    errors = []
    if missing:
        errors.append(f"missing model files: {', '.join(missing)}")
    if provider_error:
        errors.append(provider_error)
    return FaceRuntimeStatus(
        ready=not errors,
        provider=provider,
        model=f"insightface-{settings.face_insightface_model}",
        device=info.device,
        model_dir=str(model_dir),
        error="; ".join(errors) or None,
    )


@dataclass(frozen=True)
class _FaceLibrary:
    """Enrolled face vectors bucketed by dimension, so a query only meets same-dim vectors."""

    signature: tuple[int, int, str]
    loaded_at: float
    buckets: dict[int, tuple[list[uuid.UUID], Any]]


_FACE_LIBRARY_LOCK = threading.Lock()
_FACE_LIBRARY: _FaceLibrary | None = None


def invalidate_face_library_cache() -> None:
    """Drop the cached library matrix after the face table changes in this process."""

    global _FACE_LIBRARY
    with _FACE_LIBRARY_LOCK:
        _FACE_LIBRARY = None


class FaceRecognitionService:
    def __init__(self, db: Session, settings: Settings) -> None:
        self.db = db
        self.settings = settings
        self.storage = StorageService(settings)

    def enroll_person_face(self, person: Person, file: UploadFile) -> FaceEmbedding:
        image_url = self.storage.save_upload(file)
        image = Image(image_url=image_url, source_type="face_enrollment")
        self.db.add(image)
        self.db.flush()

        candidate = self._best_candidate(image_url, allow_fallback=True)
        if candidate is None:
            raise ValueError("No readable face image found")

        face = FaceEmbedding(
            person_id=person.id,
            image_id=image.id,
            face_bbox=candidate.bbox,
            embedding=candidate.embedding,
            face_model=candidate.model,
            quality_score=candidate.quality_score,
        )
        if not person.avatar_url:
            person.avatar_url = image_url
            self.db.add(person)
        self.db.add(face)
        self.db.commit()
        self.db.refresh(face)
        self._index_face_vector(face)
        invalidate_face_library_cache()
        return face

    def enroll_person_face_from_crop(self, person: Person, crop: PersonCrop) -> FaceEmbedding:
        candidate = self._best_candidate(crop.crop_url, allow_fallback=False)
        if candidate is None:
            raise ValueError("No readable face found in crop")
        image = self.db.get(Image, crop.image_id)
        if image is None:
            raise ValueError("Crop source image not found")

        face = FaceEmbedding(
            person_id=person.id,
            image_id=crop.image_id,
            crop_id=crop.id,
            face_bbox=candidate.bbox,
            embedding=candidate.embedding,
            face_model=candidate.model,
            quality_score=candidate.quality_score,
        )
        if not person.avatar_url:
            person.avatar_url = crop.crop_url
            self.db.add(person)
        crop.person_id = person.id
        self.db.add(crop)
        self.db.add(face)
        self.db.flush()

        manual_match = FaceMatch(person=person, face_embedding=face, similarity=1.0)
        existing_event = self.db.scalar(
            select(RecognitionEvent)
            .where(RecognitionEvent.crop_id == crop.id)
            .order_by(RecognitionEvent.created_at.desc())
            .limit(1)
        )
        if existing_event is None:
            self._create_event(
                image=image,
                crop=crop,
                candidate=candidate,
                match=manual_match,
                result_type="known",
            )
        else:
            self._update_event(
                event=existing_event,
                image=image,
                crop=crop,
                candidate=candidate,
                match=manual_match,
                result_type="known",
            )
        ObservationIndexService(self.db, self.settings).upsert_crop(crop)
        self.db.commit()
        self.db.refresh(face)
        self._index_face_vector(face)
        invalidate_face_library_cache()
        return face

    def list_person_faces(self, person_id: uuid.UUID, limit: int = 50) -> list[FaceEmbedding]:
        stmt = (
            select(FaceEmbedding)
            .where(FaceEmbedding.person_id == person_id)
            .order_by(FaceEmbedding.created_at.desc())
            .limit(limit)
        )
        return list(self.db.scalars(stmt))

    def compare_person_crops(
        self,
        query: PersonCrop,
        candidates: list[PersonCrop],
        *,
        min_quality: float,
    ) -> dict[uuid.UUID, FaceCropComparison]:
        """Compare reliable faces in a small ReID shortlist without persisting query vectors.

        Missing and low-quality faces are deliberately absent from the result. ReID can then
        degrade to body and attribute evidence instead of treating "no face" as a mismatch.
        """

        query_path = self._resolve_data_url(query.crop_url)
        if query_path is None or not query_path.exists():
            return {}
        return self.compare_image_to_crops(
            query_path,
            candidates,
            min_quality=min_quality,
        )

    def compare_person_crop_gallery(
        self,
        queries: list[PersonCrop],
        candidates: list[PersonCrop],
        *,
        min_quality: float,
    ) -> dict[uuid.UUID, FaceCropComparison]:
        """Use the clearest measurable face in a query tracklet.

        Doorway crops frequently alternate between frontal, profile and back views. Treating the
        originally clicked frame as the only possible face made the advertised face priority
        depend on luck; the body-verified tracklet is a safe place to choose a better view.
        """

        best: FaceCandidate | None = None
        for crop in queries:
            candidate = self._best_candidate(crop.crop_url, allow_fallback=False)
            if candidate is None or candidate.quality_score < min_quality:
                continue
            if best is None or candidate.quality_score > best.quality_score:
                best = candidate
        if best is None:
            return {}
        return self._compare_face_to_crops(best, candidates, min_quality=min_quality)

    def compare_image_to_crops(
        self,
        query_path: Path,
        candidates: list[PersonCrop],
        *,
        min_quality: float,
    ) -> dict[uuid.UUID, FaceCropComparison]:
        query_face = self._best_candidate_path(query_path, allow_fallback=False)
        if query_face is None or query_face.quality_score < min_quality:
            return {}
        return self._compare_face_to_crops(query_face, candidates, min_quality=min_quality)

    def _compare_face_to_crops(
        self,
        query_face: FaceCandidate,
        candidates: list[PersonCrop],
        *,
        min_quality: float,
    ) -> dict[uuid.UUID, FaceCropComparison]:
        compared: dict[uuid.UUID, FaceCropComparison] = {}
        for crop in candidates:
            candidate_face = self._best_candidate(crop.crop_url, allow_fallback=False)
            if candidate_face is None or candidate_face.quality_score < min_quality:
                continue
            if len(candidate_face.embedding) != len(query_face.embedding):
                continue
            compared[crop.id] = FaceCropComparison(
                similarity=self._cosine_similarity(
                    query_face.embedding,
                    candidate_face.embedding,
                ),
                query_quality=query_face.quality_score,
                candidate_quality=candidate_face.quality_score,
            )
        return compared

    def recognize_upload(
        self,
        file: UploadFile,
        top_k: int = 5,
        threshold: float | None = None,
    ) -> FaceRecognitionResponse:
        image_url = self.storage.save_upload(file)
        image = Image(image_url=image_url, source_type="face_query")
        self.db.add(image)
        self.db.flush()

        candidate = self._best_candidate(image_url, allow_fallback=True)
        if candidate is None:
            event = self._create_event(
                image=image,
                crop=None,
                candidate=None,
                match=None,
                result_type="no_face",
            )
            self.db.commit()
            self.db.refresh(event)
            return FaceRecognitionResponse(
                result_type="no_face",
                threshold=self._threshold(threshold),
                image_id=image.id,
                event_id=event.id,
                face_bbox=None,
                matches=[],
            )

        matches = self._search_matches(candidate.embedding, top_k=top_k)
        best = self._best_known_match(matches, threshold)
        event = self._create_event(
            image=image,
            crop=None,
            candidate=candidate,
            match=best,
            result_type="known" if best else "unknown",
        )
        self.db.commit()
        self.db.refresh(event)
        return self._recognition_response(
            image=image,
            crop=None,
            candidate=candidate,
            event=event,
            match=best,
            matches=matches,
            threshold=threshold,
        )

    def search_upload(
        self,
        file: UploadFile,
        top_k: int = 5,
        min_similarity: float | None = None,
    ) -> FaceSearchResponse:
        image_url = self.storage.save_upload(file)
        image = Image(image_url=image_url, source_type="face_search")
        self.db.add(image)
        self.db.flush()

        candidate = self._best_candidate(image_url, allow_fallback=True)
        if candidate is None:
            self.db.commit()
            return FaceSearchResponse(image_id=image.id, face_bbox=None, matches=[])

        matches = self._search_matches(
            candidate.embedding,
            top_k=top_k,
            min_similarity=min_similarity,
        )
        self.db.commit()
        return FaceSearchResponse(
            image_id=image.id,
            face_bbox=candidate.bbox,
            matches=[self._match_item(match) for match in matches],
        )

    def search_image(
        self,
        image: Image,
        top_k: int = 5,
        min_similarity: float | None = None,
        allow_fallback: bool = True,
    ) -> FaceSearchResponse:
        candidate = self._best_candidate(image.image_url, allow_fallback=allow_fallback)
        if candidate is None:
            return FaceSearchResponse(image_id=image.id, face_bbox=None, matches=[])

        matches = self._search_matches(
            candidate.embedding,
            top_k=top_k,
            min_similarity=min_similarity,
        )
        return FaceSearchResponse(
            image_id=image.id,
            face_bbox=candidate.bbox,
            matches=[self._match_item(match) for match in matches],
        )

    def recognize_crop(
        self,
        crop: PersonCrop,
        image: Image,
        existing_event: RecognitionEvent | None = None,
        require_ingest_enabled: bool = True,
    ) -> RecognitionEvent | None:
        if require_ingest_enabled and not self.settings.face_recognition_on_ingest:
            return None
        if not self.has_known_faces():
            return None
        candidate = self._best_candidate(
            crop.crop_url,
            allow_fallback=self.settings.face_fallback_to_full_image,
        )
        if candidate is None:
            if existing_event is not None:
                old_person_id = existing_event.person_id
                event = self._update_event(
                    event=existing_event,
                    image=image,
                    crop=crop,
                    candidate=None,
                    match=None,
                    result_type="no_face",
                )
                if old_person_id is not None and crop.person_id == old_person_id:
                    crop.person_id = None
                    self.db.add(crop)
                ObservationIndexService(self.db, self.settings).upsert_crop(crop)
                self.db.commit()
                self.db.refresh(event)
                return event
            if not require_ingest_enabled:
                event = self._create_event(
                    image=image,
                    crop=crop,
                    candidate=None,
                    match=None,
                    result_type="no_face",
                )
                ObservationIndexService(self.db, self.settings).upsert_crop(crop)
                self.db.commit()
                self.db.refresh(event)
                return event
            return None

        matches = self._search_matches(candidate.embedding, top_k=1)
        best = self._best_known_match(matches, None)
        result_type = "known" if best else "unknown"
        old_person_id = existing_event.person_id if existing_event is not None else None
        if existing_event is None:
            event = self._create_event(
                image=image,
                crop=crop,
                candidate=candidate,
                match=best,
                result_type=result_type,
            )
        else:
            event = self._update_event(
                event=existing_event,
                image=image,
                crop=crop,
                candidate=candidate,
                match=best,
                result_type=result_type,
            )
        if best is not None:
            crop.person_id = best.person.id
            self.db.add(crop)
        elif old_person_id is not None and crop.person_id == old_person_id:
            crop.person_id = None
            self.db.add(crop)
        ObservationIndexService(self.db, self.settings).upsert_crop(crop)
        self.db.commit()
        self.db.refresh(event)
        return event

    def rebuild_crop_recognition(self, limit: int = 500, force: bool = False) -> dict[str, object]:
        if not self.has_known_faces():
            return {
                "requested": limit,
                "seen": 0,
                "skipped": 0,
                "events_created": 0,
                "events_updated": 0,
                "matched": 0,
                "errors": ["face library is empty"],
            }

        stmt = (
            select(PersonCrop, Image)
            .join(Image, PersonCrop.image_id == Image.id)
            .order_by(PersonCrop.created_at.desc())
            .limit(limit)
        )
        seen = 0
        skipped = 0
        events_created = 0
        events_updated = 0
        matched = 0
        errors: list[str] = []
        for crop, image in list(self.db.execute(stmt)):
            seen += 1
            existing_event = self.db.scalar(
                select(RecognitionEvent)
                .where(RecognitionEvent.crop_id == crop.id)
                .order_by(RecognitionEvent.created_at.desc())
                .limit(1)
            )
            if not force and existing_event is not None and existing_event.person_id is not None:
                skipped += 1
                continue
            try:
                event = self.recognize_crop(
                    crop,
                    image,
                    existing_event=existing_event,
                    require_ingest_enabled=False,
                )
            except Exception as exc:
                errors.append(str(exc))
                continue
            if event is not None:
                if existing_event is None:
                    events_created += 1
                else:
                    events_updated += 1
                if event.person_id is not None:
                    matched += 1
            else:
                skipped += 1
        return {
            "requested": limit,
            "seen": seen,
            "skipped": skipped,
            "events_created": events_created,
            "events_updated": events_updated,
            "matched": matched,
            "errors": errors,
        }

    def rebuild_library_embeddings(
        self,
        limit: int = 500,
        allow_fallback: bool = False,
    ) -> dict[str, object]:
        stmt = (
            select(FaceEmbedding, Image, PersonCrop)
            .join(Image, FaceEmbedding.image_id == Image.id, isouter=True)
            .join(PersonCrop, FaceEmbedding.crop_id == PersonCrop.id, isouter=True)
            .where(FaceEmbedding.person_id.is_not(None))
            .order_by(FaceEmbedding.created_at.desc())
            .limit(limit)
        )
        seen = 0
        updated = 0
        skipped = 0
        errors: list[str] = []
        for face, image, crop in list(self.db.execute(stmt)):
            seen += 1
            source_url = crop.crop_url if crop is not None else image.image_url if image else None
            if not source_url:
                skipped += 1
                continue
            try:
                candidate = self._best_candidate(source_url, allow_fallback=allow_fallback)
            except Exception as exc:
                errors.append(f"{face.id}: {exc}")
                skipped += 1
                continue
            if candidate is None:
                skipped += 1
                continue
            face.embedding = candidate.embedding
            face.face_bbox = candidate.bbox
            face.face_model = candidate.model
            face.quality_score = candidate.quality_score
            self.db.add(face)
            updated += 1
        self.db.commit()
        if updated:
            self._rebuild_milvus_face_index(limit=limit)
        if updated:
            # Vectors change in place here, so the count/created_at fingerprint cannot see it.
            invalidate_face_library_cache()
        return {
            "requested": limit,
            "seen": seen,
            "updated": updated,
            "skipped": skipped,
            "errors": errors,
        }

    def diagnose_recent_crops(self, limit: int = 20) -> list[dict[str, object]]:
        stmt = (
            select(PersonCrop, Image)
            .join(Image, PersonCrop.image_id == Image.id)
            .order_by(PersonCrop.created_at.desc())
            .limit(limit)
        )
        items: list[dict[str, object]] = []
        threshold = self._threshold(None)
        for crop, image in list(self.db.execute(stmt)):
            existing_event = self.db.scalar(
                select(RecognitionEvent)
                .where(RecognitionEvent.crop_id == crop.id)
                .order_by(RecognitionEvent.created_at.desc())
                .limit(1)
            )
            base: dict[str, object] = {
                "crop_id": crop.id,
                "image_id": crop.image_id,
                "crop_url": crop.crop_url,
                "image_url": image.thumbnail_url or image.image_url,
                "captured_at": crop.captured_at or crop.created_at,
                "existing_result_type": existing_event.result_type if existing_event else None,
                "existing_person_id": existing_event.person_id if existing_event else None,
                "existing_similarity": (
                    float(existing_event.similarity)
                    if existing_event and existing_event.similarity is not None
                    else None
                ),
                "detection_score": None,
                "face_bbox": None,
                "top_person_id": None,
                "top_person_name": None,
                "top_similarity": None,
                "threshold": threshold,
            }
            try:
                candidate = self._best_candidate(
                    crop.crop_url,
                    allow_fallback=self.settings.face_fallback_to_full_image,
                )
            except Exception as exc:
                items.append(
                    {
                        **base,
                        "verdict": "error",
                        "reason": str(exc),
                        "can_enroll": False,
                    }
                )
                continue

            if candidate is None:
                items.append(
                    {
                        **base,
                        "verdict": "no_face",
                        "reason": "未检测到可用人脸，不能补入人脸库。",
                        "can_enroll": False,
                    }
                )
                continue

            matches = self._search_matches(candidate.embedding, top_k=1)
            best = matches[0] if matches else None
            similarity = best.similarity if best else None
            verdict = "known" if similarity is not None and similarity >= threshold else "unknown"
            reason = self._diagnostic_reason(candidate, similarity, threshold)
            items.append(
                {
                    **base,
                    "detection_score": candidate.quality_score,
                    "face_bbox": candidate.bbox,
                    "top_person_id": best.person.id if best else None,
                    "top_person_name": best.person.name if best else None,
                    "top_similarity": similarity,
                    "verdict": verdict,
                    "reason": reason,
                    "can_enroll": candidate.quality_score >= 0.55,
                }
            )
        return items

    def has_known_faces(self) -> bool:
        stmt = (
            select(FaceEmbedding.id)
            .where(FaceEmbedding.person_id.is_not(None))
            .where(FaceEmbedding.embedding.is_not(None))
            .limit(1)
        )
        return self.db.scalar(stmt) is not None

    def _best_candidate(self, data_url: str, allow_fallback: bool) -> FaceCandidate | None:
        image_path = self._resolve_data_url(data_url)
        if image_path is None or not image_path.exists():
            return None
        return self._best_candidate_path(image_path, allow_fallback=allow_fallback)

    def _best_candidate_path(
        self,
        image_path: Path,
        *,
        allow_fallback: bool,
    ) -> FaceCandidate | None:
        candidates = self._extract_candidates(image_path, allow_fallback=allow_fallback)
        if not candidates:
            return None
        return max(candidates, key=lambda item: item.quality_score)

    def _extract_candidates(self, image_path: Path, allow_fallback: bool) -> list[FaceCandidate]:
        provider = self.settings.face_embedding_provider.lower()
        if provider == "insightface":
            return self._extract_with_insightface(image_path, allow_fallback=allow_fallback)
        if provider in {"opencv", "none", ""}:
            return self._extract_with_opencv(image_path, allow_fallback=allow_fallback)
        raise ValueError(f"Unsupported face embedding provider: {provider}")

    def _extract_with_insightface(
        self,
        image_path: Path,
        allow_fallback: bool,
    ) -> list[FaceCandidate]:
        try:
            import cv2  # type: ignore[import-not-found]

            recognizer = InsightFaceCudaRecognizer(
                model_name=self.settings.face_insightface_model,
                det_size=self.settings.face_insightface_det_size,
                device=self.settings.face_embedding_device,
                root=self.settings.face_insightface_root,
                allow_download=self.settings.face_insightface_allow_download,
            )
            candidates = self._service_candidates(recognizer.extract(image_path))
            upscaled_path, scale = self._write_upscaled_face_candidate_image(cv2, image_path)
            if upscaled_path is not None:
                try:
                    upscaled_candidates = self._service_candidates(
                        recognizer.extract(upscaled_path),
                        bbox_scale=scale,
                        model_suffix=f"-upscaled-{scale:.2f}x",
                    )
                finally:
                    upscaled_path.unlink(missing_ok=True)
                if upscaled_candidates:
                    candidates = self._best_candidates_by_quality(
                        candidates,
                        upscaled_candidates,
                    )
        except Exception as exc:
            if allow_fallback:
                logger.warning(
                    "InsightFace inference failed on %s, degrading to the OpenCV extractor: %s",
                    image_path.name,
                    exc,
                    exc_info=True,
                )
                return self._extract_with_opencv(image_path, allow_fallback=True)
            raise ValueError(f"InsightFace inference failed: {exc}") from exc

        if candidates or not allow_fallback:
            return candidates
        # The OpenCV extractor produces a downscaled-grayscale vector, not a face embedding.
        # Recognition still "works" against it, so say loudly that quality just dropped.
        logger.warning(
            "InsightFace found no face in %s at det_size=%s; degrading to the OpenCV extractor, "
            "whose embeddings are far weaker. Try a smaller FACE_INSIGHTFACE_DET_SIZE.",
            image_path.name,
            self.settings.face_insightface_det_size,
        )
        return self._extract_with_opencv(image_path, allow_fallback=True)

    def _best_candidates_by_quality(
        self,
        original: list[FaceCandidate],
        upscaled: list[FaceCandidate],
    ) -> list[FaceCandidate]:
        if not original:
            return upscaled
        original_best = max(original, key=lambda item: item.quality_score)
        upscaled_best = max(upscaled, key=lambda item: item.quality_score)
        if upscaled_best.quality_score >= original_best.quality_score:
            return upscaled
        return original

    def _service_candidates(
        self,
        algorithm_candidates: list[Any],
        bbox_scale: float = 1.0,
        model_suffix: str = "",
    ) -> list[FaceCandidate]:
        candidates: list[FaceCandidate] = []
        scale = bbox_scale if bbox_scale > 0 else 1.0
        for candidate in algorithm_candidates:
            bbox = candidate.bbox
            candidates.append(
                FaceCandidate(
                    embedding=candidate.embedding,
                    bbox={
                        "x": max(0.0, float(bbox.get("x", 0.0)) / scale),
                        "y": max(0.0, float(bbox.get("y", 0.0)) / scale),
                        "width": max(1.0, float(bbox.get("width", 1.0)) / scale),
                        "height": max(1.0, float(bbox.get("height", 1.0)) / scale),
                    },
                    quality_score=candidate.quality_score,
                    model=f"{candidate.model}{model_suffix}",
                )
            )
        return candidates

    def _write_upscaled_face_candidate_image(
        self,
        cv2: Any,
        image_path: Path,
    ) -> tuple[Path | None, float]:
        image = cv2.imread(str(image_path))
        if image is None:
            return None, 1.0

        height, width = image.shape[:2]
        scale = self._face_candidate_upscale_factor(width=width, height=height)
        if scale <= 1.01:
            return None, 1.0

        resized = cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_CUBIC,
        )
        target = Path(tempfile.gettempdir()) / f"sightindex-face-upscale-{uuid.uuid4().hex}.jpg"
        ok = cv2.imwrite(str(target), resized, [int(cv2.IMWRITE_JPEG_QUALITY), 97])
        if not ok:
            return None, 1.0
        return target, scale

    def _face_candidate_upscale_factor(self, *, width: int, height: int) -> float:
        if width <= 0 or height <= 0:
            return 1.0

        scale = 1.0
        min_width = self.settings.face_candidate_upscale_min_width
        min_height = self.settings.face_candidate_upscale_min_height
        if min_width and width < min_width:
            scale = max(scale, min_width / width)
        if min_height and height < min_height:
            scale = max(scale, min_height / height)
        return min(scale, self.settings.face_candidate_upscale_max_factor)

    def _extract_with_opencv(
        self,
        image_path: Path,
        allow_fallback: bool,
    ) -> list[FaceCandidate]:
        try:
            import cv2  # type: ignore[import-not-found]
        except Exception as exc:
            raise ValueError(f"OpenCV is not installed: {exc}") from exc

        image = cv2.imread(str(image_path))
        if image is None:
            return []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]
        boxes: list[tuple[int, int, int, int]] = []

        cascade_path = ""
        if hasattr(cv2, "data"):
            cascade_path = str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
        if cascade_path and Path(cascade_path).exists():
            cascade = cv2.CascadeClassifier(cascade_path)
            if not cascade.empty():
                raw_boxes = cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=4,
                    minSize=(24, 24),
                )
                boxes = [tuple(map(int, box)) for box in raw_boxes]

        if not boxes and allow_fallback:
            boxes = [(0, 0, width, height)]

        candidates: list[FaceCandidate] = []
        frame_area = max(1, width * height)
        for x, y, box_width, box_height in boxes:
            x1 = max(0, min(x, width - 1))
            y1 = max(0, min(y, height - 1))
            x2 = max(x1 + 1, min(x1 + box_width, width))
            y2 = max(y1 + 1, min(y1 + box_height, height))
            face = gray[y1:y2, x1:x2]
            embedding = self._opencv_embedding(cv2, face)
            if not embedding:
                continue
            quality = min(1.0, max(0.01, ((x2 - x1) * (y2 - y1)) / frame_area))
            candidates.append(
                FaceCandidate(
                    embedding=embedding,
                    bbox={
                        "x": float(x1),
                        "y": float(y1),
                        "width": float(x2 - x1),
                        "height": float(y2 - y1),
                    },
                    quality_score=quality,
                    model="opencv-gray32",
                )
            )
        return candidates

    def _opencv_embedding(self, cv2: Any, face: Any) -> list[float]:
        resized = cv2.resize(face, (32, 32), interpolation=cv2.INTER_AREA)
        vector = [float(value) / 255.0 for value in resized.flatten()]
        return self._normalize(vector)

    def _search_matches(
        self,
        query_vector: list[float],
        top_k: int,
        min_similarity: float | None = None,
    ) -> list[FaceMatch]:
        milvus_matches = self._search_milvus_matches(query_vector, top_k, min_similarity)
        if milvus_matches is not None:
            return milvus_matches
        library = self._face_library()
        if library is None:
            return self._scan_matches(query_vector, top_k, min_similarity)
        bucket = library.buckets.get(len(query_vector))
        if bucket is None:
            return []
        face_ids, matrix = bucket
        scores = matrix @ np.asarray(query_vector, dtype="float64")
        if min_similarity is None:
            kept = np.arange(scores.shape[0])
        else:
            kept = np.flatnonzero(scores >= min_similarity)
        if kept.size == 0:
            return []
        # A stable sort keeps the created_at desc order of the snapshot for equal scores,
        # which is what the previous list.sort() did.
        ranked = kept[np.argsort(-scores[kept], kind="stable")][:top_k]
        return self._resolve_matches(
            [(face_ids[int(index)], float(scores[int(index)])) for index in ranked]
        )

    def _scan_matches(
        self,
        query_vector: list[float],
        top_k: int,
        min_similarity: float | None,
    ) -> list[FaceMatch]:
        stmt = (
            select(FaceEmbedding, Person)
            .join(Person, FaceEmbedding.person_id == Person.id)
            .where(FaceEmbedding.embedding.is_not(None))
            .order_by(FaceEmbedding.created_at.desc())
            .limit(self.settings.face_max_library_scan)
        )
        matches: list[FaceMatch] = []
        for face, person in self.db.execute(stmt):
            if not face.embedding:
                continue
            if len(face.embedding) != len(query_vector):
                continue
            similarity = self._cosine_similarity(query_vector, face.embedding)
            if min_similarity is not None and similarity < min_similarity:
                continue
            matches.append(FaceMatch(person=person, face_embedding=face, similarity=similarity))
        matches.sort(key=lambda item: item.similarity, reverse=True)
        return matches[:top_k]

    def _search_milvus_matches(
        self,
        query_vector: list[float],
        top_k: int,
        min_similarity: float | None,
    ) -> list[FaceMatch] | None:
        """Search enrolled face vectors in Milvus, with SQL fallback on any unavailable path."""
        if (
            not self.settings.milvus_enabled
            or self.settings.milvus_metric_type.strip().upper() != "COSINE"
            or len(query_vector) != self.settings.face_embedding_dim
        ):
            return None
        try:
            hits = MilvusVectorIndex(self.settings).search_vector(
                "face_embedding", query_vector, top_k
            )
        except (VectorIndexError, ValueError):
            logger.warning(
                "Milvus face search unavailable; falling back to SQL scan", exc_info=True
            )
            return None
        scored = [
            (hit.object_id, hit.score)
            for hit in hits
            if min_similarity is None or hit.score >= min_similarity
        ]
        return self._resolve_matches(scored)

    def _index_face_vector(self, face: FaceEmbedding) -> None:
        if (
            not self.settings.milvus_enabled
            or self.settings.milvus_metric_type.strip().upper() != "COSINE"
            or not face.embedding
            or len(face.embedding) != self.settings.face_embedding_dim
        ):
            return
        try:
            index = MilvusVectorIndex(self.settings)
            index.upsert_vector("face_embedding", face.id, list(face.embedding))
        except VectorIndexError:
            logger.warning(
                "Could not index face %s in Milvus; SQL remains authoritative",
                face.id,
                exc_info=True,
            )

    def _rebuild_milvus_face_index(self, limit: int) -> None:
        if (
            not self.settings.milvus_enabled
            or self.settings.milvus_metric_type.strip().upper() != "COSINE"
        ):
            return
        indexed = 0
        index = MilvusVectorIndex(self.settings)
        stmt = (
            select(FaceEmbedding)
            .where(FaceEmbedding.person_id.is_not(None))
            .where(FaceEmbedding.embedding.is_not(None))
            .order_by(FaceEmbedding.created_at.desc())
            .limit(limit)
        )
        try:
            for face in self.db.scalars(stmt):
                if face.embedding and len(face.embedding) == self.settings.face_embedding_dim:
                    index.upsert_vector(
                        "face_embedding", face.id, list(face.embedding), flush=False
                    )
                    indexed += 1
            if indexed:
                index.flush("face_embedding")
        except VectorIndexError:
            logger.warning(
                "Could not rebuild Milvus face index; SQL remains authoritative",
                exc_info=True,
            )

    def _resolve_matches(
        self,
        scored_face_ids: list[tuple[uuid.UUID, float]],
    ) -> list[FaceMatch]:
        if not scored_face_ids:
            return []
        rows = self.db.execute(
            select(FaceEmbedding, Person)
            .join(Person, FaceEmbedding.person_id == Person.id)
            .where(FaceEmbedding.id.in_([face_id for face_id, _ in scored_face_ids]))
        )
        by_face_id = {face.id: (face, person) for face, person in rows}
        matches: list[FaceMatch] = []
        for face_id, similarity in scored_face_ids:
            row = by_face_id.get(face_id)
            if row is None:  # deleted between snapshot and lookup
                continue
            face, person = row
            matches.append(FaceMatch(person=person, face_embedding=face, similarity=similarity))
        return matches

    def _face_library(self) -> _FaceLibrary | None:
        if np is None or not self.settings.face_library_cache_enabled:
            return None
        signature = self._face_library_signature()
        ttl = self.settings.face_library_cache_ttl_seconds
        global _FACE_LIBRARY
        with _FACE_LIBRARY_LOCK:
            cached = _FACE_LIBRARY
        if (
            cached is not None
            and cached.signature == signature
            and monotonic() - cached.loaded_at < ttl
        ):
            return cached
        library = self._load_face_library(signature)
        with _FACE_LIBRARY_LOCK:
            _FACE_LIBRARY = library
        return library

    def _face_library_signature(self) -> tuple[int, int, str]:
        """Cheap fingerprint of the enrolled library: scan limit, row count, newest row."""

        total, latest = self.db.execute(
            select(func.count(FaceEmbedding.id), func.max(FaceEmbedding.created_at))
            .where(FaceEmbedding.person_id.is_not(None))
            .where(FaceEmbedding.embedding.is_not(None))
        ).one()
        return (self.settings.face_max_library_scan, int(total or 0), str(latest))

    def _load_face_library(self, signature: tuple[int, int, str]) -> _FaceLibrary:
        stmt = (
            select(FaceEmbedding.id, FaceEmbedding.embedding)
            .join(Person, FaceEmbedding.person_id == Person.id)
            .where(FaceEmbedding.embedding.is_not(None))
            .order_by(FaceEmbedding.created_at.desc())
            .limit(self.settings.face_max_library_scan)
        )
        ids_by_dim: dict[int, list[uuid.UUID]] = {}
        vectors_by_dim: dict[int, list[list[float]]] = {}
        for face_id, embedding in self.db.execute(stmt):
            if not embedding:
                continue
            dim = len(embedding)
            ids_by_dim.setdefault(dim, []).append(face_id)
            vectors_by_dim.setdefault(dim, []).append(embedding)
        buckets = {
            dim: (ids_by_dim[dim], np.asarray(vectors, dtype="float64"))
            for dim, vectors in vectors_by_dim.items()
        }
        return _FaceLibrary(signature=signature, loaded_at=monotonic(), buckets=buckets)

    def _best_known_match(
        self,
        matches: list[FaceMatch],
        threshold: float | None,
    ) -> FaceMatch | None:
        if not matches:
            return None
        best = matches[0]
        if best.similarity < self._threshold(threshold):
            return None
        return best

    def _create_event(
        self,
        image: Image,
        crop: PersonCrop | None,
        candidate: FaceCandidate | None,
        match: FaceMatch | None,
        result_type: str,
    ) -> RecognitionEvent:
        event = RecognitionEvent(
            **self._event_values(
                image=image,
                crop=crop,
                candidate=candidate,
                match=match,
                result_type=result_type,
            )
        )
        self.db.add(event)
        return event

    def _update_event(
        self,
        event: RecognitionEvent,
        image: Image,
        crop: PersonCrop | None,
        candidate: FaceCandidate | None,
        match: FaceMatch | None,
        result_type: str,
    ) -> RecognitionEvent:
        values = self._event_values(
            image=image,
            crop=crop,
            candidate=candidate,
            match=match,
            result_type=result_type,
        )
        for key, value in values.items():
            setattr(event, key, value)
        self.db.add(event)
        return event

    def _event_values(
        self,
        image: Image,
        crop: PersonCrop | None,
        candidate: FaceCandidate | None,
        match: FaceMatch | None,
        result_type: str,
    ) -> dict[str, object]:
        similarity = match.similarity if match else None
        return {
            "image_id": image.id,
            "crop_id": crop.id if crop else None,
            "person_id": match.person.id if match else None,
            "camera_id": crop.camera_id if crop else image.camera_id,
            "location_id": crop.location_id if crop else image.location_id,
            "confidence": candidate.quality_score if candidate else None,
            "similarity": similarity,
            "face_bbox": candidate.bbox if candidate else None,
            "result_type": result_type,
            "recognized_at": (crop.captured_at if crop else image.captured_at)
            or local_now(self.settings),
        }

    def _recognition_response(
        self,
        image: Image,
        crop: PersonCrop | None,
        candidate: FaceCandidate,
        event: RecognitionEvent,
        match: FaceMatch | None,
        matches: list[FaceMatch],
        threshold: float | None,
    ) -> FaceRecognitionResponse:
        return FaceRecognitionResponse(
            result_type="known" if match else "unknown",
            person=match.person if match else None,
            similarity=match.similarity if match else (matches[0].similarity if matches else None),
            threshold=self._threshold(threshold),
            image_id=image.id,
            crop_id=crop.id if crop else None,
            event_id=event.id,
            face_bbox=candidate.bbox,
            matches=[self._match_item(item) for item in matches],
        )

    def _match_item(self, match: FaceMatch) -> FaceMatchItem:
        return FaceMatchItem(
            person_id=match.person.id,
            person_name=match.person.name,
            face_embedding_id=match.face_embedding.id,
            similarity=match.similarity,
            quality_score=(
                float(match.face_embedding.quality_score)
                if match.face_embedding.quality_score is not None
                else None
            ),
            image_id=match.face_embedding.image_id,
            crop_id=match.face_embedding.crop_id,
        )

    def _threshold(self, threshold: float | None) -> float:
        return self.settings.face_match_threshold if threshold is None else threshold

    def _resolve_data_url(self, url: str) -> Path | None:
        prefix = "/data/"
        if not url.startswith(prefix):
            return None
        return self.settings.data_dir / url.removeprefix(prefix)

    def _normalize(self, vector: list[float]) -> list[float]:
        norm = math.sqrt(sum(value * value for value in vector))
        if norm <= 0:
            return []
        return [value / norm for value in vector]

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        length = min(len(left), len(right))
        if length == 0:
            return -1.0
        return float(sum(left[index] * float(right[index]) for index in range(length)))

    def _diagnostic_reason(
        self,
        candidate: FaceCandidate,
        similarity: float | None,
        threshold: float,
    ) -> str:
        width = float(candidate.bbox.get("width", 0.0))
        height = float(candidate.bbox.get("height", 0.0))
        if width < 40 or height < 40:
            return "检测到人脸但脸框偏小，建议用更清晰、更近的正脸补库。"
        if candidate.quality_score < 0.55:
            return "人脸检测分数偏低，可能有模糊、侧脸或遮挡。"
        if similarity is None:
            return "检测到人脸，但人脸库为空或没有同维度可比特征。"
        if similarity >= threshold:
            return "已超过阈值，可作为人脸轨迹命中。"
        if similarity >= threshold - 0.1:
            return "相似度接近阈值，可补充同摄像头正脸照片或小幅评估阈值。"
        return "相似度明显低于阈值，优先补充该人员在当前摄像头下的清晰正脸。"
