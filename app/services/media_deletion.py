"""Delete one image together with everything derived from it.

A single image drags a subtree behind it: person crops, the observation index, the face
extraction cache, ReID feedback, events, embeddings, durable index jobs, disk files and three
separate Milvus collections. Only ``reid_match_feedback`` declares ``ON DELETE CASCADE``; every
other reference is ``NO ACTION``, so the deletes have to run in a fixed order or PostgreSQL
rejects them (``person_observation_index_crop_id_fkey`` is the first to fire).

The order below was verified against the Beijing test database inside a rolled-back
transaction. See ``docs/media-deletion-api.zh-CN.md``.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import ColumnElement, and_, delete, func, or_, select, update
from sqlalchemy.orm import Session

from app.config.settings import Settings
from app.models.events import CountingEvent, RecognitionEvent
from app.models.media import Image, PersonCrop, PersonObservationIndex, VideoStream
from app.models.reid import ReidMatchFeedback
from app.models.vectors import CropFaceExtraction, FaceEmbedding, VectorIndexJob, VLEmbedding
from app.services.storage import StorageService

logger = logging.getLogger(__name__)

# Vector object types that can carry a claim on this image or its crops, with the collection
# suffix they map to. ReID lives in its own collection: a different model and dimension.
VECTOR_TARGETS: tuple[tuple[str, str], ...] = (
    ("image", "vl_images"),
    ("person_crop", "vl_person_crops"),
    ("reid_person_crop", "reid_person_crops"),
)

# Job targets whose object_id is a crop id or an image id owned by this subtree.
JOB_TARGETS: tuple[str, ...] = ("person_crop", "reid_person_crop", "person_attributes")

MILVUS_DELETE_BATCH = 200


class MediaDeletionUnavailable(RuntimeError):
    """A backend this deletion must write to is down, so nothing was destroyed."""


@dataclass
class ImageDeletionPlan:
    """Everything the caller needs, read before any row disappears."""

    image_id: uuid.UUID
    crop_ids: list[uuid.UUID]
    file_urls: list[str]
    warnings: list[str] = field(default_factory=list)


@dataclass
class ImageDeletionResult:
    image_id: uuid.UUID
    dry_run: bool
    counts: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


class MediaDeletionService:
    def __init__(self, db: Session, settings: Settings) -> None:
        self.db = db
        self.settings = settings
        self.storage = StorageService(settings)

    # -- entry point ----------------------------------------------------------------

    def delete_image(
        self,
        image_id: uuid.UUID,
        *,
        dry_run: bool = False,
        delete_files: bool = True,
        reason: str | None = None,
    ) -> ImageDeletionResult | None:
        """Return the deletion result, or ``None`` when the image does not exist.

        ``None`` is the idempotent case: deleting an already deleted image is not an error.
        """

        image = self.db.get(Image, image_id)
        if image is None:
            return None

        plan = self._build_plan(image)

        if dry_run:
            counts = self._count_rows(plan)
            counts["files"] = len(plan.file_urls)
            counts["milvus"] = self._vector_id_counts(plan)
            return ImageDeletionResult(image_id, True, counts, plan.warnings)

        # Refuse before touching anything when a backend we must write to cannot answer.
        # Skipping the vector delete would leave hits that point at rows which no longer exist.
        self._require_vector_backend()

        counts = self._delete_rows(plan)
        counts["milvus"] = self._delete_vectors(plan)
        counts["files"] = self._remove_files(plan.file_urls) if delete_files else 0

        logger.info(
            "deleted image %s (crops=%s files=%s reason=%s)",
            image_id,
            len(plan.crop_ids),
            counts["files"],
            reason or "-",
        )
        return ImageDeletionResult(image_id, False, counts, plan.warnings)

    # -- planning -------------------------------------------------------------------

    def _build_plan(self, image: Image) -> ImageDeletionPlan:
        crop_ids = list(
            self.db.scalars(select(PersonCrop.id).where(PersonCrop.image_id == image.id))
        )
        crop_urls = list(
            self.db.scalars(select(PersonCrop.crop_url).where(PersonCrop.image_id == image.id))
        )
        # image.video_url is deliberately absent: every keyframe of a video shares that file.
        file_urls = [url for url in (image.image_url, image.thumbnail_url, *crop_urls) if url]
        return ImageDeletionPlan(
            image_id=image.id,
            crop_ids=crop_ids,
            file_urls=file_urls,
            warnings=self._warnings(image.id, crop_ids),
        )

    def _warnings(self, image_id: uuid.UUID, crop_ids: Sequence[uuid.UUID]) -> list[str]:
        enrolled = self.db.scalar(
            select(func.count())
            .select_from(FaceEmbedding)
            .where(
                FaceEmbedding.person_id.is_not(None),
                or_(
                    FaceEmbedding.image_id == image_id,
                    FaceEmbedding.crop_id.in_(list(crop_ids)),
                ),
            )
        )
        if not enrolled:
            return []
        return [
            f"face_embeddings 里有 {enrolled} 条属于已登记人员，删除后该人员的人脸库条目会消失，"
            "人物轨迹（mode=face）会缺一段"
        ]

    # -- the ordered work -----------------------------------------------------------

    def _conditions(
        self, plan: ImageDeletionPlan
    ) -> list[tuple[str, type, ColumnElement[bool]]]:
        """Child tables in delete order.

        ``NO ACTION`` constraints are checked when a statement ends, so every table that
        references ``person_crops`` or ``images`` has to be cleared before its parent.
        """

        image_id = plan.image_id
        crop_ids = list(plan.crop_ids)
        return [
            (
                "reid_match_feedback",
                ReidMatchFeedback,
                or_(
                    ReidMatchFeedback.query_crop_id.in_(crop_ids),
                    ReidMatchFeedback.candidate_crop_id.in_(crop_ids),
                ),
            ),
            (
                "crop_face_extractions",
                CropFaceExtraction,
                CropFaceExtraction.crop_id.in_(crop_ids),
            ),
            (
                "person_observation_index",
                PersonObservationIndex,
                or_(
                    PersonObservationIndex.crop_id.in_(crop_ids),
                    PersonObservationIndex.image_id == image_id,
                ),
            ),
            (
                "counting_events",
                CountingEvent,
                or_(CountingEvent.crop_id.in_(crop_ids), CountingEvent.image_id == image_id),
            ),
            (
                "recognition_events",
                RecognitionEvent,
                or_(
                    RecognitionEvent.crop_id.in_(crop_ids),
                    RecognitionEvent.image_id == image_id,
                ),
            ),
            (
                "face_embeddings",
                FaceEmbedding,
                or_(FaceEmbedding.crop_id.in_(crop_ids), FaceEmbedding.image_id == image_id),
            ),
            (
                "vl_embeddings",
                VLEmbedding,
                or_(
                    and_(VLEmbedding.object_type == "image", VLEmbedding.object_id == image_id),
                    and_(
                        VLEmbedding.object_type == "person_crop",
                        VLEmbedding.object_id.in_(crop_ids),
                    ),
                ),
            ),
            (
                "vector_index_jobs",
                VectorIndexJob,
                or_(
                    VectorIndexJob.object_id == image_id,
                    and_(
                        VectorIndexJob.target.in_(list(JOB_TARGETS)),
                        VectorIndexJob.object_id.in_(crop_ids),
                    ),
                ),
            ),
            ("person_crops", PersonCrop, PersonCrop.image_id == image_id),
            ("images", Image, Image.id == image_id),
        ]

    def _count_rows(self, plan: ImageDeletionPlan) -> dict[str, Any]:
        counts: dict[str, Any] = {}
        for key, model, condition in self._conditions(plan):
            counts[key] = self.db.scalar(
                select(func.count()).select_from(model).where(condition)
            )
        counts["images"] = 1
        return counts

    def _delete_rows(self, plan: ImageDeletionPlan) -> dict[str, Any]:
        counts: dict[str, Any] = {}
        try:
            # Release the stream pointer first; leaving it set blocks the images delete.
            self.db.execute(
                update(VideoStream)
                .where(VideoStream.last_frame_image_id == plan.image_id)
                .values(last_frame_image_id=None)
            )
            for key, _model, condition in self._conditions(plan):
                result = self.db.execute(delete(_model).where(condition))
                counts[key] = self._rowcount(result)
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise
        counts["images"] = 1
        return counts

    @staticmethod
    def _rowcount(result: Any) -> int:
        try:
            return max(int(result.rowcount or 0), 0)
        except (TypeError, ValueError):  # pragma: no cover - driver without rowcount
            return 0

    # -- external backends ----------------------------------------------------------

    def _require_vector_backend(self) -> None:
        if not self.settings.milvus_enabled:
            return
        from app.services.vector_index import MilvusVectorIndex

        index = MilvusVectorIndex(self.settings)
        available, error = index.probe()
        if not available:
            raise MediaDeletionUnavailable(
                "Milvus 当前不可用，删除会留下无法清理的向量，已中止：" + (error or "unknown")
            )

    def _vector_id_counts(self, plan: ImageDeletionPlan) -> dict[str, int]:
        return {
            object_type: len(self._ids_for(object_type, plan))
            for object_type, _suffix in VECTOR_TARGETS
        }

    @staticmethod
    def _ids_for(object_type: str, plan: ImageDeletionPlan) -> list[uuid.UUID]:
        if object_type == "image":
            return [plan.image_id]
        return list(plan.crop_ids)

    def _delete_vectors(self, plan: ImageDeletionPlan) -> dict[str, int]:
        """Erase the vectors after the rows are gone.

        A failure here is reported rather than raised: the rows are already committed, so
        pretending the deletion did not happen would be worse than naming the residue.
        """

        counts: dict[str, int] = {object_type: 0 for object_type, _suffix in VECTOR_TARGETS}
        if not self.settings.milvus_enabled:
            return counts

        from app.services.vector_index import MilvusVectorIndex

        index = MilvusVectorIndex(self.settings)
        for object_type, _suffix in VECTOR_TARGETS:
            ids = self._ids_for(object_type, plan)
            if not ids:
                continue
            try:
                collection = index._collection(object_type)
                for start in range(0, len(ids), MILVUS_DELETE_BATCH):
                    batch = ids[start : start + MILVUS_DELETE_BATCH]
                    quoted = ", ".join(f'"{value}"' for value in batch)
                    collection.delete(expr=f"object_id in [{quoted}]")
                collection.flush()
                counts[object_type] = len(ids)
            except Exception as exc:
                message = f"Milvus 删除 {object_type} 向量失败，可能存在残留：{exc}"
                logger.warning("image deletion %s: %s", plan.image_id, message)
                plan.warnings.append(message)
        return counts

    def _remove_files(self, file_urls: Sequence[str]) -> int:
        return sum(1 for url in file_urls if self.storage.remove_data_url(url))
