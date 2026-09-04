from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import monotonic
from typing import Literal
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config.settings import Settings, get_settings
from app.models.events import CountingEvent, RecognitionEvent
from app.models.media import Image, PersonCrop, PersonObservationIndex, VideoStream
from app.models.persons import Person
from app.models.vectors import FaceEmbedding
from app.schemas.events import PersonTrajectoryPoint
from app.schemas.persons import PersonCreate
from app.services.embeddings import EmbeddingRuntimeError
from app.services.observation_index import ObservationIndexService
from app.services.reid import ReidRuntimeError
from app.services.reid_index import ReidIndexService
from app.services.time_utils import database_datetime
from app.services.vector_index import MilvusVectorIndex, VectorIndexError

TrajectoryMode = Literal["all", "face", "vector", "reid"]


@dataclass(frozen=True)
class TrajectoryVectorSeed:
    path: Path
    min_score: float


class PersonService:
    def __init__(self, db: Session, settings: Settings | None = None) -> None:
        self.db = db
        self.settings = settings or get_settings()
        self._camera_names: dict[uuid.UUID, tuple[str | None, str | None]] | None = None

    def _camera_labels(self, camera_id: uuid.UUID | None) -> tuple[str | None, str | None]:
        """Names for a camera id, read once per service instance.

        A trajectory point that carries only a uuid renders as `camera 2c2a45b3` -- the same
        thing the observation table used to show where a place name belongs.
        """

        if self._camera_names is None:
            self._camera_names = {
                stream.camera_id: (stream.name, stream.location_name)
                for stream in self.db.scalars(
                    select(VideoStream).where(VideoStream.camera_id.is_not(None))
                )
            }
        if camera_id is None:
            return None, None
        return self._camera_names.get(camera_id, (None, None))

    def create(self, payload: PersonCreate) -> Person:
        person = Person(**payload.model_dump())
        self.db.add(person)
        self.db.commit()
        self.db.refresh(person)
        return person

    def list(self, query: str | None = None, limit: int = 50) -> list[Person]:
        stmt = select(Person).order_by(Person.created_at.desc()).limit(limit)
        if query:
            stmt = stmt.where(Person.name.ilike(f"%{query}%"))
        return list(self.db.scalars(stmt))

    def get(self, person_id: uuid.UUID) -> Person | None:
        return self.db.get(Person, person_id)

    def label_crop(self, person: Person, crop: PersonCrop) -> PersonCrop:
        """Names the body in a crop, without going through a face.

        Face enrolment already sets person_id, but only as a side effect of finding a face --
        which rules out every crop shot from behind, and every deployment not running face
        recognition at all. ReID needs no face: the trajectory gallery is seeded from crops
        carrying this person_id, so writing it is the whole job.

        Deliberately no RecognitionEvent and no FaceEmbedding. Those record what a recogniser
        concluded; this records what a person asserted, and conflating the two would let a
        manual label masquerade as a face match in the observation table.
        """

        crop.person_id = person.id
        self.db.add(crop)
        if not person.avatar_url and crop.crop_url:
            person.avatar_url = crop.crop_url
            self.db.add(person)
        self.db.flush()
        ObservationIndexService(self.db, self.settings).upsert_crop(crop)
        self.db.commit()
        self.db.refresh(crop)
        return crop

    def unlabel_crop(self, crop: PersonCrop) -> PersonCrop:
        """Takes the name back off a crop, for when the label was wrong."""

        crop.person_id = None
        self.db.add(crop)
        self.db.flush()
        ObservationIndexService(self.db, self.settings).upsert_crop(crop)
        self.db.commit()
        self.db.refresh(crop)
        return crop

    def labelled_crops(self, person_id: uuid.UUID, limit: int) -> list[PersonCrop]:
        """The crops seeding this person's ReID gallery, newest first."""

        return list(
            self.db.scalars(
                select(PersonCrop)
                .where(PersonCrop.person_id == person_id)
                .order_by(PersonCrop.created_at.desc())
                .limit(limit)
            )
        )

    def events(
        self,
        person_id: uuid.UUID,
        limit: int = 100,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        min_similarity: float | None = None,
    ) -> list[RecognitionEvent]:
        stmt = (
            select(RecognitionEvent)
            .where(RecognitionEvent.person_id == person_id)
            .order_by(RecognitionEvent.recognized_at.desc())
            .limit(limit)
        )
        if start_time:
            stmt = stmt.where(
                RecognitionEvent.recognized_at >= self._database_time_boundary(start_time)
            )
        if end_time:
            stmt = stmt.where(
                RecognitionEvent.recognized_at <= self._database_time_boundary(end_time)
            )
        if min_similarity is not None:
            stmt = stmt.where(RecognitionEvent.similarity >= min_similarity)
        return list(self.db.scalars(stmt))

    def trajectory(
        self,
        person: Person,
        limit: int = 100,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        min_similarity: float | None = None,
        mode: TrajectoryMode = "all",
        backfill_missing: bool = False,
    ) -> list[PersonTrajectoryPoint]:
        points, _ = self.trajectory_report(
            person=person,
            limit=limit,
            start_time=start_time,
            end_time=end_time,
            min_similarity=min_similarity,
            mode=mode,
            backfill_missing=backfill_missing,
        )
        return points

    def trajectory_report(
        self,
        person: Person,
        limit: int = 100,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        min_similarity: float | None = None,
        mode: TrajectoryMode = "all",
        backfill_missing: bool = False,
    ) -> tuple[list[PersonTrajectoryPoint], list[str]]:
        """Trajectory points plus warnings, so a failed ReID lookup is visible in the response
        instead of masquerading as "no matches".

        Modes: `face` and `all` read the observation table (face-recognised rows); `vector`
        searches the generic VL embedding, which encodes appearance, not identity; `reid`
        searches the SapiensID identity index.
        """

        if (
            start_time is not None
            and end_time is not None
            and self._compare_datetime(start_time, end_time) > 0
        ):
            raise ValueError("start_time must be earlier than or equal to end_time")
        warnings: list[str] = []
        include_face = mode in {"all", "face"}
        include_vector = mode == "vector"
        include_reid = mode == "reid"
        if include_face:
            observation_points = self._observation_trajectory_points(
                person=person,
                limit=limit,
                start_time=start_time,
                end_time=end_time,
                min_similarity=min_similarity,
            )
            if observation_points or not backfill_missing:
                return observation_points, warnings

        events = self.events(
            person_id=person.id,
            limit=limit,
            start_time=start_time,
            end_time=end_time,
            min_similarity=min_similarity,
        )
        if include_face and backfill_missing and not events:
            self._backfill_face_trajectory_events(
                person=person,
                limit=limit,
                start_time=start_time,
                end_time=end_time,
            )
            events = self.events(
                person_id=person.id,
                limit=limit,
                start_time=start_time,
                end_time=end_time,
                min_similarity=min_similarity,
            )
        vector_matches: dict[uuid.UUID, float] = {}
        match_label = "vl_vector"
        if include_vector:
            vector_matches = self._vector_crop_matches(person, events)
        elif include_reid:
            match_label = "reid"
            reid_top_k = min(
                500,
                max(
                    self.settings.person_trajectory_vector_top_k,
                    limit * 4,
                ),
            )
            vector_matches, reid_error = self._reid_crop_matches(
                person,
                events,
                top_k=reid_top_k,
                min_score=min_similarity,
            )
            if reid_error:
                warnings.append(reid_error)
            if start_time is not None or end_time is not None:
                warnings.append(
                    "ReID 时间筛选在 ANN Top-K 召回后执行，较早候选可能被更新的高分 crop 截断"
                )
        event_crop_ids = {event.crop_id for event in events if event.crop_id}
        if mode == "all":
            vector_crop_ids = set(vector_matches) - event_crop_ids
        elif mode in {"vector", "reid"}:
            vector_crop_ids = set(vector_matches)
        else:
            vector_crop_ids = set()
        vector_events = self._events_by_crop_id(vector_crop_ids)
        vector_counting_events = self._counting_events_by_crop_id(vector_crop_ids)
        image_ids = {event.image_id for event in events if event.image_id and include_face}
        crop_ids = (set(event_crop_ids) if include_face else set()) | vector_crop_ids
        image_ids.update(event.image_id for event in vector_events.values() if event.image_id)
        images = {
            image.id: image
            for image in self.db.scalars(select(Image).where(Image.id.in_(image_ids)))
        } if image_ids else {}
        crops = {
            crop.id: crop
            for crop in self.db.scalars(select(PersonCrop).where(PersonCrop.id.in_(crop_ids)))
        } if crop_ids else {}
        image_ids.update(crop.image_id for crop in crops.values() if crop.image_id)
        if image_ids:
            images.update(
                {
                    image.id: image
                    for image in self.db.scalars(select(Image).where(Image.id.in_(image_ids)))
                }
            )

        points: list[PersonTrajectoryPoint] = []
        if include_face:
            points.extend(
                self._face_trajectory_point(
                    event=event,
                    person=person,
                    images=images,
                    crops=crops,
                    vector_score=vector_matches.get(event.crop_id) if event.crop_id else None,
                )
                for event in events
            )
        if include_vector or include_reid:
            vector_points: list[PersonTrajectoryPoint] = []
            reid_missing_capture_time = 0
            for crop_id in vector_crop_ids:
                crop = crops.get(crop_id)
                if crop is None:
                    continue
                event = vector_events.get(crop.id)
                image = images.get(crop.image_id)
                if not self._is_allowed_vector_match(
                    crop,
                    person,
                    event,
                    image=image,
                    require_camera=include_reid,
                ):
                    continue
                if include_reid and not (
                    crop.captured_at or (image.captured_at if image else None)
                ):
                    reid_missing_capture_time += 1
                    continue
                vector_points.append(
                    self._vector_trajectory_point(
                        crop=crop,
                        person=person,
                        event=event,
                        counting_event=vector_counting_events.get(crop.id),
                        image=image,
                        vector_score=vector_matches[crop.id],
                        source=match_label,
                    )
                )
            points.extend(vector_points)
            if reid_missing_capture_time:
                warnings.append(
                    f"已排除 {reid_missing_capture_time} 个缺少真实采集时间的 ReID crop；"
                    "不会用数据库创建时间冒充跨摄像头出现时间"
                )
        points = [
            point
            for point in points
            if self._is_within_time_range(point.recognized_at, start_time, end_time)
        ]
        points.sort(key=lambda item: item.recognized_at, reverse=True)
        return points[:limit], warnings

    def _observation_trajectory_points(
        self,
        *,
        person: Person,
        limit: int,
        start_time: datetime | None,
        end_time: datetime | None,
        min_similarity: float | None,
    ) -> list[PersonTrajectoryPoint]:
        stmt = (
            select(PersonObservationIndex)
            .where(PersonObservationIndex.person_id == person.id)
            .order_by(
                PersonObservationIndex.captured_at.desc(),
                PersonObservationIndex.created_at.desc(),
            )
            .limit(limit)
        )
        if start_time:
            stmt = stmt.where(
                PersonObservationIndex.captured_at >= self._database_time_boundary(start_time)
            )
        if end_time:
            stmt = stmt.where(
                PersonObservationIndex.captured_at <= self._database_time_boundary(end_time)
            )
        if min_similarity is not None:
            stmt = stmt.where(PersonObservationIndex.face_similarity >= min_similarity)
        rows = list(self.db.scalars(stmt))
        # The observation row does not carry the face box or the recognition timestamp; the
        # event that produced it does. Without this join the trajectory lost face_bbox
        # entirely, and rows whose crop had no capture time were silently dropped.
        events = self._events_by_crop_id({row.crop_id for row in rows if row.crop_id})
        return [
            self._observation_trajectory_point(
                row=row,
                person=person,
                event=events.get(row.crop_id) if row.crop_id else None,
            )
            for row in rows
        ]

    def _observation_trajectory_point(
        self,
        *,
        row: PersonObservationIndex,
        person: Person,
        event: RecognitionEvent | None = None,
    ) -> PersonTrajectoryPoint:
        return PersonTrajectoryPoint(
            event_id=event.id if event else None,
            image_id=row.image_id,
            crop_id=row.crop_id,
            person_id=person.id,
            person_name=row.person_name or person.name,
            camera_id=row.camera_id,
            camera_name=row.camera_name,
            location_id=row.location_id,
            location_name=row.location_name,
            similarity=row.face_similarity,
            vector_score=None,
            confidence=row.face_confidence,
            face_bbox=event.face_bbox if event else None,
            # These rows exist because face recognition assigned the person. The old label
            # switched to "face_vector" whenever the crop happened to carry a VL embedding,
            # which describes the crop, not how it was matched.
            match_source="face",
            result_type=row.recognition_result_type or "known",
            recognized_at=row.captured_at
            or (event.recognized_at if event else None)
            or row.created_at,
            image_url=row.thumbnail_url or row.image_url,
            crop_url=row.crop_url,
        )

    def _face_trajectory_point(
        self,
        *,
        event: RecognitionEvent,
        person: Person,
        images: dict[uuid.UUID, Image],
        crops: dict[uuid.UUID, PersonCrop],
        vector_score: float | None = None,
    ) -> PersonTrajectoryPoint:
        return PersonTrajectoryPoint(
            event_id=event.id,
            image_id=event.image_id,
            crop_id=event.crop_id,
            person_id=person.id,
            person_name=person.name,
            camera_id=event.camera_id,
            camera_name=self._camera_labels(event.camera_id)[0],
            location_id=event.location_id,
            location_name=self._camera_labels(event.camera_id)[1],
            similarity=float(event.similarity) if event.similarity is not None else None,
            vector_score=vector_score,
            confidence=float(event.confidence) if event.confidence is not None else None,
            face_bbox=event.face_bbox,
            match_source="face_vector" if vector_score is not None else "face",
            result_type=event.result_type,
            recognized_at=event.recognized_at,
            image_url=images[event.image_id].thumbnail_url
            or images[event.image_id].image_url
            if event.image_id in images
            else None,
            crop_url=crops[event.crop_id].crop_url if event.crop_id in crops else None,
        )

    def _vector_trajectory_point(
        self,
        *,
        crop: PersonCrop,
        person: Person,
        event: RecognitionEvent | None,
        counting_event: CountingEvent | None,
        image: Image | None,
        vector_score: float,
        source: str = "vl_vector",
    ) -> PersonTrajectoryPoint:
        if source == "reid":
            event_time = crop.captured_at or (image.captured_at if image else None)
            if event_time is None:
                raise ValueError("ReID trajectory points require a real captured_at value")
        else:
            event_time = (
                event.recognized_at
                if event is not None
                else counting_event.counted_at
                if counting_event is not None
                else crop.captured_at
                or crop.created_at
            )
        camera_id = (
            crop.camera_id
            or (image.camera_id if image else None)
            or (event.camera_id if event else None)
        )
        camera_name, location_name = self._camera_labels(camera_id)
        return PersonTrajectoryPoint(
            event_id=event.id if event else None,
            counting_event_id=counting_event.id if counting_event else None,
            image_id=crop.image_id,
            crop_id=crop.id,
            person_id=person.id,
            person_name=person.name,
            camera_id=camera_id,
            camera_name=camera_name,
            location_id=(
                crop.location_id
                or (image.location_id if image else None)
                or (event.location_id if event else None)
            ),
            location_name=location_name,
            similarity=float(event.similarity) if event and event.similarity is not None else None,
            vector_score=vector_score,
            confidence=float(event.confidence) if event and event.confidence is not None else None,
            face_bbox=event.face_bbox if event else None,
            match_source=source,
            result_type="reid_match" if source == "reid" else "vector_match",
            recognized_at=event_time,
            image_url=image.thumbnail_url or image.image_url if image else None,
            crop_url=crop.crop_url,
        )

    def _reid_crop_matches(
        self,
        person: Person,
        face_events: list[RecognitionEvent],
        *,
        top_k: int,
        min_score: float | None,
    ) -> tuple[dict[uuid.UUID, float], str | None]:
        """Identity matches from the SapiensID index, or an explanation of why there are none.

        Seeds are the person's confirmed body crops only. Face-enrollment portraits are the
        wrong domain for a whole-body model and are not silently substituted; with no body
        crops the caller gets a warning rather than matches built on bad seeds.
        """

        max_seconds = self.settings.person_trajectory_reid_max_seconds
        deadline = monotonic() + max_seconds
        bounded_settings = self.settings.model_copy(
            update={
                "reid_timeout_seconds": max(
                    1,
                    min(self.settings.reid_timeout_seconds, int(max_seconds)),
                ),
                "reid_queue_timeout_seconds": min(
                    self.settings.reid_queue_timeout_seconds,
                    max_seconds,
                ),
                "milvus_timeout_seconds": max(
                    0.1,
                    min(self.settings.milvus_timeout_seconds, max_seconds),
                ),
            }
        )
        service = ReidIndexService(self.db, bounded_settings)
        if not service.is_enabled():
            return {}, "ReID 未配置或不可用（REID_ENABLED / REID_SERVICE_URL / MILVUS_ENABLED）"
        pending_count = service.pending_count()
        seed_paths = [
            path
            for crop in self._trajectory_seed_crops(
                person,
                face_events,
                limit=self.settings.reid_gallery_size,
            )
            if crop.crop_url and (path := self._resolve_data_url(crop.crop_url)) and path.exists()
        ]
        if not seed_paths:
            return {}, "该人员没有可用的人体裁剪作为 ReID 底库；请先入库人体 crop"
        self._release_read_connection_before_external_io()
        gallery_warnings: list[str] = []
        try:
            aggregated = service.gallery_matches(
                seed_paths,
                top_k,
                deadline=deadline,
                warnings=gallery_warnings,
            )
        except (ReidRuntimeError, VectorIndexError) as exc:
            return {}, f"ReID 检索失败：{exc}"
        # Per-vote filtering happened inside the search; the zero-filled aggregation can still
        # land below the bar, and a one-off hit diluted by missing votes must not surface.
        threshold = max(self.settings.reid_min_score, min_score or 0.0)
        matches = {
            crop_id: score for crop_id, score in aggregated.items() if score >= threshold
        }
        if pending_count:
            gallery_warnings.append(
                f"ReID 索引仍有至少 {pending_count} 个 crop 未完成，当前结果可能不完整"
            )
        return matches, "；".join(gallery_warnings) or None

    def _vector_crop_matches(
        self,
        person: Person,
        face_events: list[RecognitionEvent],
    ) -> dict[uuid.UUID, float]:
        if not self.settings.person_trajectory_vector_enabled:
            return {}
        deadline = monotonic() + self.settings.person_trajectory_vector_max_seconds
        vector_index = MilvusVectorIndex(self._trajectory_vector_settings())
        if not vector_index.is_enabled():
            return {}

        seeds = self._trajectory_vector_seeds(person, face_events)
        self._release_read_connection_before_external_io()
        matches: dict[uuid.UUID, float] = {}
        for seed in seeds:
            remaining_seconds = deadline - monotonic()
            if remaining_seconds <= 0:
                break
            vector_index = MilvusVectorIndex(
                self._trajectory_vector_settings(remaining_seconds=remaining_seconds)
            )
            if not vector_index.is_enabled():
                break
            try:
                hits = vector_index.search_image(
                    "person_crop",
                    seed.path,
                    self.settings.person_trajectory_vector_top_k,
                )
            except (EmbeddingRuntimeError, VectorIndexError):
                continue
            for hit in hits or []:
                if hit.score < seed.min_score:
                    continue
                current = matches.get(hit.object_id)
                if current is None or hit.score > current:
                    matches[hit.object_id] = hit.score
        return matches

    def _trajectory_vector_settings(self, remaining_seconds: float | None = None) -> Settings:
        embedding_timeout = min(
            self.settings.visual_embedding_service_timeout_seconds,
            self.settings.person_trajectory_vector_embedding_timeout_seconds,
        )
        milvus_timeout = self.settings.milvus_timeout_seconds
        if remaining_seconds is not None:
            embedding_timeout = max(1, min(embedding_timeout, int(remaining_seconds)))
            milvus_timeout = max(0.1, min(milvus_timeout, remaining_seconds))
        if (
            embedding_timeout == self.settings.visual_embedding_service_timeout_seconds
            and milvus_timeout == self.settings.milvus_timeout_seconds
        ):
            return self.settings
        return self.settings.model_copy(
            update={
                "visual_embedding_service_timeout_seconds": embedding_timeout,
                "milvus_timeout_seconds": milvus_timeout,
            }
        )

    def _release_read_connection_before_external_io(self) -> None:
        try:
            self.db.expunge_all()
            if self.db.in_transaction():
                self.db.rollback()
        except Exception:
            return

    def _trajectory_vector_seeds(
        self,
        person: Person,
        face_events: list[RecognitionEvent],
    ) -> list[TrajectoryVectorSeed]:
        seeds: list[TrajectoryVectorSeed] = []
        for crop in self._trajectory_seed_crops(
            person,
            face_events,
            limit=self.settings.person_trajectory_vector_seed_limit,
        ):
            if path := self._resolve_data_url(crop.crop_url):
                seeds.append(
                    TrajectoryVectorSeed(
                        path=path,
                        min_score=self.settings.person_trajectory_vector_min_score,
                    )
                )
        if len(seeds) < self.settings.person_trajectory_vector_seed_limit:
            seeds.extend(
                TrajectoryVectorSeed(
                    path=path,
                    min_score=self.settings.person_trajectory_face_seed_vector_min_score,
                )
                for path in self._trajectory_face_seed_paths(
                    person,
                    limit=self.settings.person_trajectory_vector_seed_limit - len(seeds),
                )
            )
        unique_seeds = list(dict.fromkeys(seed for seed in seeds if seed.path.exists()))
        return unique_seeds[: self.settings.person_trajectory_vector_seed_limit]

    def _trajectory_seed_crops(
        self,
        person: Person,
        face_events: list[RecognitionEvent],
        *,
        limit: int,
    ) -> list[PersonCrop]:
        seed_ids = [event.crop_id for event in face_events if event.crop_id]
        if len(seed_ids) < limit:
            extra_stmt = (
                select(PersonCrop.id)
                .where(PersonCrop.person_id == person.id)
                .order_by(PersonCrop.created_at.desc())
                .limit(limit)
            )
            seed_ids.extend(self.db.scalars(extra_stmt))
        unique_ids = list(dict.fromkeys(seed_ids))[:limit]
        if not unique_ids:
            return []
        crops_by_id = {
            crop.id: crop
            for crop in self.db.scalars(select(PersonCrop).where(PersonCrop.id.in_(unique_ids)))
        }
        return [crops_by_id[crop_id] for crop_id in unique_ids if crop_id in crops_by_id]

    def _trajectory_face_seed_paths(self, person: Person, limit: int) -> list[Path]:
        if limit <= 0:
            return []
        stmt = (
            select(FaceEmbedding, Image, PersonCrop)
            .join(Image, FaceEmbedding.image_id == Image.id, isouter=True)
            .join(PersonCrop, FaceEmbedding.crop_id == PersonCrop.id, isouter=True)
            .where(FaceEmbedding.person_id == person.id)
            .where(FaceEmbedding.embedding.is_not(None))
            .order_by(FaceEmbedding.created_at.desc())
            .limit(limit)
        )
        paths: list[Path] = []
        for _face, image, crop in self.db.execute(stmt):
            url = crop.crop_url if crop is not None else image.image_url if image else None
            if url and (path := self._resolve_data_url(url)):
                paths.append(path)
        if person.avatar_url and len(paths) < limit:
            if path := self._resolve_data_url(person.avatar_url):
                paths.append(path)
        return paths

    def _events_by_crop_id(self, crop_ids: set[uuid.UUID]) -> dict[uuid.UUID, RecognitionEvent]:
        if not crop_ids:
            return {}
        stmt = (
            select(RecognitionEvent)
            .where(RecognitionEvent.crop_id.in_(crop_ids))
            .order_by(RecognitionEvent.recognized_at.desc())
        )
        events: dict[uuid.UUID, RecognitionEvent] = {}
        for event in self.db.scalars(stmt):
            if event.crop_id is not None and event.crop_id not in events:
                events[event.crop_id] = event
        return events

    def _counting_events_by_crop_id(
        self, crop_ids: set[uuid.UUID]
    ) -> dict[uuid.UUID, CountingEvent]:
        if not crop_ids:
            return {}
        stmt = (
            select(CountingEvent)
            .where(CountingEvent.crop_id.in_(crop_ids))
            .order_by(CountingEvent.counted_at.desc())
        )
        events: dict[uuid.UUID, CountingEvent] = {}
        for event in self.db.scalars(stmt):
            if event.crop_id is not None and event.crop_id not in events:
                events[event.crop_id] = event
        return events

    def _backfill_face_trajectory_events(
        self,
        *,
        person: Person,
        limit: int,
        start_time: datetime | None,
        end_time: datetime | None,
    ) -> None:
        latest_face_embedding_at = self._latest_person_face_embedding_created_at(person)
        if latest_face_embedding_at is None:
            return

        from app.services.faces import FaceRecognitionService

        face_service = FaceRecognitionService(self.db, self.settings)
        stmt = (
            select(PersonCrop, Image)
            .join(Image, PersonCrop.image_id == Image.id)
            .order_by(PersonCrop.created_at.desc())
            .limit(limit)
        )
        if start_time:
            stmt = stmt.where(
                PersonCrop.captured_at >= self._database_time_boundary(start_time)
            )
        if end_time:
            stmt = stmt.where(
                PersonCrop.captured_at <= self._database_time_boundary(end_time)
            )

        candidates = list(self.db.execute(stmt))
        existing_events = self._latest_recognition_events_for_crop_ids(
            {crop.id for crop, _image in candidates}
        )
        for crop, image in candidates:
            existing_event = existing_events.get(crop.id)
            if existing_event is not None and existing_event.person_id == person.id:
                continue
            if existing_event is not None and existing_event.person_id is not None:
                continue
            if existing_event is not None:
                continue
            try:
                face_service.recognize_crop(
                    crop,
                    image,
                    existing_event=existing_event,
                    require_ingest_enabled=False,
                )
            except Exception:
                continue

    def _latest_person_face_embedding_created_at(self, person: Person) -> datetime | None:
        stmt = (
            select(FaceEmbedding.created_at)
            .where(FaceEmbedding.person_id == person.id)
            .where(FaceEmbedding.embedding.is_not(None))
            .order_by(FaceEmbedding.created_at.desc())
            .limit(1)
        )
        return self.db.scalar(stmt)

    def _latest_recognition_event_for_crop(
        self,
        crop_id: uuid.UUID,
    ) -> RecognitionEvent | None:
        return self.db.scalar(
            select(RecognitionEvent)
            .where(RecognitionEvent.crop_id == crop_id)
            .order_by(RecognitionEvent.created_at.desc())
            .limit(1)
        )

    def _latest_recognition_events_for_crop_ids(
        self,
        crop_ids: set[uuid.UUID],
    ) -> dict[uuid.UUID, RecognitionEvent]:
        if not crop_ids:
            return {}
        stmt = (
            select(RecognitionEvent)
            .where(RecognitionEvent.crop_id.in_(crop_ids))
            .order_by(RecognitionEvent.crop_id, RecognitionEvent.created_at.desc())
        )
        events: dict[uuid.UUID, RecognitionEvent] = {}
        for event in self.db.scalars(stmt):
            if event.crop_id is not None and event.crop_id not in events:
                events[event.crop_id] = event
        return events

    def _is_allowed_vector_match(
        self,
        crop: PersonCrop,
        person: Person,
        event: RecognitionEvent | None,
        *,
        image: Image | None = None,
        require_camera: bool = False,
    ) -> bool:
        if (
            require_camera
            and crop.camera_id is None
            and (image is None or image.camera_id is None)
            and (event is None or event.camera_id is None)
        ):
            return False
        if crop.person_id is not None and crop.person_id != person.id:
            return False
        if event is not None and event.person_id is not None and event.person_id != person.id:
            return False
        return True

    def _is_within_time_range(
        self,
        value: datetime,
        start_time: datetime | None,
        end_time: datetime | None,
    ) -> bool:
        if start_time is not None and self._compare_datetime(value, start_time) < 0:
            return False
        if end_time is not None and self._compare_datetime(value, end_time) > 0:
            return False
        return True

    def _compare_datetime(self, left: datetime, right: datetime) -> int:
        comparable_left = self._normalize_datetime_for_compare(left, right)
        if comparable_left < right:
            return -1
        if comparable_left > right:
            return 1
        return 0

    def _normalize_datetime_for_compare(self, value: datetime, boundary: datetime) -> datetime:
        local_timezone = ZoneInfo(self.settings.local_timezone)
        aware_value = value.replace(tzinfo=local_timezone) if value.tzinfo is None else value
        if boundary.tzinfo is not None:
            return aware_value.astimezone(boundary.tzinfo)
        return aware_value.astimezone(local_timezone).replace(tzinfo=None)

    def _database_time_boundary(self, value: datetime) -> datetime:
        """Match each database dialect's DateTime(timezone=True) round-trip semantics."""

        return database_datetime(value, self.settings, self.db.get_bind().dialect.name)

    def _resolve_data_url(self, url: str) -> Path | None:
        prefix = "/data/"
        if not url.startswith(prefix):
            return None
        return self.settings.data_dir / Path(url.removeprefix(prefix))
