from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.config.settings import Settings, get_settings
from app.models.events import RecognitionEvent
from app.models.media import Image, PersonCrop, PersonObservationIndex
from app.models.persons import Person
from app.schemas.common import SearchFilters
from app.schemas.media import (
    ImageSearchRequest,
    SearchResponse,
    SearchResultItem,
    VisualSearchRequest,
)
from app.services.embeddings import EmbeddingRuntimeError
from app.services.observation_index import ObservationIndexService, ObservationSearchHit
from app.services.rerank import EmbeddingRerankService, VLMRerankService
from app.services.vector_index import MilvusVectorIndex, VectorIndexError, VectorSearchHit
from app.services.vlm import VLMRuntimeError

STRUCTURED_CONFIDENCE_MIN = 0.6


@dataclass(frozen=True)
class StructuredCondition:
    field: str
    values: tuple[object, ...]

    def as_dict(self) -> dict[str, object]:
        return {"field": self.field, "values": list(self.values)}


class VisualSearchService:
    """Business search over people, structured labels, time, and location.

    CLIP text/image similarity is intentionally not part of this service's business
    recall path.  Identity vectors remain available through the face and ReID
    services, while this service only returns explicit database/tag matches.
    """

    def __init__(self, db: Session, settings: Settings | None = None) -> None:
        self.db = db
        self.settings = settings or get_settings()
        self.vector_index = MilvusVectorIndex(self.settings)

    def search(self, payload: VisualSearchRequest) -> SearchResponse:
        if payload.target == "image":
            return SearchResponse(items=[])
        return self._search_person_crops(payload)

    def search_by_image(self, payload: ImageSearchRequest) -> SearchResponse:
        # Generic CLIP image similarity was removed from the business search path.
        # Face-image lookup and person ReID use their dedicated identity spaces.
        return SearchResponse(items=[])

    def _search_person_crops(self, payload: VisualSearchRequest) -> SearchResponse:
        requested_top_k = payload.top_k
        structured_result, conditions = StructuredSearchService(self.db, self.settings).search(
            payload.query,
            top_k=requested_top_k,
            filters=payload.filters,
        )
        person_result = self._search_person_crops_by_known_person(payload)
        if person_result is not None:
            return SearchResponse(items=person_result.items[:requested_top_k])

        # Parsed label queries are strict AND matches. Do not mix in observation
        # keyword hits because they can satisfy only part of a multi-label query.
        if conditions:
            return SearchResponse(items=structured_result.items[:requested_top_k])

        observation_items = [
            self._search_item_from_observation(hit)
            for hit in ObservationIndexService(self.db, self.settings).search(
                payload.query,
                top_k=requested_top_k,
                filters=payload.filters,
            )
        ]
        if observation_items:
            return SearchResponse(items=observation_items[:requested_top_k])

        label_result = self._search_person_crops_by_label(payload)
        if label_result is not None:
            return SearchResponse(items=label_result.items[:requested_top_k])

        # Never replace an empty search with unrelated recent crops.
        return SearchResponse(items=[])

    def _search_person_crops_by_known_person(
        self,
        payload: VisualSearchRequest,
    ) -> SearchResponse | None:
        persons = self._query_known_persons(payload.query)
        if not persons:
            return None
        person_ids = [person.id for person in persons]
        filters = payload.filters
        if filters.person_id and filters.person_id not in person_ids:
            return SearchResponse(items=[])

        observation_stmt = (
            select(PersonObservationIndex)
            .where(PersonObservationIndex.person_id.in_(person_ids))
            .order_by(
                PersonObservationIndex.captured_at.desc(),
                PersonObservationIndex.created_at.desc(),
            )
            .limit(payload.top_k)
        )
        observation_stmt = ObservationIndexService(self.db, self.settings)._apply_filters(
            observation_stmt,
            filters,
        )
        observation_items = [
            self._search_item_from_observation(ObservationSearchHit(row=row, score=1.5))
            for row in self.db.scalars(observation_stmt)
        ]
        if observation_items:
            return SearchResponse(items=observation_items)

        crop_stmt = (
            select(PersonCrop)
            .where(PersonCrop.person_id.in_(person_ids))
            .order_by(PersonCrop.captured_at.desc(), PersonCrop.created_at.desc())
            .limit(payload.top_k)
        )
        crop_stmt = self._apply_crop_filters(crop_stmt, filters)
        person_by_id = {person.id: person for person in persons}
        crops = list(self.db.scalars(crop_stmt))

        if not crops:
            event_stmt = (
                select(RecognitionEvent)
                .where(RecognitionEvent.person_id.in_(person_ids))
                .order_by(RecognitionEvent.recognized_at.desc(), RecognitionEvent.created_at.desc())
                .limit(payload.top_k)
            )
            if filters.start_time:
                event_stmt = event_stmt.where(RecognitionEvent.recognized_at >= filters.start_time)
            if filters.end_time:
                event_stmt = event_stmt.where(RecognitionEvent.recognized_at <= filters.end_time)
            if filters.camera_id:
                event_stmt = event_stmt.where(RecognitionEvent.camera_id == filters.camera_id)
            if filters.location_id:
                event_stmt = event_stmt.where(RecognitionEvent.location_id == filters.location_id)
            crop_ids = [
                event.crop_id
                for event in self.db.scalars(event_stmt)
                if event.crop_id is not None
            ]
            if crop_ids:
                event_crop_stmt = select(PersonCrop).where(PersonCrop.id.in_(crop_ids))
                event_crop_stmt = self._apply_crop_filters(event_crop_stmt, filters)
                crop_by_id = {crop.id: crop for crop in self.db.scalars(event_crop_stmt)}
                crops = [crop_by_id[crop_id] for crop_id in crop_ids if crop_id in crop_by_id]

        return SearchResponse(
            items=[
                self._search_item_from_crop(
                    crop,
                    1.5,
                    None,
                    person_by_id.get(crop.person_id),
                )
                for crop in crops
            ]
        )

    def _query_known_persons(self, query: str) -> list[Person]:
        normalized = query.strip().lower()
        if not normalized:
            return []
        stmt = select(Person).where(Person.status == "active").order_by(Person.created_at.desc())
        persons = []
        for person in self.db.scalars(stmt):
            candidates = [
                person.name,
                person.employee_no,
                person.phone,
            ]
            if any(value and str(value).strip().lower() in normalized for value in candidates):
                persons.append(person)
        return persons

    def _apply_crop_filters(self, stmt: Any, filters: SearchFilters) -> Any:
        if filters.person_id:
            stmt = stmt.where(PersonCrop.person_id == filters.person_id)
        if filters.camera_id:
            stmt = stmt.where(PersonCrop.camera_id == filters.camera_id)
        if filters.location_id:
            stmt = stmt.where(PersonCrop.location_id == filters.location_id)
        if filters.start_time:
            stmt = stmt.where(PersonCrop.captured_at >= filters.start_time)
        if filters.end_time:
            stmt = stmt.where(PersonCrop.captured_at <= filters.end_time)
        return stmt

    def _merge_person_crop_results(
        self,
        vector_result: SearchResponse,
        observation_result: SearchResponse,
        structured_result: SearchResponse,
        *,
        limit: int,
    ) -> SearchResponse:
        merged: dict[object, SearchResultItem] = {}
        ranks: dict[object, int] = {}
        for rank, result in (
            (3, vector_result),
            (2, observation_result),
            (1, structured_result),
        ):
            for item in result.items:
                key = item.crop_id or item.image_id
                if key is None:
                    continue
                existing = merged.get(key)
                if existing is None:
                    merged[key] = item
                    ranks[key] = rank
                    continue
                merged[key] = self._merge_search_item(existing, item)
                ranks[key] = max(ranks.get(key, 0), rank)
        items = list(merged.values())
        items.sort(
            key=lambda item: (
                ranks.get(item.crop_id or item.image_id, 0),
                item.score,
            ),
            reverse=True,
        )
        return SearchResponse(items=items[:limit])

    def _merge_search_item(
        self,
        primary: SearchResultItem,
        secondary: SearchResultItem,
    ) -> SearchResultItem:
        data = primary.model_dump()
        secondary_data = secondary.model_dump()
        for key, value in secondary_data.items():
            if key == "score":
                if float(data.get(key) or 0.0) <= 0.0:
                    data[key] = float(value or 0.0)
            elif key in {"original_score", "embedding_rerank_score", "rerank_score"}:
                data[key] = data.get(key) if data.get(key) is not None else value
            elif value is not None and data.get(key) is None:
                data[key] = value
            elif key in {"attributes", "labels_zh", "labels_en"} and value:
                data[key] = data.get(key) or value
        return SearchResultItem(**data)

    def _search_item_from_observation(
        self,
        hit: ObservationSearchHit,
    ) -> SearchResultItem:
        row = hit.row
        return SearchResultItem(
            crop_id=row.crop_id,
            image_id=row.image_id,
            image_url=row.image_url,
            crop_url=row.crop_url,
            score=hit.score,
            captured_at=row.captured_at,
            location_id=row.location_id,
            location_name=row.location_name,
            camera_id=row.camera_id,
            camera_name=row.camera_name,
            person_id=row.person_id,
            person_name=row.person_name or "未知",
            attributes=row.attributes,
            labels_zh=row.labels_zh,
            labels_en=row.labels_en,
        )

    def _candidate_payload(self, payload: VisualSearchRequest) -> VisualSearchRequest:
        candidate_limits = [payload.top_k]
        if self.settings.embedding_rerank_enabled:
            candidate_limits.append(self.settings.embedding_rerank_candidate_limit)
        if (
            payload.rerank and self.settings.vlm_rerank_service_url
        ) or self.settings.vlm_rerank_enabled:
            candidate_limits.append(self.settings.vlm_rerank_candidate_limit)
        candidate_top_k = max(candidate_limits)
        return payload.model_copy(update={"top_k": min(candidate_top_k, 100)})

    def _maybe_rerank_person_crops(
        self,
        payload: VisualSearchRequest,
        result: SearchResponse,
        limit: int,
    ) -> SearchResponse:
        should_embedding_rerank = self.settings.embedding_rerank_enabled
        should_vlm_rerank = (
            payload.rerank and bool(self.settings.vlm_rerank_service_url)
        ) or self.settings.vlm_rerank_enabled
        if not should_embedding_rerank and not should_vlm_rerank:
            return SearchResponse(items=result.items[:limit])
        if not result.items:
            return result
        current_items = result.items
        if should_embedding_rerank:
            try:
                embedding_limit = limit
                if should_vlm_rerank:
                    embedding_limit = max(limit, self.settings.vlm_rerank_candidate_limit)
                embedding_items = EmbeddingRerankService(self.settings).rerank_person_crops(
                    payload.query,
                    current_items,
                    limit=embedding_limit,
                )
            except EmbeddingRuntimeError:
                embedding_items = []
            if embedding_items:
                current_items = embedding_items
        if not should_vlm_rerank:
            return SearchResponse(items=current_items[:limit])
        try:
            reranked_items = VLMRerankService(self.settings).rerank_person_crops(
                payload.query,
                current_items,
                limit=limit,
            )
        except VLMRuntimeError:
            return SearchResponse(items=current_items[:limit])
        if not reranked_items:
            return SearchResponse(items=current_items[:limit])
        return SearchResponse(items=reranked_items)

    def _search_person_crops_by_label(self, payload: VisualSearchRequest) -> SearchResponse | None:
        labels = self._query_labels(payload.query)
        if not labels:
            return None
        stmt = select(PersonCrop).order_by(PersonCrop.created_at.desc()).limit(payload.top_k)
        stmt = stmt.where(or_(*(PersonCrop.bbox["label"].as_string() == label for label in labels)))
        filters = payload.filters
        if filters.person_id:
            stmt = stmt.where(PersonCrop.person_id == filters.person_id)
        if filters.camera_id:
            stmt = stmt.where(PersonCrop.camera_id == filters.camera_id)
        if filters.location_id:
            stmt = stmt.where(PersonCrop.location_id == filters.location_id)
        if filters.start_time:
            stmt = stmt.where(PersonCrop.captured_at >= filters.start_time)
        if filters.end_time:
            stmt = stmt.where(PersonCrop.captured_at <= filters.end_time)
        crops = list(self.db.scalars(stmt))
        return SearchResponse(
            items=[
                SearchResultItem(
                    crop_id=crop.id,
                    image_id=crop.image_id,
                    crop_url=crop.crop_url,
                    score=1.0,
                    captured_at=crop.captured_at or crop.created_at,
                    location_id=crop.location_id,
                    camera_id=crop.camera_id,
                    person_name=str((crop.bbox or {}).get("label") or "未知"),
                    attributes=crop.attributes,
                )
                for crop in crops
            ]
        )

    def _query_labels(self, query: str) -> list[str]:
        normalized = query.lower()
        labels: list[str] = []
        if any(token in normalized for token in ("抽烟", "吸烟", "smoking", "smoke")):
            labels.append("smoking")
        if any(token in normalized for token in ("打电话", "手机", "phone", "call")):
            labels.append("phone")
        return labels

    def _search_images(self, payload: VisualSearchRequest) -> SearchResponse:
        vector_hits = self._try_vector_search("image", payload)
        if vector_hits is not None:
            return self._image_results_from_hits(vector_hits, payload)

        stmt = select(Image).order_by(Image.created_at.desc()).limit(payload.top_k)
        filters = payload.filters
        if filters.camera_id:
            stmt = stmt.where(Image.camera_id == filters.camera_id)
        if filters.location_id:
            stmt = stmt.where(Image.location_id == filters.location_id)
        if filters.start_time:
            stmt = stmt.where(Image.captured_at >= filters.start_time)
        if filters.end_time:
            stmt = stmt.where(Image.captured_at <= filters.end_time)
        images = list(self.db.scalars(stmt))
        return SearchResponse(
            items=[
                SearchResultItem(
                    image_id=image.id,
                    image_url=image.image_url,
                    score=0.0,
                    captured_at=image.captured_at,
                    location_id=image.location_id,
                )
                for image in images
            ]
        )

    def _try_vector_search(
        self,
        object_type: str,
        payload: VisualSearchRequest,
    ) -> list[VectorSearchHit] | None:
        if not self.vector_index.is_enabled():
            return None
        try:
            hits = self.vector_index.search_text(object_type, payload.query, payload.top_k)
        except (EmbeddingRuntimeError, VectorIndexError):
            return []
        return self._filter_vector_hits(hits or [])

    def _filter_vector_hits(self, hits: list[VectorSearchHit]) -> list[VectorSearchHit]:
        min_score = self.settings.visual_search_min_score
        if min_score <= 0:
            return hits
        return [hit for hit in hits if hit.score >= min_score]

    def _query_image_path(self, payload: ImageSearchRequest) -> Path | None:
        image = self.db.get(Image, payload.image_id)
        if image is None:
            return None
        source_url = image.image_url
        if payload.target == "person_crop":
            crop = self.db.scalar(
                select(PersonCrop)
                .where(PersonCrop.image_id == payload.image_id)
                .order_by(PersonCrop.created_at.desc())
            )
            if crop is not None:
                source_url = crop.crop_url
        return self._resolve_data_url(source_url)

    def _resolve_data_url(self, url: str) -> Path | None:
        prefix = "/data/"
        if not url.startswith(prefix):
            return None
        return self.settings.data_dir / url.removeprefix(prefix)

    def _person_crop_results_from_hits(
        self,
        hits: list[VectorSearchHit],
        payload: VisualSearchRequest,
    ) -> SearchResponse:
        hits = self._filter_vector_hits(hits)
        if not hits:
            return SearchResponse(items=[])
        scores = {hit.object_id: hit.score for hit in hits}
        ids = [hit.object_id for hit in hits]
        stmt = select(PersonCrop).where(PersonCrop.id.in_(ids))
        filters = payload.filters
        if filters.person_id:
            stmt = stmt.where(PersonCrop.person_id == filters.person_id)
        if filters.camera_id:
            stmt = stmt.where(PersonCrop.camera_id == filters.camera_id)
        if filters.location_id:
            stmt = stmt.where(PersonCrop.location_id == filters.location_id)
        if filters.start_time:
            stmt = stmt.where(PersonCrop.captured_at >= filters.start_time)
        if filters.end_time:
            stmt = stmt.where(PersonCrop.captured_at <= filters.end_time)

        crops_by_id = {crop.id: crop for crop in self.db.scalars(stmt)}
        observations_by_crop_id = ObservationIndexService(
            self.db,
            self.settings,
        ).rows_by_crop_ids(ids, filters)
        return SearchResponse(
            items=[
                self._search_item_from_crop(
                    crop,
                    scores[crop.id],
                    observations_by_crop_id.get(crop.id),
                )
                for object_id in ids
                if (crop := crops_by_id.get(object_id)) is not None
            ]
        )

    def _search_item_from_crop(
        self,
        crop: PersonCrop,
        score: float,
        observation: PersonObservationIndex | None = None,
        person: Person | None = None,
    ) -> SearchResultItem:
        if observation is not None:
            return SearchResultItem(
                crop_id=crop.id,
                image_id=crop.image_id,
                image_url=observation.image_url,
                crop_url=crop.crop_url,
                score=score,
                captured_at=observation.captured_at,
                location_id=observation.location_id,
                location_name=observation.location_name,
                camera_id=observation.camera_id,
                camera_name=observation.camera_name,
                person_id=observation.person_id,
                person_name=observation.person_name or "未知",
                attributes=observation.attributes or crop.attributes,
                labels_zh=observation.labels_zh,
                labels_en=observation.labels_en,
            )
        return SearchResultItem(
            crop_id=crop.id,
            image_id=crop.image_id,
            crop_url=crop.crop_url,
            score=score,
            captured_at=crop.captured_at,
            location_id=crop.location_id,
            camera_id=crop.camera_id,
            person_id=crop.person_id,
            person_name=person.name if person else ("未知" if crop.person_id is None else None),
            attributes=crop.attributes,
        )

    def _image_results_from_hits(
        self,
        hits: list[VectorSearchHit],
        payload: VisualSearchRequest,
    ) -> SearchResponse:
        hits = self._filter_vector_hits(hits)
        if not hits:
            return SearchResponse(items=[])
        scores = {hit.object_id: hit.score for hit in hits}
        ids = [hit.object_id for hit in hits]
        stmt = select(Image).where(Image.id.in_(ids))
        filters = payload.filters
        if filters.camera_id:
            stmt = stmt.where(Image.camera_id == filters.camera_id)
        if filters.location_id:
            stmt = stmt.where(Image.location_id == filters.location_id)
        if filters.start_time:
            stmt = stmt.where(Image.captured_at >= filters.start_time)
        if filters.end_time:
            stmt = stmt.where(Image.captured_at <= filters.end_time)

        images_by_id = {image.id: image for image in self.db.scalars(stmt)}
        return SearchResponse(
            items=[
                SearchResultItem(
                    image_id=image.id,
                    image_url=image.thumbnail_url or image.image_url,
                    score=scores[image.id],
                    captured_at=image.captured_at,
                    location_id=image.location_id,
                )
                for object_id in ids
                if (image := images_by_id.get(object_id)) is not None
            ]
        )


class StructuredSearchService:
    """Search person crops by normalized attributes extracted by VLM."""

    def __init__(self, db: Session, settings: Settings | None = None) -> None:
        self.db = db
        self.settings = settings or get_settings()

    def search(
        self,
        query: str,
        top_k: int = 8,
        filters: SearchFilters | None = None,
    ) -> tuple[SearchResponse, list[StructuredCondition]]:
        conditions = self.parse_query(query)
        if not conditions:
            return SearchResponse(items=[]), []
        filters = filters or SearchFilters()
        stmt = select(PersonCrop).order_by(PersonCrop.created_at.desc()).limit(500)
        if filters.person_id:
            stmt = stmt.where(PersonCrop.person_id == filters.person_id)
        if filters.camera_id:
            stmt = stmt.where(PersonCrop.camera_id == filters.camera_id)
        if filters.location_id:
            stmt = stmt.where(PersonCrop.location_id == filters.location_id)
        if filters.start_time:
            stmt = stmt.where(PersonCrop.captured_at >= filters.start_time)
        if filters.end_time:
            stmt = stmt.where(PersonCrop.captured_at <= filters.end_time)

        items: list[SearchResultItem] = []
        observation_service = ObservationIndexService(self.db, self.settings)
        for crop in self.db.scalars(stmt):
            score = self._match_score(crop, conditions)
            if score <= 0:
                continue
            observation = observation_service.rows_by_crop_ids([crop.id], filters).get(crop.id)
            items.append(
                SearchResultItem(
                    crop_id=crop.id,
                    image_id=crop.image_id,
                    image_url=observation.image_url if observation else None,
                    crop_url=crop.crop_url,
                    score=score,
                    captured_at=(
                        observation.captured_at
                        if observation
                        else crop.captured_at or crop.created_at
                    ),
                    location_id=observation.location_id if observation else crop.location_id,
                    location_name=observation.location_name if observation else None,
                    camera_id=observation.camera_id if observation else crop.camera_id,
                    camera_name=observation.camera_name if observation else None,
                    person_id=observation.person_id if observation else crop.person_id,
                    person_name=(
                        observation.person_name
                        if observation and observation.person_name
                        else ("未知" if crop.person_id is None else None)
                    ),
                    attributes=observation.attributes if observation else crop.attributes,
                    labels_zh=observation.labels_zh if observation else None,
                    labels_en=observation.labels_en if observation else None,
                )
            )
            if len(items) >= top_k and items[-1].score >= 1.0:
                break
        items.sort(key=lambda item: item.score, reverse=True)
        return SearchResponse(items=items), conditions

    def parse_query(self, query: str) -> list[StructuredCondition]:
        normalized = query.lower()
        conditions: list[StructuredCondition] = []
        if any(token in normalized for token in ("光头", "秃头", "没有头发", "bald")):
            conditions.append(StructuredCondition("hair", ("bald", "shaved")))
        elif any(token in normalized for token in ("短发", "short hair", "short_hair")):
            conditions.append(StructuredCondition("hair", ("short_hair", "shaved")))
        elif any(token in normalized for token in ("长发", "long hair", "long_hair")):
            conditions.append(StructuredCondition("hair", ("long_hair",)))

        if any(token in normalized for token in ("戴帽", "帽子", "hat", "cap")):
            conditions.append(StructuredCondition("hat", (True,)))
        if any(token in normalized for token in ("眼镜", "glasses")):
            conditions.append(StructuredCondition("glasses", (True,)))
        if any(token in normalized for token in ("背包", "书包", "双肩包", "backpack", "bag")):
            conditions.append(StructuredCondition("backpack", (True,)))
        if any(token in normalized for token in ("手机", "打电话", "看手机", "玩手机", "phone")):
            conditions.append(StructuredCondition("holding_phone", (True,)))
        if any(token in normalized for token in ("抽烟", "吸烟", "smoking", "smoke")):
            conditions.append(StructuredCondition("smoking", (True,)))
        if any(token in normalized for token in ("跌倒", "摔倒", "倒地", "fall", "fallen")):
            conditions.append(StructuredCondition("falling", (True,)))
        if any(token in normalized for token in ("打架", "斗殴", "互殴", "fight", "fighting")):
            conditions.append(StructuredCondition("fighting", (True,)))

        color = self._query_upper_color(normalized)
        if color:
            conditions.append(StructuredCondition("upper_color", (color,)))
        lower_color = self._query_lower_color(normalized)
        if lower_color:
            conditions.append(StructuredCondition("lower_color", (lower_color,)))
        for field, value in self._query_lengths(normalized):
            conditions.append(StructuredCondition(field, (value,)))
        facing = self._query_facing(normalized)
        if facing:
            conditions.append(StructuredCondition("facing", (facing,)))
        stature = self._query_stature(normalized)
        if stature:
            conditions.append(StructuredCondition("stature", (stature,)))
        return conditions

    def _query_upper_color(self, query: str) -> str | None:
        color_tokens = (
            ("white", ("白衣", "白上衣", "白色上衣", "白色衣服", "white shirt", "white coat")),
            ("black", ("黑衣", "黑上衣", "黑色上衣", "黑色衣服", "black shirt", "black coat")),
            ("red", ("红衣", "红上衣", "红色上衣", "红色衣服", "red shirt", "red coat")),
            (
                "yellow",
                ("黄衣", "黄上衣", "黄色上衣", "黄色衣服", "yellow shirt", "yellow coat"),
            ),
            ("blue", ("蓝衣", "蓝上衣", "蓝色上衣", "蓝色衣服", "blue shirt", "blue coat")),
            (
                "green",
                ("绿衣", "绿上衣", "绿色上衣", "绿色衣服", "green shirt", "green coat"),
            ),
            ("gray", ("灰衣", "灰上衣", "灰色上衣", "灰色衣服", "grey shirt", "gray shirt")),
            (
                "brown",
                (
                    "棕衣",
                    "棕色上衣",
                    "褐色上衣",
                    "棕色衣服",
                    "褐色衣服",
                    "brown shirt",
                    "brown coat",
                ),
            ),
            ("orange", ("橙衣", "橙色上衣", "橙色衣服", "orange shirt", "orange coat")),
            ("purple", ("紫衣", "紫色上衣", "紫色衣服", "purple shirt", "purple coat")),
            ("pink", ("粉衣", "粉色上衣", "粉色衣服", "pink shirt", "pink coat")),
            # What the pixel reader falls back to when saturation is too low to name a hue.
            # Without these the only searchable answer it produces is unsearchable.
            (
                "light",
                ("浅色上衣", "浅色衣服", "浅上衣", "白色系上衣", "light top", "light shirt"),
            ),
            ("dark", ("深色上衣", "深色衣服", "深上衣", "dark top", "dark shirt")),
        )
        for color, tokens in color_tokens:
            if any(token in query for token in tokens):
                return color
        return None

    @staticmethod
    def _query_lengths(query: str) -> list[tuple[str, str]]:
        """Sleeve and trouser length. Note there is no "长袖": the extractor never claims one."""

        found = []
        for field, value, tokens in (
            ("upper_length", "short", ("短袖", "半袖", "t恤", "short sleeve")),
            ("lower_length", "short", ("短裤", "短裙", "shorts", "short trousers")),
            ("lower_length", "long", ("长裤", "长裙", "long trousers", "long pants")),
        ):
            if any(token in query for token in tokens):
                found.append((field, value))
        return found

    @staticmethod
    def _query_stature(query: str) -> str | None:
        if any(token in query for token in ("高个", "个子高", "大高个", "身材高", "tall")):
            return "tall"
        if any(token in query for token in ("矮个", "个子矮", "身材矮", "short person")):
            return "short"
        return None

    @staticmethod
    def _query_facing(query: str) -> str | None:
        if any(token in query for token in ("正面", "面向镜头", "facing camera")):
            return "front"
        if any(token in query for token in ("背面", "背影", "背对", "from behind")):
            return "back"
        return None

    def _query_lower_color(self, query: str) -> str | None:
        color_tokens = (
            ("white", ("白裤", "白色裤", "白裙", "white pants", "white trousers")),
            ("black", ("黑裤", "黑色裤", "黑裙", "black pants", "black trousers")),
            ("red", ("红裤", "红色裤", "红裙", "red pants", "red skirt")),
            ("yellow", ("黄裤", "黄色裤", "黄裙", "yellow pants", "yellow skirt")),
            ("blue", ("蓝裤", "蓝色裤", "blue pants", "blue jeans")),
            ("green", ("绿裤", "绿色裤", "green pants")),
            ("gray", ("灰裤", "灰色裤", "grey pants", "gray pants")),
            ("brown", ("棕色裤", "褐色裤", "brown pants")),
            (
                "light",
                ("浅色下装", "浅色裤", "浅裤", "浅色裙", "light pants", "light trousers"),
            ),
            ("dark", ("深色下装", "深色裤", "深裤", "深色裙", "dark pants", "dark trousers")),
        )
        for color, tokens in color_tokens:
            if any(token in query for token in tokens):
                return color
        return None

    def _matches(self, crop: PersonCrop, conditions: list[StructuredCondition]) -> bool:
        return self._match_score(crop, conditions) >= 1.0

    def _match_score(self, crop: PersonCrop, conditions: list[StructuredCondition]) -> float:
        attributes = crop.attributes or {}
        matched = sum(
            self._condition_matches(attributes, crop.bbox or {}, condition)
            for condition in conditions
        )
        if matched != len(conditions):
            return 0.0
        return 1.0

    def _condition_matches(
        self,
        attributes: dict[str, Any],
        bbox: dict[str, Any],
        condition: StructuredCondition,
    ) -> bool:
        detector_label = str(bbox.get("label") or "").strip().lower()
        label_fields = {
            "phone": "holding_phone",
            "smoking": "smoking",
            "falling": "falling",
            "fallen": "falling",
            "fighting": "fighting",
        }
        if label_fields.get(detector_label) == condition.field and True in condition.values:
            return True
        if not self._confidence_allows(attributes, condition.field):
            return False
        values = self._attribute_values(attributes, bbox, condition.field)
        if not values:
            return False
        expected = set(condition.values)
        return any(self._normalize_value(value) in expected for value in values)

    def _attribute_values(
        self,
        attributes: dict[str, Any],
        bbox: dict[str, Any],
        field: str,
    ) -> list[object]:
        paths = {
            "hair": (("appearance", "hair"), ("hair",)),
            "hat": (("appearance", "hat"), ("hat",)),
            "glasses": (("appearance", "glasses"), ("glasses",)),
            "upper_color": (("clothing", "upper_color"), ("upper_color",)),
            "lower_color": (("clothing", "lower_color"), ("lower_color",)),
            "upper_length": (("clothing", "upper_length"), ("upper_length",)),
            "lower_length": (("clothing", "lower_length"), ("lower_length",)),
            "facing": (("facing",),),
            "stature": (("stature", "band"),),
            "backpack": (("objects", "backpack"), ("has_backpack",), ("backpack",)),
            "holding_phone": (
                ("objects", "holding_phone"),
                ("behavior", "looking_at_phone"),
                ("holding_phone",),
            ),
            "smoking": (("behavior", "smoking"), ("objects", "cigarette"), ("smoking",)),
            "falling": (
                ("behavior", "falling"),
                ("behavior", "fallen"),
                ("behavior", "lying_on_ground"),
                ("falling",),
            ),
            "fighting": (
                ("behavior", "fighting"),
                ("behavior", "physical_conflict"),
                ("fighting",),
            ),
        }
        values: list[object] = []
        for path in paths.get(field, ((field,),)):
            value = self._nested_value(attributes, path)
            normalized = self._normalize_value(value)
            if normalized is not None and normalized != "unknown":
                values.append(value)
        if field == "holding_phone" and bbox.get("label") == "phone":
            values.append(True)
        if field == "smoking" and bbox.get("label") == "smoking":
            values.append(True)
        return values

    def _confidence_allows(self, attributes: dict[str, Any], field: str) -> bool:
        paths = {
            "hair": (("appearance", "hair_confidence"), ("hair_confidence",)),
            "hat": (("appearance", "hat_confidence"), ("hat_confidence",)),
            "glasses": (("appearance", "glasses_confidence"), ("glasses_confidence",)),
            "upper_color": (
                ("clothing", "upper_color_confidence"),
                ("upper_color_confidence",),
            ),
            "lower_color": (
                ("clothing", "lower_color_confidence"),
                ("lower_color_confidence",),
            ),
            "backpack": (("objects", "backpack_confidence"), ("backpack_confidence",)),
            "holding_phone": (
                ("objects", "holding_phone_confidence"),
                ("behavior", "looking_at_phone_confidence"),
                ("holding_phone_confidence",),
            ),
            "smoking": (
                ("behavior", "smoking_confidence"),
                ("objects", "cigarette_confidence"),
                ("smoking_confidence",),
            ),
            "falling": (
                ("behavior", "falling_confidence"),
                ("behavior", "fallen_confidence"),
                ("behavior", "lying_on_ground_confidence"),
                ("falling_confidence",),
            ),
            "fighting": (
                ("behavior", "fighting_confidence"),
                ("behavior", "physical_conflict_confidence"),
                ("fighting_confidence",),
            ),
        }
        confidences: list[float] = []
        for path in paths.get(field, ()):
            value = self._nested_value(attributes, path)
            if isinstance(value, int | float):
                confidences.append(float(value))
        if not confidences:
            return True
        return max(confidences) >= STRUCTURED_CONFIDENCE_MIN

    def _nested_value(self, data: dict[str, Any], path: tuple[str, ...]) -> object | None:
        current: object = data
        for key in path:
            if not isinstance(current, dict) or key not in current:
                return None
            current = current[key]
        return current

    def _normalize_value(self, value: object) -> object | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value.strip().lower()
        return value
