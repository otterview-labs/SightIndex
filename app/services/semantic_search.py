import hashlib
import math
import uuid
from datetime import UTC

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session

from app.config.settings import Settings
from app.models.media import Image, PersonCrop
from app.models.persons import Person
from app.models.vectors import VLEmbedding
from app.schemas.common import SearchFilters
from app.schemas.semantic_search import (
    SemanticSearchItem,
    SemanticSearchRequest,
    SemanticSearchResponse,
    SemanticSearchStatus,
)
from app.services.embeddings import EmbeddingRuntimeError
from app.services.vector_index import MilvusVectorIndex, VectorIndexError
from app.services.vlm import VLMStructuredAnalysisService

SEMANTIC_NOTICE = (
    "以下仅为视觉语义相似候选，颜色、性别、帽子等条件未逐项核验，"
    "相似度不是准确率，也不能确认人物身份。同内容、同相机且同身份标注的裁剪已合并。"
)


class SemanticSearchUnavailable(RuntimeError):
    pass


class SemanticSearchService:
    def __init__(self, db: Session, settings: Settings) -> None:
        self.db = db
        self.settings = settings
        self.index = MilvusVectorIndex(settings)

    def _configured(self) -> bool:
        return (
            self.settings.milvus_enabled
            and self.settings.milvus_metric_type.upper() == "COSINE"
            and self.settings.visual_embedding_provider.lower()
            in {"qwen3_vl", "qwen3_vl_http"}
            and bool(self.settings.milvus_visual_collection_prefix)
            and self.index.visual_embedding.is_enabled()
        )

    def _indexed(self) -> Select[tuple[uuid.UUID]]:
        return select(VLEmbedding.object_id).where(
            VLEmbedding.object_type == "person_crop",
            VLEmbedding.embedding_model == self.settings.visual_embedding_model,
            VLEmbedding.embedding_dim == self.settings.visual_embedding_dim,
        )

    def status(self) -> SemanticSearchStatus:
        total = self.db.scalar(select(func.count()).select_from(PersonCrop)) or 0
        indexed = self.db.scalar(
            select(func.count()).select_from(PersonCrop).where(PersonCrop.id.in_(self._indexed()))
        ) or 0
        labeled = self.db.scalar(
            select(func.count()).select_from(PersonCrop).where(
                PersonCrop.attributes.is_not(None),
                PersonCrop.attributes["source"].as_string().is_not(None),
            )
        ) or 0
        return SemanticSearchStatus(
            enabled=self.settings.semantic_search_enabled,
            configured=self._configured(),
            model=self.settings.visual_embedding_model,
            min_score=self.settings.semantic_search_min_score,
            total_crops=total,
            indexed_crops=indexed,
            labeled_crops=labeled,
            attributes_enabled=VLMStructuredAnalysisService(self.settings).is_enabled(),
            auto_index_on_ingest=self.settings.vector_index_on_ingest,
        )

    def _scope(self, filters: SearchFilters) -> Select[tuple[uuid.UUID]]:
        if filters.extra:
            raise ValueError("语义检索不支持 extra 标签筛选，请使用严格标签匹配")
        start = filters.start_time
        end = filters.end_time
        if start is not None and end is not None:
            normalized_start = start.replace(tzinfo=UTC) if start.tzinfo is None else start
            normalized_end = end.replace(tzinfo=UTC) if end.tzinfo is None else end
            if normalized_start > normalized_end:
                raise ValueError("开始时间不能晚于结束时间")
        statement = select(PersonCrop.id)
        for field in ("person_id", "camera_id", "location_id"):
            value = getattr(filters, field)
            if value is not None:
                statement = statement.where(getattr(PersonCrop, field) == value)
        if start is not None:
            statement = statement.where(PersonCrop.captured_at >= start)
        if end is not None:
            statement = statement.where(PersonCrop.captured_at <= end)
        return statement

    def _fingerprint(self, url: str) -> str | None:
        if not url.startswith("/data/"):
            return None
        root = self.settings.data_dir.resolve()
        try:
            crop_path = (root / url.removeprefix("/data/")).resolve()
            if not crop_path.is_relative_to(root) or not crop_path.is_file():
                return None
            if crop_path.stat().st_size > 20 * 1024 * 1024:
                return None
            with crop_path.open("rb") as source:
                return hashlib.file_digest(source, "sha256").hexdigest()
        except OSError:
            return None

    def search(self, payload: SemanticSearchRequest) -> SemanticSearchResponse:
        if not self.settings.semantic_search_enabled or not self._configured():
            raise SemanticSearchUnavailable("Qwen 语义检索未启用或尚未正确配置")
        scope = self._scope(payload.filters)
        scope_count = self.db.scalar(
            select(func.count()).select_from(scope.subquery())
        ) or 0
        indexed_scope = scope.where(PersonCrop.id.in_(self._indexed()))
        object_ids = list(self.db.scalars(
            indexed_scope.order_by(PersonCrop.id).limit(self.settings.semantic_search_max_scope + 1)
        ))
        if len(object_ids) > self.settings.semantic_search_max_scope:
            raise ValueError("检索范围过大，请缩小相机或时间范围后重试")
        response = SemanticSearchResponse(
            items=[],
            model=self.settings.visual_embedding_model,
            min_score=self.settings.semantic_search_min_score,
            notice=SEMANTIC_NOTICE,
            scope_crops=scope_count,
            indexed_scope_crops=len(object_ids),
        )
        if not scope_count:
            return response
        if not object_ids:
            raise SemanticSearchUnavailable("筛选范围内尚无当前模型索引，请先补建历史向量")
        if len(object_ids) < scope_count:
            response.notice += f" 当前范围仅 {len(object_ids)}/{scope_count} 个裁剪有索引。"
        candidate_limit = min(len(object_ids), max(50, payload.top_k * 4), 400)
        try:
            hits = self.index.search_text_for_objects(
                "person_crop", payload.query, candidate_limit, object_ids
            )
        except (EmbeddingRuntimeError, VectorIndexError) as exc:
            raise SemanticSearchUnavailable("语义检索服务暂不可用，请稍后重试") from exc
        if not hits:
            raise SemanticSearchUnavailable("向量索引尚未就绪，请检查索引覆盖后重试")
        response.candidates_examined = len(hits)
        response.shortlist_limited = len(object_ids) > candidate_limit
        if response.shortlist_limited:
            response.notice += " 本次仅核对相似度最高的有限候选，去重后可能少于请求数量。"
        rows = self.db.execute(
            select(PersonCrop, Image.image_url, Person.name)
            .join(Image, Image.id == PersonCrop.image_id)
            .outerjoin(Person, Person.id == PersonCrop.person_id)
            .where(
                PersonCrop.id.in_([hit.object_id for hit in hits]),
                PersonCrop.id.in_(object_ids),
                PersonCrop.id.in_(scope),
            )
        )
        crops = {crop.id: (crop, image_url, person_name) for crop, image_url, person_name in rows}
        merged: dict[
            tuple[str, uuid.UUID | None, uuid.UUID | None, uuid.UUID | None],
            SemanticSearchItem,
        ] = {}
        for hit in sorted(hits, key=lambda item: (-item.score, str(item.object_id))):
            if not math.isfinite(hit.score) or hit.score < response.min_score:
                continue
            row = crops.get(hit.object_id)
            if row is None:
                continue
            crop, image_url, person_name = row
            fingerprint = self._fingerprint(crop.crop_url)
            if fingerprint is None:
                continue
            key = (fingerprint, crop.camera_id, crop.location_id, crop.person_id)
            if key in merged:
                merged[key].duplicate_crop_ids.append(crop.id)
                continue
            merged[key] = SemanticSearchItem(
                crop_id=crop.id,
                image_id=crop.image_id,
                image_url=image_url,
                crop_url=crop.crop_url,
                score=hit.score,
                captured_at=crop.captured_at,
                camera_id=crop.camera_id,
                location_id=crop.location_id,
                person_id=crop.person_id,
                person_name=person_name,
                attributes=crop.attributes,
            )
        response.items = list(merged.values())[:payload.top_k]
        return response
