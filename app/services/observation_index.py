from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import func, or_, select, update
from sqlalchemy.orm import Session

from app.config.settings import Settings
from app.models.events import RecognitionEvent
from app.models.media import Image, PersonCrop, PersonObservationIndex, VideoStream
from app.models.persons import Person
from app.models.vectors import FaceEmbedding, VectorIndexCapacityLock, VLEmbedding
from app.schemas.common import SearchFilters
from app.services.time_utils import database_datetime


@dataclass(frozen=True)
class ObservationSearchHit:
    row: PersonObservationIndex
    score: float


class ObservationIndexService:
    def __init__(self, db: Session, settings: Settings) -> None:
        self.db = db
        self.settings = settings

    def upsert_crop(self, crop: PersonCrop) -> PersonObservationIndex:
        self._lock_writes_in_session()
        # The session runs with autoflush=False, and callers upsert right after creating the
        # recognition event in the same transaction. Without a flush the SELECT below cannot
        # see that event, and the row is written with a NULL similarity that nothing ever
        # corrects unless some later pipeline happens to upsert the same crop again.
        self.db.flush()
        image = self.db.get(Image, crop.image_id)
        person = self.db.get(Person, crop.person_id) if crop.person_id else None
        event = self._latest_recognition_event(crop.id)
        if event and event.person_id and (person is None or person.id != event.person_id):
            person = self.db.get(Person, event.person_id)
        face_embedding = self._face_embedding(person.id if person else None, crop.id)
        vl_embedding = self._vl_embedding(crop.id)
        stream = self._stream(crop.camera_id or (image.camera_id if image else None))

        row = self.db.scalar(
            select(PersonObservationIndex).where(PersonObservationIndex.crop_id == crop.id)
        )
        if row is None:
            row = PersonObservationIndex(crop_id=crop.id)

        attributes = crop.attributes or {}
        labels_zh = attribute_labels(attributes, "zh")
        labels_en = attribute_labels(attributes, "en")
        # captured_at is a camera/recognition fact, not an ingest timestamp. crop.created_at is
        # SQLite UTC-naive (CURRENT_TIMESTAMP), while camera timestamps are local-naive; mixing
        # the two made time-window queries unknowable. Keep NULL when neither source exists.
        captured_at = (
            crop.captured_at
            or (image.captured_at if image else None)
            or (event.recognized_at if event else None)
        )
        camera_id = crop.camera_id or (image.camera_id if image else None)
        location_id = crop.location_id or (image.location_id if image else None)
        row.image_id = crop.image_id
        row.person_id = person.id if person else None
        row.person_name = person.name if person else None
        row.employee_no = person.employee_no if person else None
        row.department = person.department if person else None
        row.recognition_result_type = event.result_type if event else None
        row.face_similarity = (
            float(event.similarity) if event and event.similarity is not None else None
        )
        row.face_confidence = (
            float(event.confidence) if event and event.confidence is not None else None
        )
        row.face_embedding_id = face_embedding.id if face_embedding else None
        row.face_embedding_model = face_embedding.face_model if face_embedding else None
        row.camera_id = camera_id
        row.camera_name = stream.name if stream else (str(camera_id) if camera_id else None)
        row.location_id = location_id
        row.location_name = (
            (stream.location_name if stream and stream.location_name else None)
            or (str(location_id) if location_id else None)
        )
        row.captured_at = captured_at
        row.image_url = image.thumbnail_url or image.image_url if image else None
        row.crop_url = crop.crop_url
        row.thumbnail_url = image.thumbnail_url if image else None
        row.bbox = crop.bbox
        row.attributes = attributes or None
        row.labels_zh = labels_zh or None
        row.labels_en = labels_en or None
        row.vl_embedding_id = vl_embedding.id if vl_embedding else None
        row.vl_embedding_model = vl_embedding.embedding_model if vl_embedding else None
        row.vl_embedding_dim = vl_embedding.embedding_dim if vl_embedding else None
        row.milvus_collection = self._milvus_collection("person_crop") if vl_embedding else None
        row.milvus_object_id = str(crop.id) if vl_embedding else None
        row.has_face_embedding = face_embedding is not None
        row.has_vl_embedding = vl_embedding is not None
        row.search_text = self._search_text(row)
        self.db.add(row)
        return row

    def _lock_writes_in_session(self) -> None:
        """Serialize the unique crop-row upsert across face, VL and ReID workers."""

        result = self.db.execute(
            update(VectorIndexCapacityLock)
            .where(VectorIndexCapacityLock.target == "observation_index")
            .values(revision=VectorIndexCapacityLock.revision + 1)
            .execution_options(synchronize_session=False)
        )
        if result.rowcount != 1:
            raise RuntimeError("Observation index lock is missing; run init_db()")

    def rebuild(self, limit: int = 1000) -> dict[str, object]:
        crops = list(
            self.db.scalars(select(PersonCrop).order_by(PersonCrop.created_at.desc()).limit(limit))
        )
        indexed = 0
        errors: list[str] = []
        for crop in crops:
            try:
                self.upsert_crop(crop)
                indexed += 1
            except Exception as exc:
                errors.append(f"{crop.id}: {exc}")
        self.db.commit()
        return {"requested": limit, "seen": len(crops), "indexed": indexed, "errors": errors}

    def search(
        self,
        query: str,
        *,
        top_k: int,
        filters: SearchFilters,
    ) -> list[ObservationSearchHit]:
        query_text = query.strip()
        stmt = select(PersonObservationIndex).order_by(
            PersonObservationIndex.captured_at.desc(),
            PersonObservationIndex.created_at.desc(),
        )
        stmt = self._apply_filters(stmt, filters)
        predicates = self._query_predicates(query_text)
        if predicates:
            stmt = stmt.where(or_(*predicates))
        rows = list(self.db.scalars(stmt.limit(max(top_k * 5, top_k))))
        hits = [
            ObservationSearchHit(row=row, score=self._score_row(row, query_text))
            for row in rows
        ]
        hits = [hit for hit in hits if hit.score > 0 or not predicates]
        hits.sort(
            key=lambda hit: (
                hit.score,
                hit.row.captured_at.timestamp() if hit.row.captured_at else 0.0,
                hit.row.created_at.timestamp() if hit.row.created_at else 0.0,
            ),
            reverse=True,
        )
        return hits[:top_k]

    def rows_by_crop_ids(
        self,
        crop_ids: list[uuid.UUID],
        filters: SearchFilters,
    ) -> dict[uuid.UUID, PersonObservationIndex]:
        if not crop_ids:
            return {}
        stmt = select(PersonObservationIndex).where(PersonObservationIndex.crop_id.in_(crop_ids))
        stmt = self._apply_filters(stmt, filters)
        return {row.crop_id: row for row in self.db.scalars(stmt)}

    def list_rows(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        query: str | None = None,
        filters: SearchFilters | None = None,
        only_named: bool = False,
        only_face_vector: bool = False,
        only_vl_vector: bool = False,
        only_labeled: bool = False,
    ) -> tuple[list[PersonObservationIndex], int]:
        filters = filters or SearchFilters()
        stmt = select(PersonObservationIndex)
        stmt = self._apply_filters(stmt, filters)
        stmt = self._apply_list_filters(
            stmt,
            query=query,
            only_named=only_named,
            only_face_vector=only_face_vector,
            only_vl_vector=only_vl_vector,
            only_labeled=only_labeled,
        )
        total = self.db.scalar(
            select(func.count()).select_from(stmt.order_by(None).subquery())
        ) or 0
        rows = list(
            self.db.scalars(
                stmt.order_by(
                    PersonObservationIndex.captured_at.desc(),
                    PersonObservationIndex.created_at.desc(),
                )
                .offset(offset)
                .limit(limit)
            )
        )
        return rows, total

    def _query_predicates(self, query: str) -> list[Any]:
        if not query:
            return []
        like = f"%{query}%"
        predicates: list[Any] = [
            PersonObservationIndex.person_name.ilike(like),
            PersonObservationIndex.employee_no.ilike(like),
            PersonObservationIndex.department.ilike(like),
            PersonObservationIndex.camera_name.ilike(like),
            PersonObservationIndex.location_name.ilike(like),
            PersonObservationIndex.search_text.ilike(like),
        ]
        for token in self._query_tokens(query):
            token_like = f"%{token}%"
            predicates.append(PersonObservationIndex.search_text.ilike(token_like))
        return predicates

    def _score_row(self, row: PersonObservationIndex, query: str) -> float:
        if not query:
            return 0.1
        score = 0.0
        lowered = query.lower()
        if row.person_name and row.person_name.lower() in lowered:
            score += 1.5
        if row.employee_no and row.employee_no.lower() in lowered:
            score += 1.2
        search_text = (row.search_text or "").lower()
        labels_text = " ".join(
            str(value).lower()
            for labels in (row.labels_zh or {}, row.labels_en or {})
            for value in labels.values()
        )
        for token in self._query_tokens(query):
            if token.lower() in search_text or token.lower() in labels_text:
                score += 0.3
        if row.person_name:
            score += 0.2
        if row.attributes:
            score += 0.1
        return score

    def _apply_filters(self, stmt: Any, filters: SearchFilters) -> Any:
        if filters.person_id:
            stmt = stmt.where(PersonObservationIndex.person_id == filters.person_id)
        if filters.camera_id:
            stmt = stmt.where(PersonObservationIndex.camera_id == filters.camera_id)
        if filters.location_id:
            stmt = stmt.where(PersonObservationIndex.location_id == filters.location_id)
        if filters.start_time:
            stmt = stmt.where(
                PersonObservationIndex.captured_at
                >= database_datetime(
                    filters.start_time,
                    self.settings,
                    self.db.get_bind().dialect.name,
                )
            )
        if filters.end_time:
            stmt = stmt.where(
                PersonObservationIndex.captured_at
                <= database_datetime(
                    filters.end_time,
                    self.settings,
                    self.db.get_bind().dialect.name,
                )
            )
        return stmt

    def _apply_list_filters(
        self,
        stmt: Any,
        *,
        query: str | None,
        only_named: bool,
        only_face_vector: bool,
        only_vl_vector: bool,
        only_labeled: bool,
    ) -> Any:
        query_text = (query or "").strip()
        if query_text:
            predicates = self._query_predicates(query_text)
            if predicates:
                stmt = stmt.where(or_(*predicates))
        if only_named:
            stmt = stmt.where(PersonObservationIndex.person_id.is_not(None))
        if only_face_vector:
            stmt = stmt.where(PersonObservationIndex.has_face_embedding.is_(True))
        if only_vl_vector:
            stmt = stmt.where(PersonObservationIndex.has_vl_embedding.is_(True))
        if only_labeled:
            stmt = stmt.where(PersonObservationIndex.labels_zh.is_not(None))
        return stmt

    def _query_tokens(self, query: str) -> list[str]:
        normalized = query.lower()
        tokens = [
            token
            for token in (
                "黑色",
                "黑衣",
                "黑上衣",
                "黑色上衣",
                "黑色衣服",
                "黑裤",
                "白色",
                "白衣",
                "白上衣",
                "白色上衣",
                "白色衣服",
                "白裤",
                "红色",
                "红衣",
                "红上衣",
                "蓝色",
                "蓝衣",
                "蓝上衣",
                "灰色",
                "灰衣",
                "灰上衣",
                "黄色",
                "黄衣",
                "黄上衣",
                "绿色",
                "绿衣",
                "绿上衣",
                "背包",
                "书包",
                "双肩包",
                "帽子",
                "眼镜",
                "手机",
                "抽烟",
                "打架",
                "跌倒",
                "black",
                "white",
                "red",
                "blue",
                "gray",
                "backpack",
                "hat",
                "glasses",
                "phone",
            )
            if token in normalized
        ]
        tokens.extend(self._semantic_tokens(normalized))
        tokens = list(dict.fromkeys(tokens))
        if not tokens and len(query) <= 32:
            tokens.append(query)
        return tokens

    def _semantic_tokens(self, query: str) -> list[str]:
        tokens: list[str] = []
        for color, aliases in _COLOR_QUERY_ALIASES.items():
            if any(alias in query for alias in aliases):
                tokens.extend([color, _ZH_LABEL_VALUES.get(color, color)])
        if any(alias in query for alias in ("背包", "书包", "双肩包", "backpack", "bag")):
            tokens.extend(["backpack", "背包", "是"])
        if any(alias in query for alias in ("帽子", "戴帽", "hat", "cap")):
            tokens.extend(["hat", "帽子", "是"])
        if any(alias in query for alias in ("眼镜", "glasses")):
            tokens.extend(["glasses", "眼镜", "是"])
        return tokens

    def _latest_recognition_event(self, crop_id: uuid.UUID) -> RecognitionEvent | None:
        return self.db.scalar(
            select(RecognitionEvent)
            .where(RecognitionEvent.crop_id == crop_id)
            .order_by(RecognitionEvent.recognized_at.desc(), RecognitionEvent.created_at.desc())
        )

    def _face_embedding(
        self,
        person_id: uuid.UUID | None,
        crop_id: uuid.UUID,
    ) -> FaceEmbedding | None:
        stmt = select(FaceEmbedding).where(FaceEmbedding.crop_id == crop_id)
        if person_id:
            stmt = stmt.where(FaceEmbedding.person_id == person_id)
        return self.db.scalar(stmt.order_by(FaceEmbedding.created_at.desc()))

    def _vl_embedding(self, crop_id: uuid.UUID) -> VLEmbedding | None:
        return self.db.scalar(
            select(VLEmbedding)
            .where(VLEmbedding.object_type == "person_crop", VLEmbedding.object_id == crop_id)
            .order_by(VLEmbedding.created_at.desc())
        )

    def _stream(self, camera_id: uuid.UUID | None) -> VideoStream | None:
        if camera_id is None:
            return None
        return self.db.scalar(select(VideoStream).where(VideoStream.camera_id == camera_id))

    def _milvus_collection(self, object_type: str) -> str:
        suffix = {"person_crop": "vl_person_crops", "image": "vl_images"}[object_type]
        prefix = self.settings.milvus_collection_prefix.strip("_")
        return f"{prefix}_{suffix}" if prefix else suffix

    def _search_text(self, row: PersonObservationIndex) -> str:
        parts: list[str] = [
            row.person_name or "",
            row.employee_no or "",
            row.department or "",
            row.camera_name or "",
            row.location_name or "",
            row.crop_url or "",
            row.image_url or "",
            row.captured_at.isoformat() if row.captured_at else "",
        ]
        for labels in (row.labels_zh or {}, row.labels_en or {}):
            for key, value in labels.items():
                parts.extend([str(key), str(value)])
        return " ".join(part for part in parts if part)


def attribute_labels(attributes: dict[str, Any], label_language: str) -> dict[str, object]:
    if label_language == "en":
        return _english_attribute_labels(attributes)
    return _chinese_attribute_labels(attributes)


def _chinese_attribute_labels(attributes: dict[str, Any]) -> dict[str, object]:
    object_type = _text(attributes.get("object_type"))
    if object_type == "vehicle":
        return {
            "对象类型": "车辆",
            "车辆颜色": _zh_value(attributes.get("vehicle_color")),
            "车辆类型": _zh_value(attributes.get("vehicle_type")),
            "车辆品牌": _zh_value(attributes.get("vehicle_brand")),
            "车牌颜色": _zh_value(attributes.get("plate_color")),
            "置信度": attributes.get("confidence"),
            "备注": attributes.get("notes") or "",
        }

    appearance = _dict(attributes.get("appearance"))
    clothing = _dict(attributes.get("clothing"))
    objects = _dict(attributes.get("objects"))
    behavior = _dict(attributes.get("behavior"))
    return {
        "对象类型": "人员",
        "发型": _zh_value(appearance.get("hair")),
        "性别": _zh_value(appearance.get("gender")),
        "年龄段": _zh_value(appearance.get("age_group")),
        "上衣颜色": _zh_value(attributes.get("top_color") or clothing.get("upper_color")),
        "下装颜色": _zh_value(attributes.get("bottom_color") or clothing.get("lower_color")),
        # Omitted when unread rather than shown as 未知. These three come off the pose keypoints
        # and each abstains often by design -- sleeve length only ever claims "short" -- so a
        # fixed row would fill the panel with blanks the reader has to scan past.
        **_optional_zh(
            ("袖长", _LENGTH_LABELS["upper"].get(_text(clothing.get("upper_length")))),
            ("裤长", _LENGTH_LABELS["lower"].get(_text(clothing.get("lower_length")))),
            ("朝向", _FACING_LABELS.get(_text(attributes.get("facing")))),
            ("身高", _STATURE_LABELS.get(_text(_dict(attributes.get("stature")).get("band")))),
        ),
        "帽子": _zh_bool(
            attributes.get("has_hat") if "has_hat" in attributes else appearance.get("hat")
        ),
        "眼镜": _zh_bool(
            attributes.get("has_glasses")
            if "has_glasses" in attributes
            else appearance.get("glasses")
        ),
        "背包": _zh_bool(
            attributes.get("has_backpack")
            if "has_backpack" in attributes
            else objects.get("backpack")
        ),
        "拿手机": _zh_bool(objects.get("holding_phone")),
        "抽烟": _zh_bool(behavior.get("smoking")),
        "跌倒": _zh_bool(behavior.get("falling")),
        "倒地": _zh_bool(behavior.get("lying_on_ground")),
        "打架": _zh_bool(behavior.get("fighting") or behavior.get("physical_conflict")),
        "置信度": attributes.get("confidence"),
        "备注": attributes.get("notes") or "",
    }


def _english_attribute_labels(attributes: dict[str, Any]) -> dict[str, object]:
    object_type = _text(attributes.get("object_type"))
    if object_type == "vehicle":
        return {
            "object_type": "vehicle",
            "vehicle_color": _text_or_unknown(attributes.get("vehicle_color")),
            "vehicle_type": _text_or_unknown(attributes.get("vehicle_type")),
            "vehicle_brand": attributes.get("vehicle_brand"),
            "plate_color": attributes.get("plate_color"),
            "confidence": attributes.get("confidence"),
            "notes": attributes.get("notes") or "",
        }

    appearance = _dict(attributes.get("appearance"))
    clothing = _dict(attributes.get("clothing"))
    objects = _dict(attributes.get("objects"))
    behavior = _dict(attributes.get("behavior"))
    return {
        "object_type": "person",
        "hair": _text_or_unknown(appearance.get("hair")),
        "gender": _text_or_unknown(appearance.get("gender")),
        "age_group": _text_or_unknown(appearance.get("age_group")),
        "upper_color": _text_or_unknown(attributes.get("top_color") or clothing.get("upper_color")),
        "lower_color": _text_or_unknown(
            attributes.get("bottom_color") or clothing.get("lower_color")
        ),
        **_optional_zh(
            ("upper_length", _text(clothing.get("upper_length"))),
            ("lower_length", _text(clothing.get("lower_length"))),
            ("facing", _text(attributes.get("facing"))),
            ("stature", _text(_dict(attributes.get("stature")).get("band"))),
        ),
        "hat": _bool_label(
            attributes.get("has_hat") if "has_hat" in attributes else appearance.get("hat")
        ),
        "glasses": _bool_label(
            attributes.get("has_glasses")
            if "has_glasses" in attributes
            else appearance.get("glasses")
        ),
        "backpack": _bool_label(
            attributes.get("has_backpack")
            if "has_backpack" in attributes
            else objects.get("backpack")
        ),
        "holding_phone": _bool_label(objects.get("holding_phone")),
        "smoking": _bool_label(behavior.get("smoking")),
        "falling": _bool_label(behavior.get("falling")),
        "lying_on_ground": _bool_label(behavior.get("lying_on_ground")),
        "fighting": _bool_label(behavior.get("fighting") or behavior.get("physical_conflict")),
        "confidence": attributes.get("confidence"),
        "notes": attributes.get("notes") or "",
    }


def _dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    return text or None


def _text_or_unknown(value: object) -> str:
    return _text(value) or "unknown"


def _zh_bool(value: object) -> str:
    if value is True:
        return "是"
    if value is False:
        return "否"
    return "未知"


def _bool_label(value: object) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "unknown"


# Sleeve and trouser length share the words "short" and "long" but not the garment, so the
# label depends on which one is being read.
_LENGTH_LABELS = {
    "upper": {"short": "短袖", "long": "长袖"},
    "lower": {"short": "短裤/短裙", "long": "长裤"},
}
_FACING_LABELS = {"front": "正面", "back": "背面"}
# Relative to the crowd at this doorway, not to a population this deployment never measured.
_STATURE_LABELS = {"tall": "偏高", "average": "中等", "short": "偏矮"}


def _optional_zh(*pairs: tuple[str, str | None]) -> dict[str, object]:
    return {key: value for key, value in pairs if value}


def _zh_value(value: object) -> str:
    text = _text(value)
    if text is None:
        return "未知"
    return _ZH_LABEL_VALUES.get(text, text)


_ZH_LABEL_VALUES = {
    "unknown": "未知",
    "none": "无",
    "person": "人员",
    "vehicle": "车辆",
    "black": "黑色",
    "white": "白色",
    "gray": "灰色",
    "grey": "灰色",
    # The CV extractor names a hue only when the pixels justify one; on low-saturation footage
    # -- 70% of the crops measured here -- it reports brightness instead of inventing a colour.
    "light": "浅色",
    "dark": "深色",
    "red": "红色",
    "green": "绿色",
    "blue": "蓝色",
    "yellow": "黄色",
    "orange": "橙色",
    "brown": "棕色",
    "purple": "紫色",
    "pink": "粉色",
    "golden": "金色",
    "bald": "光头",
    "shaved": "寸头",
    "short_hair": "短发",
    "long_hair": "长发",
    "male": "男性",
    "female": "女性",
    "child": "儿童",
    "adult": "成人",
    "elderly": "老人",
    "sedan": "轿车",
    "suv": "SUV",
    "van": "面包车",
    "hatchback": "两厢车",
    "mpv": "MPV",
    "pickup": "皮卡",
    "bus": "公交车",
    "truck": "货车",
    "estate": "旅行车",
}


_COLOR_QUERY_ALIASES = {
    "black": (
        "黑色",
        "黑衣",
        "黑上衣",
        "黑色上衣",
        "黑色衣服",
        "黑裤",
        "black",
    ),
    "white": (
        "白色",
        "白衣",
        "白上衣",
        "白色上衣",
        "白色衣服",
        "白裤",
        "white",
    ),
    "red": ("红色", "红衣", "红上衣", "红色上衣", "red"),
    "blue": ("蓝色", "蓝衣", "蓝上衣", "蓝色上衣", "blue"),
    "gray": ("灰色", "灰衣", "灰上衣", "灰色上衣", "grey", "gray"),
    "yellow": ("黄色", "黄衣", "黄上衣", "黄色上衣", "yellow"),
    "green": ("绿色", "绿衣", "绿上衣", "绿色上衣", "green"),
    "brown": ("棕色", "褐色", "棕衣", "brown"),
}
