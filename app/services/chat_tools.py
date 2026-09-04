from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, time
from typing import Any, Protocol
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config.settings import Settings
from app.models.events import RecognitionEvent
from app.models.media import Image, PersonCrop, VideoStream
from app.models.persons import Person
from app.models.vectors import FaceEmbedding
from app.schemas.chat import ChatResponse, ChatToolCall
from app.schemas.common import SearchFilters
from app.schemas.media import ImageSearchRequest, SearchResponse, VisualSearchRequest
from app.schemas.persons import FaceSearchResponse
from app.services.faces import FaceRecognitionService
from app.services.persons import PersonService
from app.services.search import StructuredCondition, StructuredSearchService, VisualSearchService
from app.services.statistics import StatisticsService


@dataclass
class ChatToolContext:
    db: Session
    settings: Settings
    search: VisualSearchService
    structured_search: StructuredSearchService
    statistics: StatisticsService
    request_context: dict[str, Any]

    def local_timezone(self) -> ZoneInfo:
        try:
            return ZoneInfo(self.settings.local_timezone)
        except ZoneInfoNotFoundError:
            return ZoneInfo("Asia/Shanghai")

    def message_start_time(self, message: str) -> datetime | None:
        if "今天" not in message and "today" not in message.lower():
            return None
        today = datetime.now(self.local_timezone()).date()
        return datetime.combine(today, time.min, tzinfo=self.local_timezone())

    def mentioned_person(self, message: str) -> Person | None:
        people = list(self.db.scalars(select(Person).order_by(Person.created_at.desc()).limit(200)))
        for person in people:
            if person.name and person.name in message:
                return person
            if person.employee_no and person.employee_no in message:
                return person
        return None


class ChatTool(Protocol):
    name: str

    def can_handle(self, message: str, context: ChatToolContext) -> bool:
        ...

    def run(self, message: str, context: ChatToolContext) -> ChatResponse:
        ...


class StreamStatusTool:
    name = "stream_status"

    def can_handle(self, message: str, context: ChatToolContext) -> bool:
        lower_message = message.lower()
        return any(token in lower_message for token in ("视频流", "stream", "camera", "摄像头"))

    def run(self, message: str, context: ChatToolContext) -> ChatResponse:
        total = context.db.scalar(select(func.count()).select_from(VideoStream)) or 0
        running = (
            context.db.scalar(
                select(func.count())
                .select_from(VideoStream)
                .where(VideoStream.status.in_(["running", "starting"]))
            )
            or 0
        )
        streams = list(
            context.db.scalars(select(VideoStream).order_by(VideoStream.updated_at.desc()).limit(8))
        )
        items = [
            {
                "stream_id": str(stream.id),
                "name": stream.name,
                "status": stream.status,
                "last_error": stream.last_error,
                "started_at": stream.started_at.isoformat() if stream.started_at else None,
                "updated_at": stream.updated_at.isoformat() if stream.updated_at else None,
                "title": f"{stream.name} · {stream.status}",
            }
            for stream in streams
        ]
        result = {"total": total, "running": running, "items": items}
        return ChatResponse(
            answer=f"当前共有 {total} 路视频流，其中 {running} 路处于运行或启动中。",
            data=result,
            tool_name=self.name,
            tool_params={"limit": 8},
            tool_calls=[ChatToolCall(name=self.name, params={"limit": 8}, result=result)],
        )


class PersonTrajectoryTool:
    name = "person_trajectory"

    def can_handle(self, message: str, context: ChatToolContext) -> bool:
        if context.mentioned_person(message) is None:
            return False
        lower_message = message.lower()
        tokens = ("轨迹", "在哪", "在哪里", "出现", "经过", "看到", "trajectory", "where")
        return any(token in lower_message for token in tokens)

    def run(self, message: str, context: ChatToolContext) -> ChatResponse:
        person = context.mentioned_person(message)
        if person is None:
            return ChatResponse(answer="没有匹配到人员。", tool_name=self.name)

        mode = self._trajectory_mode(message)
        start_time = context.message_start_time(message)
        points = PersonService(context.db, context.settings).trajectory(
            person=person,
            limit=8,
            start_time=start_time,
            mode=mode,
        )
        items: list[dict[str, Any]] = []
        for point in points:
            items.append(_trajectory_point_item(point))

        person_profile = _person_profile(person)
        mode_text = {"all": "人脸 + 向量", "face": "人脸", "vector": "向量"}[mode]
        time_scope = "今天" if start_time else "最近"
        if not items:
            answer = f"没有查到 {person.name} 的{time_scope}{mode_text}轨迹。"
        else:
            latest = _format_local_datetime(points[0].recognized_at, context)
            location_text = _trajectory_location_text(points[0])
            location_clause = f"，地点：{location_text}" if location_text else ""
            profile_clause = _person_profile_clause(person_profile)
            answer = (
                f"查到 {person.name}{profile_clause} 的{time_scope}{mode_text}轨迹 "
                f"{len(items)} 条；最近一次在 {latest}{location_clause}，"
                f"来源 {_match_source_label(points[0].match_source)}。"
            )
        params = {
            "person_id": str(person.id),
            "person_name": person.name,
            "limit": 8,
            "mode": mode,
            "start_time": start_time.isoformat() if start_time else None,
        }
        result = {"items": items, "person": person_profile}
        return ChatResponse(
            answer=answer,
            data=result,
            tool_name=self.name,
            tool_params=params,
            tool_calls=[ChatToolCall(name=self.name, params=params, result=result)],
        )

    def _trajectory_mode(self, message: str) -> str:
        lower_message = message.lower()
        if "向量" in lower_message or "vector" in lower_message:
            return "vector"
        if "人脸" in lower_message or "face" in lower_message:
            return "face"
        return "all"


class PersonAttendanceTool:
    name = "person_attendance"

    def can_handle(self, message: str, context: ChatToolContext) -> bool:
        lower_message = message.lower()
        if self._asks_earliest_person(message):
            return True
        attendance_tokens = (
            "上班",
            "到岗",
            "到公司",
            "来了",
            "来了吗",
            "有没有来",
            "几点来",
            "几点到",
            "最早",
            "第一次",
            "最后一次",
            "出现了几次",
            "出现几次",
            "多少次",
            "attendance",
            "arrival",
            "arrive",
            "check in",
            "clock in",
            "work time",
        )
        if not any(token in lower_message for token in attendance_tokens):
            return False
        return context.mentioned_person(message) is not None or _looks_like_named_person_query(
            message
        )

    def run(self, message: str, context: ChatToolContext) -> ChatResponse:
        if self._asks_earliest_person(message):
            return self._run_earliest_person(message, context)

        person = context.mentioned_person(message)
        if person is None:
            return ChatResponse(answer="没有匹配到人员。", tool_name=self.name)

        intent = self._intent(message)
        start_time = self._start_time(message, context)
        service = PersonService(context.db, context.settings)
        points = service.trajectory(
            person=person,
            limit=200,
            start_time=start_time,
            mode="face",
        )
        mode = "face"
        if not points and self._should_use_vector_fallback(message):
            points = service.trajectory(
                person=person,
                limit=200,
                start_time=start_time,
                mode="all",
            )
            mode = "all"
        ascending_points = sorted(points, key=lambda point: point.recognized_at)
        descending_points = sorted(points, key=lambda point: point.recognized_at, reverse=True)
        if intent == "latest":
            ordered_points = descending_points
            selected = ordered_points[0] if ordered_points else None
        else:
            ordered_points = ascending_points
            selected = ordered_points[0] if ordered_points else None

        person_profile = _person_profile(person)
        items = [_trajectory_point_item(point) for point in ordered_points[:20]]
        source = _trajectory_source(points)
        time_scope = "今天" if start_time else "最近"
        result = {
            "person": person_profile,
            "time_scope": time_scope,
            "source": source,
            "appearance_count": len(points),
            "selected_appearance": _trajectory_point_item(selected) if selected else None,
            "first_appearance": (
                _trajectory_point_item(ascending_points[0]) if ascending_points else None
            ),
            "last_appearance": (
                _trajectory_point_item(descending_points[0]) if descending_points else None
            ),
            "items": items,
        }
        params = {
            "person_id": str(person.id),
            "person_name": person.name,
            "mode": mode,
            "limit": 200,
            "intent": intent,
            "start_time": start_time.isoformat() if start_time else None,
        }
        if selected is None:
            answer = (
                f"没有查到 {person.name}{_person_profile_clause(person_profile)} "
                f"{time_scope}的到岗或出现记录。"
            )
        elif intent == "count":
            profile_clause = _person_profile_clause(person_profile)
            answer = (
                f"{person.name}{profile_clause} {time_scope}共检索到 "
                f"{len(points)} 条出现记录；"
                f"来源：{_source_label(source)}。"
            )
        elif intent == "presence":
            first_time = _format_local_datetime(selected.recognized_at, context)
            location_text = _trajectory_location_text(selected)
            location_clause = f"地点：{location_text}，" if location_text else ""
            profile_clause = _person_profile_clause(person_profile)
            answer = (
                f"{person.name}{profile_clause} {time_scope}有出现记录，"
                f"首次出现时间是 {first_time}；{location_clause}"
                f"来源：{_match_source_label(selected.match_source)}，"
                f"共检索到 {len(points)} 条轨迹。"
            )
        elif intent == "latest":
            latest_time = _format_local_datetime(selected.recognized_at, context)
            location_text = _trajectory_location_text(selected)
            location_clause = f"地点：{location_text}，" if location_text else ""
            profile_clause = _person_profile_clause(person_profile)
            answer = (
                f"{person.name}{profile_clause} {time_scope}最后一次出现时间是 "
                f"{latest_time}；{location_clause}"
                f"来源：{_match_source_label(selected.match_source)}，"
                f"共检索到 {len(points)} 条轨迹。"
            )
        else:
            first_time = _format_local_datetime(selected.recognized_at, context)
            location_text = _trajectory_location_text(selected)
            location_clause = f"地点：{location_text}，" if location_text else ""
            profile_clause = _person_profile_clause(person_profile)
            answer = (
                f"{person.name}{profile_clause} {time_scope}最早出现时间是 "
                f"{first_time}；{location_clause}"
                f"来源：{_match_source_label(selected.match_source)}，"
                f"共检索到 {len(points)} 条轨迹。"
            )
        return ChatResponse(
            answer=answer,
            data=result,
            tool_name=self.name,
            tool_params=params,
            tool_calls=[ChatToolCall(name=self.name, params=params, result=result)],
        )

    def _run_earliest_person(self, message: str, context: ChatToolContext) -> ChatResponse:
        start_time = self._start_time(message, context)
        people = list(
            context.db.scalars(select(Person).order_by(Person.created_at.desc()).limit(50))
        )
        candidates: list[tuple[datetime, Person, Any, list[Any]]] = []
        service = PersonService(context.db, context.settings)
        for person in people:
            points = service.trajectory(
                person=person,
                limit=100,
                start_time=start_time,
                mode="face",
            )
            if not points:
                continue
            earliest = min(points, key=lambda point: point.recognized_at)
            candidates.append((earliest.recognized_at, person, earliest, points))

        candidates.sort(key=lambda item: item[0])
        time_scope = "今天" if start_time else "最近"
        items = []
        for _recognized_at, person, point, points in candidates[:10]:
            item = _trajectory_point_item(point)
            item["appearance_count"] = len(points)
            item["source"] = _trajectory_source(points)
            item["person"] = _person_profile(person)
            items.append(item)

        result = {
            "time_scope": time_scope,
            "items": items,
            "appearance_count": sum(len(points) for _time, _person, _point, points in candidates),
        }
        params = {
            "mode": "face",
            "limit": 100,
            "people_limit": 50,
            "intent": "earliest_person",
            "start_time": start_time.isoformat() if start_time else None,
        }
        if not candidates:
            answer = f"没有查到任何人员{time_scope}的到岗记录。"
        else:
            _recognized_at, person, point, points = candidates[0]
            first_time = _format_local_datetime(point.recognized_at, context)
            profile = _person_profile(person)
            location_text = _trajectory_location_text(point)
            location_clause = f"地点：{location_text}，" if location_text else ""
            answer = (
                f"{time_scope}最早到岗的是 {person.name}{_person_profile_clause(profile)}，"
                f"时间是 {first_time}；{location_clause}"
                f"来源：{_match_source_label(point.match_source)}，"
                f"该人员共检索到 {len(points)} 条轨迹。"
            )
        return ChatResponse(
            answer=answer,
            data=result,
            tool_name=self.name,
            tool_params=params,
            tool_calls=[ChatToolCall(name=self.name, params=params, result=result)],
        )

    def _asks_earliest_person(self, message: str) -> bool:
        lower_message = message.lower()
        return ("谁" in message or "who" in lower_message) and any(
            token in lower_message
            for token in ("最早到岗", "最早上班", "最早来", "first arrival", "earliest")
        )

    def _intent(self, message: str) -> str:
        lower_message = message.lower()
        if any(token in lower_message for token in ("最后", "最近一次", "last", "下班")):
            return "latest"
        if any(token in lower_message for token in ("几次", "多少次", "count")):
            return "count"
        if any(token in lower_message for token in ("有没有", "来了吗", "到了吗", "到岗了吗")):
            return "presence"
        return "earliest"

    def _should_use_vector_fallback(self, message: str) -> bool:
        lower_message = message.lower()
        return any(
            token in lower_message
            for token in ("向量", "vector", "人脸 + 向量", "人脸+向量")
        )

    def _start_time(self, message: str, context: ChatToolContext) -> datetime | None:
        if start_time := context.message_start_time(message):
            return start_time
        lower_message = message.lower()
        today_tokens = (
            "上班",
            "到岗",
            "来了吗",
            "有没有来",
            "几点来",
            "几点到",
            "today",
            "attendance",
            "arrival",
            "check in",
            "clock in",
            "work time",
        )
        if any(token in lower_message for token in today_tokens):
            today = datetime.now(context.local_timezone()).date()
            return datetime.combine(today, time.min, tzinfo=context.local_timezone())
        return None


class RecentRecognitionsTool:
    name = "recent_recognitions"

    def can_handle(self, message: str, context: ChatToolContext) -> bool:
        lower_message = message.lower()
        tokens = ("最近识别", "识别到谁", "谁出现", "最近出现", "recent recognition")
        return any(token in lower_message for token in tokens)

    def run(self, message: str, context: ChatToolContext) -> ChatResponse:
        start_time = context.message_start_time(message)
        stmt = (
            select(RecognitionEvent, Person, PersonCrop, Image)
            .join(Person, RecognitionEvent.person_id == Person.id, isouter=True)
            .join(PersonCrop, RecognitionEvent.crop_id == PersonCrop.id, isouter=True)
            .join(Image, RecognitionEvent.image_id == Image.id, isouter=True)
            .order_by(RecognitionEvent.recognized_at.desc())
            .limit(8)
        )
        if start_time:
            stmt = stmt.where(RecognitionEvent.recognized_at >= start_time)
        items = []
        for event, person, crop, image in context.db.execute(stmt):
            source = "face" if event.person_id else event.result_type
            name = person.name if person else "未知"
            items.append(
                {
                    "event_id": str(event.id),
                    "person_id": str(person.id) if person else None,
                    "person_name": name,
                    "crop_id": str(crop.id) if crop else None,
                    "image_id": str(image.id) if image else None,
                    "crop_url": crop.crop_url if crop else None,
                    "image_url": image.thumbnail_url or image.image_url if image else None,
                    "score": float(event.similarity) if event.similarity is not None else None,
                    "similarity": (
                        float(event.similarity) if event.similarity is not None else None
                    ),
                    "result_type": event.result_type,
                    "match_source": source,
                    "recognized_at": event.recognized_at.isoformat(),
                    "title": f"{name} · {event.result_type}",
                }
            )
        time_scope = "今天" if start_time else "最近"
        answer = (
            f"{time_scope}识别事件返回 {len(items)} 条；最新一条是 {items[0]['person_name']}。"
            if items
            else f"{time_scope}还没有识别事件。"
        )
        result = {"items": items}
        params = {"limit": 8, "start_time": start_time.isoformat() if start_time else None}
        return ChatResponse(
            answer=answer,
            data=result,
            tool_name=self.name,
            tool_params=params,
            tool_calls=[ChatToolCall(name=self.name, params=params, result=result)],
        )


class FaceSummaryTool:
    name = "face_summary"

    def can_handle(self, message: str, context: ChatToolContext) -> bool:
        lower_message = message.lower()
        tokens = ("人脸库", "人脸", "face", "known", "unknown", "陌生人")
        return any(token in lower_message for token in tokens)

    def run(self, message: str, context: ChatToolContext) -> ChatResponse:
        person_count = context.db.scalar(select(func.count()).select_from(Person)) or 0
        face_count = context.db.scalar(select(func.count()).select_from(FaceEmbedding)) or 0
        known_events = (
            context.db.scalar(
                select(func.count())
                .select_from(RecognitionEvent)
                .where(RecognitionEvent.person_id.is_not(None))
            )
            or 0
        )
        unknown_events = (
            context.db.scalar(
                select(func.count())
                .select_from(RecognitionEvent)
                .where(RecognitionEvent.result_type.in_(["unknown", "no_face"]))
            )
            or 0
        )
        rows = context.db.execute(
            select(Person, func.count(FaceEmbedding.id))
            .join(FaceEmbedding, FaceEmbedding.person_id == Person.id, isouter=True)
            .group_by(Person.id)
            .order_by(Person.created_at.desc())
            .limit(8)
        )
        people = [
            {
                "person_id": str(person.id),
                "person_name": person.name,
                "employee_no": person.employee_no,
                "face_count": int(face_total or 0),
                "title": f"{person.name} · {int(face_total or 0)} 张人脸",
            }
            for person, face_total in rows
        ]
        result = {
            "person_count": person_count,
            "face_count": face_count,
            "known_event_count": known_events,
            "unknown_event_count": unknown_events,
            "items": people,
        }
        return ChatResponse(
            answer=(
                f"人脸库当前有 {person_count} 个人、{face_count} 张已登记人脸；"
                f"历史已命中 known 事件 {known_events} 条，"
                f"unknown/no_face 事件 {unknown_events} 条。"
            ),
            data=result,
            tool_name=self.name,
            tool_params={"limit": 8},
            tool_calls=[ChatToolCall(name=self.name, params={"limit": 8}, result=result)],
        )


class FaceImageSearchTool:
    name = "face_image_search"

    def can_handle(self, message: str, context: ChatToolContext) -> bool:
        if _uuid_from_context(context.request_context.get("last_image_id")) is None:
            return False
        lower_message = message.lower()
        identity_tokens = (
            "这是谁",
            "是谁",
            "像谁",
            "识别",
            "认一下",
            "人脸",
            "脸",
            "face",
            "who",
        )
        image_tokens = ("这张", "刚才", "上传", "图片", "照片", "image", "photo")
        return any(token in lower_message for token in identity_tokens) and any(
            token in lower_message for token in image_tokens
        )

    def run(self, message: str, context: ChatToolContext) -> ChatResponse:
        image_id = _uuid_from_context(context.request_context.get("last_image_id"))
        if image_id is None:
            return ChatResponse(
                answer="还没有可用于人脸检索的最近上传图片。",
                data={"items": [], "result_type": "missing_image"},
                tool_name=self.name,
                tool_params={"last_image_id": None},
                tool_calls=[
                    ChatToolCall(
                        name=self.name,
                        params={"last_image_id": None},
                        result={"items": [], "result_type": "missing_image"},
                    )
                ],
            )

        image = context.db.get(Image, image_id)
        if image is None:
            result = {"items": [], "result_type": "image_not_found", "image_id": str(image_id)}
            return ChatResponse(
                answer="最近上传图片记录不存在，无法做人脸检索。",
                data=result,
                tool_name=self.name,
                tool_params={"image_id": str(image_id)},
                tool_calls=[
                    ChatToolCall(
                        name=self.name,
                        params={"image_id": str(image_id)},
                        result=result,
                    )
                ],
            )

        response = FaceRecognitionService(context.db, context.settings).search_image(
            image=image,
            top_k=5,
            min_similarity=None,
            allow_fallback=context.settings.face_fallback_to_full_image,
        )
        data = _face_search_response_data(response, context)
        params = {"image_id": str(image.id), "top_k": 5}
        if response.face_bbox is None:
            answer = "人脸工具没有在最近上传图片里检测到可用人脸。"
        elif not response.matches:
            answer = "人脸工具检测到了人脸，但人脸库为空或没有同维度可比样本。"
        else:
            best = response.matches[0]
            threshold = context.settings.face_match_threshold
            if best.similarity >= threshold:
                answer = (
                    f"人脸工具 Top1 命中 {best.person_name}，"
                    f"相似度 {best.similarity:.2f}，超过当前阈值 {threshold:.2f}。"
                )
            else:
                answer = (
                    f"人脸工具 Top1 是 {best.person_name}，相似度 {best.similarity:.2f}，"
                    f"低于当前阈值 {threshold:.2f}，只能作为候选参考。"
                )
        return ChatResponse(
            answer=answer,
            data=data,
            tool_name=self.name,
            tool_params=params,
            tool_calls=[ChatToolCall(name=self.name, params=params, result=data)],
        )


class CountEventsTool:
    name = "count_events"

    def can_handle(self, message: str, context: ChatToolContext) -> bool:
        lower_message = message.lower()
        is_count = any(token in lower_message for token in ("几个", "多少", "count", "how many"))
        return is_count and not _is_visual_question(message)

    def run(self, message: str, context: ChatToolContext) -> ChatResponse:
        start_time = context.message_start_time(message)
        summary = context.statistics.count_summary(start_time=start_time)
        return ChatResponse(
            answer=(
                f"今天检测到人体裁剪 {summary.person_crop_count} 个，"
                f"画面帧 {summary.image_count} 张；"
                f"过线有效计数 {summary.counting_event_count} 次，"
                f"已识别独立人员 {summary.unique_person_count} 人，"
                f"陌生人 {summary.unique_unknown_count} 人。"
            ),
            data=summary.model_dump(),
            tool_name=self.name,
            tool_params={"start_time": start_time.isoformat() if start_time else None},
            tool_calls=[
                ChatToolCall(
                    name=self.name,
                    params={"start_time": start_time.isoformat() if start_time else None},
                    result=summary.model_dump(),
                )
            ],
        )


class VisualImageSearchTool:
    name = "visual_image_search"

    def can_handle(self, message: str, context: ChatToolContext) -> bool:
        lower_message = message.lower()
        image_tokens = ("刚才", "这张图", "这张图片", "上传", "图片", "image")
        search_tokens = ("相似", "类似", "找", "搜", "search", "similar")
        return any(token in lower_message for token in image_tokens) and any(
            token in lower_message for token in search_tokens
        )

    def run(self, message: str, context: ChatToolContext) -> ChatResponse:
        image_id = _uuid_from_context(context.request_context.get("last_image_id"))
        if image_id is None:
            return ChatResponse(
                answer="还没有可用于以图搜图的最近上传图片。先在右侧上传一张图片再问相似目标。",
                data={"items": []},
                tool_name=self.name,
                tool_params={"last_image_id": None},
                tool_calls=[
                    ChatToolCall(
                        name=self.name,
                        params={"last_image_id": None},
                        result={"items": [], "error": "missing_last_image_id"},
                    )
                ],
            )

        start_time = context.message_start_time(message)
        filters = SearchFilters(start_time=start_time)
        request = ImageSearchRequest(
            image_id=image_id,
            top_k=8,
            target="person_crop",
            filters=filters,
        )
        result = context.search.search_by_image(request)
        call = ChatToolCall(
            name=self.name,
            params=request.model_dump(mode="json"),
            result=result.model_dump(mode="json"),
        )
        top_score = max((item.score for item in result.items), default=None)
        score_text = f"，最高相似度 {top_score:.2f}" if top_score is not None else ""
        answer = f"已按最近上传图片做以图搜图，返回 {len(result.items)} 个相似候选{score_text}。"
        data = result.model_dump(mode="json")
        data["source"] = self.name
        data["tool_calls"] = [call.model_dump(mode="json")]
        return ChatResponse(
            answer=answer,
            data=data,
            tool_name=self.name,
            tool_params=request.model_dump(mode="json"),
            tool_calls=[call],
        )


class VisualTextSearchTool:
    name = "visual_text_search"

    def can_handle(self, message: str, context: ChatToolContext) -> bool:
        return True

    def run(self, message: str, context: ChatToolContext) -> ChatResponse:
        start_time = context.message_start_time(message)
        filters = SearchFilters(start_time=start_time)
        request = VisualSearchRequest(
            query=message,
            top_k=8,
            target="person_crop",
            filters=filters,
            rerank=False,
        )
        structured_result, conditions = context.structured_search.search(
            message,
            top_k=request.top_k,
            filters=filters,
        )
        structured_call = _structured_tool_call(
            message,
            request.top_k,
            filters,
            conditions,
            structured_result,
        )
        if conditions:
            data = _search_response_data(
                structured_result,
                source="structured_labels",
                conditions=conditions,
                tool_calls=[structured_call],
                vector_status=None,
            )
            match_count = len(structured_result.items)
            return ChatResponse(
                answer=(
                    f"已把“{message}”解析成结构化条件 "
                    f"{_conditions_text(conditions)}，"
                    f"返回 {match_count} 个同时满足全部标签的结果。"
                ),
                data=data,
                tool_name="search_structured",
                tool_params=structured_call.params,
                tool_calls=[structured_call],
            )

        result = context.search.search(request)
        tag_call = ChatToolCall(
            name=self.name,
            params=request.model_dump(mode="json"),
            result=result.model_dump(mode="json"),
        )
        tool_calls = [tag_call]
        time_scope = "今天" if start_time else "当前范围"
        if result.items:
            answer = (
                f"已按人员或标签关键词检索“{message}”，"
                f"{time_scope}返回 {len(result.items)} 个结果。"
            )
        else:
            answer = (
                f"没有解析出可命中的标准标签，也没有找到人员关键词“{message}”。"
                "当前支持衣服颜色、裤装颜色、帽子、眼镜、背包、手机、抽烟、"
                "跌倒、打架、发型、朝向和身材等明确标签。"
            )
        return ChatResponse(
            answer=answer,
            data=_search_response_data(
                result,
                source="label_keyword",
                conditions=[],
                tool_calls=tool_calls,
                vector_status=None,
            ),
            tool_name=self.name,
            tool_params=request.model_dump(mode="json"),
            tool_calls=tool_calls,
        )


def default_chat_tools() -> list[ChatTool]:
    return [
        StreamStatusTool(),
        PersonAttendanceTool(),
        PersonTrajectoryTool(),
        RecentRecognitionsTool(),
        FaceImageSearchTool(),
        FaceSummaryTool(),
        CountEventsTool(),
        VisualTextSearchTool(),
    ]


def _trajectory_point_item(point: Any) -> dict[str, Any]:
    item = point.model_dump(mode="json")
    item["title"] = f"{point.person_name} · {point.match_source}"
    item["score"] = point.similarity if point.similarity is not None else point.vector_score
    item["location"] = _trajectory_location(point)
    return item


def _person_profile(person: Person) -> dict[str, Any]:
    tags = _person_tags(person)
    return {
        "id": str(person.id),
        "name": person.name,
        "employee_no": person.employee_no,
        "department": person.department,
        "phone": person.phone,
        "avatar_url": person.avatar_url,
        "status": person.status,
        "tags": tags,
        "label_text": " / ".join(tag["value"] for tag in tags),
    }


def _person_tags(person: Person) -> list[dict[str, str]]:
    fields = (
        ("name", "姓名", person.name),
        ("employee_no", "工号", person.employee_no),
        ("department", "部门", person.department),
        ("phone", "手机号", person.phone),
        ("status", "状态", person.status),
    )
    return [
        {"key": key, "label": label, "value": str(value)}
        for key, label, value in fields
        if value
    ]


def _person_profile_clause(profile: dict[str, Any]) -> str:
    label_text = profile.get("label_text")
    return f"（{label_text}）" if label_text else ""


def _looks_like_named_person_query(message: str) -> bool:
    for token in ("今天", "最近", "上班", "到岗", "出现", "轨迹", "在哪", "在哪里"):
        message = message.replace(token, "")
    return any("\u4e00" <= char <= "\u9fff" for char in message)


def _trajectory_location(point: Any) -> dict[str, Any]:
    return {
        "camera_id": str(point.camera_id) if point.camera_id else None,
        "location_id": str(point.location_id) if point.location_id else None,
        "text": _trajectory_location_text(point),
    }


def _trajectory_location_text(point: Any) -> str | None:
    parts = []
    if point.location_id:
        parts.append(f"位置 {point.location_id}")
    if point.camera_id:
        parts.append(f"摄像头 {point.camera_id}")
    return " / ".join(parts) if parts else None


def _trajectory_source(points: list[Any]) -> str:
    has_face = any("face" in point.match_source for point in points)
    has_vector = any("vector" in point.match_source for point in points)
    if has_face and has_vector:
        return "face + vector"
    if has_face:
        return "face"
    if has_vector:
        return "vector"
    return "none"


def _source_label(source: str) -> str:
    labels = {
        "face + vector": "人脸 + 向量",
        "face": "人脸",
        "vector": "向量",
        "none": "无",
    }
    return labels.get(source, source)


def _match_source_label(source: str) -> str:
    labels = {
        "face_vector": "人脸 + 向量",
        "face": "人脸",
        "vector": "向量",
    }
    return labels.get(source, source)


def _format_local_datetime(value: datetime, context: ChatToolContext) -> str:
    if value.tzinfo is not None:
        value = value.astimezone(context.local_timezone())
    return value.strftime("%Y-%m-%d %H:%M:%S")


def _structured_tool_call(
    query: str,
    top_k: int,
    filters: SearchFilters,
    conditions: list[StructuredCondition],
    result: SearchResponse,
) -> ChatToolCall:
    return ChatToolCall(
        name="search_structured",
        params={
            "query": query,
            "top_k": top_k,
            "filters": filters.model_dump(mode="json"),
            "conditions": [condition.as_dict() for condition in conditions],
        },
        result=result.model_dump(mode="json"),
    )


def _search_response_data(
    result: SearchResponse,
    source: str,
    conditions: list[StructuredCondition] | None = None,
    tool_calls: list[ChatToolCall] | None = None,
    vector_status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data = result.model_dump(mode="json")
    data["source"] = source
    if vector_status is not None:
        data["vector_status"] = vector_status
    if conditions:
        data["conditions"] = [condition.as_dict() for condition in conditions]
    if tool_calls:
        data["tool_calls"] = [tool_call.model_dump(mode="json") for tool_call in tool_calls]
    return data


def _face_search_response_data(
    response: FaceSearchResponse,
    context: ChatToolContext,
) -> dict[str, Any]:
    threshold = context.settings.face_match_threshold
    if response.face_bbox is None:
        result_type = "no_face"
    elif response.matches and response.matches[0].similarity >= threshold:
        result_type = "known"
    else:
        result_type = "unknown"

    items: list[dict[str, Any]] = []
    for match in response.matches:
        crop = context.db.get(PersonCrop, match.crop_id) if match.crop_id else None
        image = context.db.get(Image, match.image_id) if match.image_id else None
        person = context.db.get(Person, match.person_id)
        face_url = None
        if crop:
            face_url = crop.crop_url
        elif image:
            face_url = image.thumbnail_url or image.image_url
        elif person:
            face_url = person.avatar_url
        items.append(
            {
                "person_id": str(match.person_id),
                "person_name": match.person_name,
                "face_embedding_id": str(match.face_embedding_id),
                "similarity": match.similarity,
                "score": match.similarity,
                "quality_score": match.quality_score,
                "image_id": str(match.image_id) if match.image_id else None,
                "crop_id": str(match.crop_id) if match.crop_id else None,
                "crop_url": crop.crop_url if crop else None,
                "image_url": image.thumbnail_url or image.image_url if image else None,
                "avatar_url": person.avatar_url if person else None,
                "face_url": face_url,
                "title": f"{match.person_name} · face {match.similarity:.2f}",
            }
        )
    return {
        "image_id": str(response.image_id) if response.image_id else None,
        "face_bbox": response.face_bbox,
        "threshold": threshold,
        "result_type": result_type,
        "items": items,
        "matches": [match.model_dump(mode="json") for match in response.matches],
    }


def _conditions_text(conditions: list[StructuredCondition]) -> str:
    return "、".join(f"{condition.field} in {list(condition.values)}" for condition in conditions)


def _visual_search_mode_text(settings: Settings) -> str:
    visual_provider = settings.visual_embedding_provider.lower()
    if settings.milvus_enabled and visual_provider != "none":
        return "视觉向量检索"
    if settings.milvus_enabled and settings.embedding_provider.lower() != "none":
        return "元信息向量检索"
    return "标签/最近裁剪兜底检索"


def _vector_status(settings: Settings) -> dict[str, Any]:
    visual_provider = settings.visual_embedding_provider.lower()
    text_provider = settings.embedding_provider.lower()
    return {
        "milvus_enabled": settings.milvus_enabled,
        "visual_embedding_provider": visual_provider,
        "text_embedding_provider": text_provider,
        "visual_vector_enabled": settings.milvus_enabled and visual_provider != "none",
        "text_vector_enabled": settings.milvus_enabled and text_provider != "none",
        "mode": _visual_search_mode_text(settings),
    }


def _is_visual_question(message: str) -> bool:
    lower_message = message.lower()
    visual_tokens = (
        "衣服",
        "外套",
        "裤",
        "裙",
        "颜色",
        "白色",
        "黑色",
        "红色",
        "黄色",
        "蓝色",
        "绿色",
        "灰色",
        "棕色",
        "褐色",
        "眼镜",
        "帽子",
        "戴帽",
        "光头",
        "秃头",
        "没有头发",
        "短发",
        "长发",
        "背包",
        "手机",
        "打电话",
        "抽烟",
        "吸烟",
        "phone",
        "smoking",
        "glasses",
    )
    return any(token in lower_message for token in visual_tokens)


def _uuid_from_context(value: object) -> uuid.UUID | None:
    if isinstance(value, uuid.UUID):
        return value
    if not isinstance(value, str):
        return None
    try:
        return uuid.UUID(value)
    except ValueError:
        return None
