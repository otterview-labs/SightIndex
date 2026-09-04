import base64
import json
import mimetypes
from pathlib import Path
from typing import Any
from urllib import error, request

from app.config.settings import Settings
from app.services.label_catalog import LABEL_CATALOG, normalize_labels


class VLMRuntimeError(RuntimeError):
    pass


class VLMCaptionService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def is_enabled(self) -> bool:
        return (
            self.settings.vlm_caption_on_index
            and self.settings.vlm_provider.lower() == "openai_compatible"
            and bool(self.settings.vlm_base_url)
            and bool(self.settings.vlm_model)
        )

    def caption_image(
        self,
        image_path: Path,
        object_type: str,
        metadata_text: str,
    ) -> str | None:
        if not self.is_enabled():
            return None
        if not image_path.exists():
            return None

        data_url = self._image_data_url(image_path)
        object_hint = "人物裁剪图" if object_type == "person_crop" else "监控整图"
        payload = {
            "model": self.settings.vlm_model,
            "messages": [
                {
                    "role": "system",
                    "content": self.settings.vlm_caption_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"对象类型：{object_hint}\n"
                                f"已有元信息：{metadata_text}\n"
                                "请输出一段 80 字以内的检索描述。"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        },
                    ],
                },
            ],
            "temperature": self.settings.vlm_temperature,
            "max_tokens": self.settings.vlm_max_tokens,
        }
        url = self.settings.vlm_base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.settings.vlm_api_key:
            headers["Authorization"] = f"Bearer {self.settings.vlm_api_key}"
        req = request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.settings.vlm_timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (OSError, error.URLError, json.JSONDecodeError) as exc:
            raise VLMRuntimeError(f"VLM caption request failed: {exc}") from exc

        content = self._extract_message_content(data)
        return content.strip() if content else None

    def _image_data_url(self, image_path: Path) -> str:
        mime_type = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
        try:
            encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        except OSError as exc:
            raise VLMRuntimeError(f"Could not read image for VLM caption: {exc}") from exc
        return f"data:{mime_type};base64,{encoded}"

    def _extract_message_content(self, data: dict[str, object]) -> str:
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise VLMRuntimeError("VLM response did not include choices")
        choice = choices[0]
        if not isinstance(choice, dict):
            raise VLMRuntimeError("VLM response choice is invalid")
        message = choice.get("message")
        if not isinstance(message, dict):
            raise VLMRuntimeError("VLM response did not include a message")
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            return "\n".join(parts)
        raise VLMRuntimeError("VLM response content is invalid")


class VLMStructuredAnalysisService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def is_enabled(self) -> bool:
        return (
            self.settings.vlm_provider.lower() == "openai_compatible"
            and bool(self.settings.vlm_base_url)
            and bool(self.settings.vlm_model)
        )

    def analyze_image(
        self,
        image_path: Path,
        object_type: str,
        bbox: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        if not self.is_enabled():
            raise VLMRuntimeError("VLM structured analysis is not configured")
        if object_type not in {"person", "vehicle"}:
            raise VLMRuntimeError(f"Unsupported structured object type: {object_type}")
        if not image_path.exists():
            raise VLMRuntimeError(f"Image does not exist: {image_path}")

        payload = {
            "model": self.settings.vlm_model,
            "messages": [
                {
                    "role": "system",
                    "content": self.settings.vlm_structured_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self._analysis_instruction(object_type, bbox)},
                        {
                            "type": "image_url",
                            "image_url": {"url": self._image_data_url(image_path)},
                        },
                    ],
                },
            ],
            "temperature": self.settings.vlm_temperature,
            "max_tokens": self.settings.vlm_structured_max_tokens,
            "response_format": {"type": "json_object"},
        }
        data = self._request_chat_completion(payload)
        content = VLMCaptionService(self.settings)._extract_message_content(data)
        parsed = self._parse_json_content(content)
        return self._normalize_attributes(object_type, parsed)

    def _request_chat_completion(self, payload: dict[str, object]) -> dict[str, object]:
        url = self.settings.vlm_base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.settings.vlm_api_key:
            headers["Authorization"] = f"Bearer {self.settings.vlm_api_key}"
        req = request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.settings.vlm_timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (OSError, error.URLError, json.JSONDecodeError) as exc:
            raise VLMRuntimeError(f"VLM structured analysis request failed: {exc}") from exc
        if not isinstance(data, dict):
            raise VLMRuntimeError("VLM structured analysis response is invalid")
        return data

    def _analysis_instruction(
        self,
        object_type: str,
        bbox: dict[str, object] | None,
    ) -> str:
        bbox_text = json.dumps(bbox or {}, ensure_ascii=False)
        if object_type == "vehicle":
            return (
                "请解析图中的车辆特征。"
                f"如果提供 bbox，请优先分析 bbox 区域：{bbox_text}。\n"
                "输出 JSON 格式："
                '{"object_type":"vehicle","vehicle_color":"blue|black|white|gray|red|'
                'green|yellow|orange|brown|golden|unknown","vehicle_type":"sedan|suv|van|'
                'hatchback|mpv|pickup|bus|truck|estate|unknown","vehicle_brand":null,'
                '"plate_color":null,"confidence":0.0,"notes":""}'
            )
        return (
            "请解析图中的人体特征。"
            f"如果提供 bbox，请优先分析 bbox 区域：{bbox_text}。\n"
            "必须尽量判断上衣颜色、下装颜色、是否背包、是否戴眼镜、是否戴帽、"
            "是否拿手机/玩手机、是否抽烟、是否跌倒/倒地、是否打架/肢体冲突。"
            "看不清时填 unknown/null/false，并把对应 confidence 设低。\n"
            "输出 JSON 格式："
            '{"object_type":"person","appearance":{"hair":"bald|shaved|short_hair|'
            'long_hair|unknown","hat":false,"glasses":false,"gender":"male|female|unknown",'
            '"age_group":"child|adult|elderly|unknown","hair_confidence":0.0,'
            '"hat_confidence":0.0,"glasses_confidence":0.0},'
            '"clothing":{"upper_color":"blue|black|white|gray|red|green|yellow|orange|'
            'brown|purple|pink|unknown",'
            '"lower_color":"blue|black|white|gray|red|green|yellow|orange|brown|purple|'
            'pink|unknown","upper_type":null,"lower_type":null,"upper_color_confidence":0.0,'
            '"lower_color_confidence":0.0},'
            '"objects":{"backpack":false,"holding_phone":false,"cigarette":false,'
            '"backpack_confidence":0.0,"holding_phone_confidence":0.0,'
            '"cigarette_confidence":0.0},'
            '"behavior":{"smoking":false,"looking_at_phone":false,"falling":false,'
            '"lying_on_ground":false,"fighting":false,"physical_conflict":false,'
            '"smoking_confidence":0.0,"looking_at_phone_confidence":0.0,'
            '"falling_confidence":0.0,"lying_on_ground_confidence":0.0,'
            '"fighting_confidence":0.0,"physical_conflict_confidence":0.0},'
            '"confidence":0.0,"notes":""}'
        )

    def _image_data_url(self, image_path: Path) -> str:
        mime_type = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
        try:
            encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        except OSError as exc:
            raise VLMRuntimeError(f"Could not read image for structured analysis: {exc}") from exc
        return f"data:{mime_type};base64,{encoded}"

    def _parse_json_content(self, content: str) -> dict[str, Any]:
        text = content.strip()
        if text.startswith("```"):
            text = text.strip("`").strip()
            if text.lower().startswith("json"):
                text = text[4:].strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end <= start:
                raise VLMRuntimeError("VLM structured analysis did not return JSON") from None
            try:
                parsed = json.loads(text[start : end + 1])
            except json.JSONDecodeError as exc:
                raise VLMRuntimeError(f"VLM structured JSON is invalid: {exc}") from exc
        if not isinstance(parsed, dict):
            raise VLMRuntimeError("VLM structured JSON root must be an object")
        return parsed

    def _normalize_attributes(
        self,
        object_type: str,
        attributes: dict[str, Any],
    ) -> dict[str, Any]:
        if object_type == "vehicle":
            return self._normalize_vehicle_attributes(attributes)
        return self._normalize_person_attributes(attributes)

    def _normalize_person_attributes(self, attributes: dict[str, Any]) -> dict[str, Any]:
        appearance = self._dict_value(attributes.get("appearance"))
        clothing = self._dict_value(attributes.get("clothing"))
        objects = self._dict_value(attributes.get("objects"))
        behavior = self._dict_value(attributes.get("behavior"))

        top_color = self._string_or_none(
            attributes.get("top_color") or clothing.get("upper_color")
        )
        bottom_color = self._string_or_none(
            attributes.get("bottom_color") or clothing.get("lower_color")
        )
        has_glasses = self._bool_or_none(
            attributes.get("has_glasses")
            if "has_glasses" in attributes
            else appearance.get("glasses")
        )
        has_hat = self._bool_or_none(
            attributes.get("has_hat") if "has_hat" in attributes else appearance.get("hat")
        )
        has_backpack = self._bool_or_none(
            attributes.get("has_backpack")
            if "has_backpack" in attributes
            else objects.get("backpack")
        )
        # Treat weak model guesses as unknown instead of indexing false positives. Confidence
        # fields are optional in older VLM responses, so only gate when one is present.
        top_color = self._keep_if_confident(top_color, clothing.get("upper_color_confidence"))
        bottom_color = self._keep_if_confident(
            bottom_color, clothing.get("lower_color_confidence")
        )
        has_hat = self._keep_bool_if_confident(has_hat, appearance.get("hat_confidence"))
        has_glasses = self._keep_bool_if_confident(
            has_glasses, appearance.get("glasses_confidence")
        )
        has_backpack = self._keep_bool_if_confident(
            has_backpack, objects.get("backpack_confidence")
        )

        normalized = {
            **attributes,
            "object_type": "person",
            "appearance": {
                **appearance,
                "hair": self._string_or_unknown(appearance.get("hair")),
                "hat": has_hat,
                "glasses": has_glasses,
                "gender": self._string_or_unknown(
                    attributes.get("gender") or appearance.get("gender")
                ),
                "age_group": self._string_or_unknown(
                    attributes.get("age_group") or appearance.get("age_group")
                ),
            },
            "clothing": {
                **clothing,
                "upper_color": top_color or "unknown",
                "lower_color": bottom_color or "unknown",
            },
            "objects": {
                **objects,
                "backpack": has_backpack,
                "holding_phone": self._bool_or_none(objects.get("holding_phone")),
                "cigarette": self._bool_or_none(objects.get("cigarette")),
            },
            "behavior": {
                **behavior,
                "smoking": self._keep_bool_if_confident(
                    self._bool_or_none(behavior.get("smoking")), behavior.get("smoking_confidence")
                ),
                "looking_at_phone": self._bool_or_none(behavior.get("looking_at_phone")),
                "falling": self._bool_or_none(
                    behavior.get("falling")
                    if "falling" in behavior
                    else behavior.get("fallen")
                ),
                "lying_on_ground": self._bool_or_none(behavior.get("lying_on_ground")),
                "fighting": self._keep_bool_if_confident(
                    self._bool_or_none(behavior.get("fighting")),
                    behavior.get("fighting_confidence"),
                ),
                "physical_conflict": self._keep_bool_if_confident(
                    self._bool_or_none(behavior.get("physical_conflict")),
                    behavior.get("physical_conflict_confidence"),
                ),
            },
            "top_color": top_color,
            "bottom_color": bottom_color,
            "gender": self._string_or_none(attributes.get("gender") or appearance.get("gender")),
            "age_group": self._string_or_none(
                attributes.get("age_group") or appearance.get("age_group")
            ),
            "has_glasses": has_glasses,
            "has_hat": has_hat,
            "has_backpack": has_backpack,
        }
        return normalized

    def _keep_if_confident(self, value: str | None, confidence: object) -> str | None:
        if value is None or confidence is None:
            return value
        return (
            value
            if self._confidence(confidence) >= self.settings.vlm_structured_min_confidence
            else None
        )

    def _keep_bool_if_confident(self, value: bool | None, confidence: object) -> bool | None:
        if value is None or confidence is None:
            return value
        return (
            value
            if self._confidence(confidence) >= self.settings.vlm_structured_min_confidence
            else None
        )

    def _confidence(self, value: object) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.0

    def _normalize_vehicle_attributes(self, attributes: dict[str, Any]) -> dict[str, Any]:
        vehicle = self._dict_value(attributes.get("vehicle"))
        color = self._string_or_none(
            attributes.get("vehicle_color") or attributes.get("color") or vehicle.get("color")
        )
        vehicle_type = self._string_or_none(
            attributes.get("vehicle_type") or attributes.get("type") or vehicle.get("type")
        )
        return {
            **attributes,
            "object_type": "vehicle",
            "vehicle_color": color or "unknown",
            "vehicle_type": vehicle_type or "unknown",
            "vehicle": {
                **vehicle,
                "color": color or "unknown",
                "type": vehicle_type or "unknown",
            },
        }

    def _dict_value(self, value: object) -> dict[str, Any]:
        return value if isinstance(value, dict) else {}

    def _string_or_none(self, value: object) -> str | None:
        if not isinstance(value, str):
            return None
        text = value.strip().lower()
        return text or None

    def _string_or_unknown(self, value: object) -> str:
        return self._string_or_none(value) or "unknown"

    def _bool_or_none(self, value: object) -> bool | None:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "yes", "1", "是", "有"}:
                return True
            if normalized in {"false", "no", "0", "否", "无"}:
                return False
        return None


class VLMSceneSummaryService:
    _TAG_GROUPS = ("target", "attribute", "behavior", "status", "scene")

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._structured_service = VLMStructuredAnalysisService(settings)

    def is_enabled(self) -> bool:
        return self._structured_service.is_enabled()

    def summarize_image(
        self,
        image_path: Path,
        context: str | None = None,
    ) -> dict[str, Any]:
        if not self.is_enabled():
            raise VLMRuntimeError("VLM scene summary is not configured")
        if not image_path.exists():
            raise VLMRuntimeError(f"Image does not exist: {image_path}")

        payload = {
            "model": self.settings.vlm_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是交通监控图像标签结构化解析器。请只输出合法 JSON，"
                        "不要输出 Markdown。中文只用于 title 和 description；"
                        "tags 内的值必须使用英文小写 snake_case。"
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self._summary_instruction(context)},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": self._structured_service._image_data_url(image_path)
                            },
                        },
                    ],
                },
            ],
            "temperature": self.settings.vlm_temperature,
            "max_tokens": self.settings.vlm_structured_max_tokens,
            "response_format": {"type": "json_object"},
        }
        data = self._structured_service._request_chat_completion(payload)
        content = VLMCaptionService(self.settings)._extract_message_content(data)
        parsed = self._structured_service._parse_json_content(content)
        return self._normalize_summary(parsed)

    def summarize_labels(self, image_path: Path) -> list[str]:
        """Keep the original labels-only scene-summary contract available."""
        if not self.is_enabled():
            raise VLMRuntimeError("VLM scene summary is not configured")
        if not image_path.exists():
            raise VLMRuntimeError(f"Image does not exist: {image_path}")

        payload = {
            "model": self.settings.vlm_model,
            "messages": [
                {
                    "role": "system",
                    "content": self._legacy_label_prompt(),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "请识别画面中可确认的人员、车辆、动物检索标签。"
                                "只输出 JSON，格式为 {\"labels\":[\"labelField:value\"]}。"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": self._structured_service._image_data_url(image_path)
                            },
                        },
                    ],
                },
            ],
            "temperature": 0,
            "max_tokens": self.settings.vlm_max_tokens,
        }
        data = self._structured_service._request_chat_completion(payload)
        content = VLMCaptionService(self.settings)._extract_message_content(data)
        parsed = self._structured_service._parse_json_content(content)
        return normalize_labels(parsed.get("labels"))

    def _legacy_label_prompt(self) -> str:
        label_lines = [
            f"{field}:{value}"
            for field, values in LABEL_CATALOG.items()
            for value in values
        ]
        return (
            "你是视觉检索标签标注器。只允许输出 JSON，不要输出自然语言解释。"
            "只输出画面中能够确认的属性；无法确认时不要猜测。"
            "labelField 必须使用下划线命名，只能从以下标签中选择：\n"
            + "\n".join(label_lines)
        )

    def _summary_instruction(self, context: str | None) -> str:
        context_text = f"业务场景：{context}\n" if context else ""
        return (
            f"{context_text}"
            "请分析图像中的车辆、人员、行为、状态和场景，返回以下 JSON 结构："
            '{"title":"一句中文标题","description":"一段中文描述",'
            '"tags":{"target":[],"attribute":[],"behavior":[],"status":[],"scene":[]},'
            '"persons":[{"role":"driver|passenger|pedestrian|unknown",'
            '"tags":{"attribute":[],"behavior":[],"status":[]}}],'
            '"count":{"persons":0,"vehicles":0},"confidence":0.0}'
            "。如果能识别品牌、车型、车牌颜色、安全带、打电话、灯光、时间、道路等，"
            "请放到对应 tags 中。看不清时不要编造。"
        )

    def _normalize_summary(self, summary: dict[str, Any]) -> dict[str, Any]:
        tags = self._normalize_tags(summary.get("tags"))
        persons = self._normalize_persons(summary.get("persons"))
        count = self._normalize_count(summary.get("count"), persons, tags)
        normalized: dict[str, Any] = {
            "title": self._string_value(summary.get("title")),
            "description": self._string_value(summary.get("description")),
            "tags": tags,
            "persons": persons,
            "count": count,
        }
        confidence = self._confidence_value(summary.get("confidence"))
        if confidence is not None:
            normalized["confidence"] = confidence
        return normalized

    def _normalize_tags(self, value: object) -> dict[str, list[str]]:
        source = value if isinstance(value, dict) else {}
        return {
            group: self._string_list(source.get(group))
            for group in self._TAG_GROUPS
        }

    def _normalize_persons(self, value: object) -> list[dict[str, object]]:
        if not isinstance(value, list):
            return []
        persons: list[dict[str, object]] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            persons.append(
                {
                    "role": self._string_value(item.get("role"), default="unknown"),
                    "tags": self._normalize_person_tags(item.get("tags")),
                }
            )
        return persons

    def _normalize_person_tags(self, value: object) -> dict[str, list[str]]:
        source = value if isinstance(value, dict) else {}
        return {
            group: self._string_list(source.get(group))
            for group in ("attribute", "behavior", "status")
        }

    def _normalize_count(
        self,
        value: object,
        persons: list[dict[str, object]],
        tags: dict[str, list[str]],
    ) -> dict[str, int]:
        source = value if isinstance(value, dict) else {}
        vehicles = self._int_value(source.get("vehicles"))
        if vehicles == 0 and any(tag in tags["target"] for tag in ("vehicle", "car", "suv")):
            vehicles = 1
        return {
            "persons": self._int_value(source.get("persons"), default=len(persons)),
            "vehicles": vehicles,
        }

    def _string_list(self, value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        tags: list[str] = []
        for item in value:
            if not isinstance(item, str):
                continue
            normalized = item.strip().lower().replace("-", "_").replace(" ", "_")
            if normalized and normalized not in tags:
                tags.append(normalized)
        return tags

    def _string_value(self, value: object, default: str = "") -> str:
        if not isinstance(value, str):
            return default
        return value.strip() or default

    def _int_value(self, value: object, default: int = 0) -> int:
        if isinstance(value, bool):
            return default
        if isinstance(value, int):
            return max(value, 0)
        if isinstance(value, float):
            return max(int(value), 0)
        return default

    def _confidence_value(self, value: object) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return max(0.0, min(float(value), 1.0))
