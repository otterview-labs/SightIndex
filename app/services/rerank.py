import base64
import json
import math
import mimetypes
import re
import sys
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib import error, request

from app.config.settings import Settings
from app.schemas.media import SearchResultItem
from app.services.embeddings import (
    EmbeddingRuntimeError,
    VisualEmbeddingService,
    _ensure_torchvision_import_compat,
)
from app.services.vlm import VLMCaptionService, VLMRuntimeError


@dataclass(frozen=True)
class RerankDecision:
    score: float
    matched: bool
    reason: str


def _map_concurrently(
    candidates: Sequence[Any],
    score_candidate: Callable[[Any], Any],
    max_workers: int,
) -> list[Any]:
    """Score candidates in parallel, keeping results in candidate order.

    Reranking one candidate is a network round trip, so a serial loop over the candidate
    limit costs the sum of every call. Exceptions still surface, because ThreadPoolExecutor
    re-raises the first failure when the result iterator is consumed.
    """

    if len(candidates) <= 1 or max_workers <= 1:
        return [score_candidate(candidate) for candidate in candidates]
    with ThreadPoolExecutor(
        max_workers=min(max_workers, len(candidates)),
        thread_name_prefix="rerank",
    ) as pool:
        return list(pool.map(score_candidate, candidates))


class VLMRerankService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def is_enabled(self) -> bool:
        if self.settings.vlm_rerank_service_url:
            return True
        return (
            self.settings.vlm_rerank_enabled
            and
            self.settings.vlm_provider.lower() == "openai_compatible"
            and bool(self.settings.vlm_base_url)
            and bool(self.settings.vlm_model)
        )

    def rerank_person_crops(
        self,
        query: str,
        items: list[SearchResultItem],
        *,
        limit: int,
    ) -> list[SearchResultItem]:
        if not self.is_enabled():
            raise VLMRuntimeError("VLM rerank is not configured")
        candidates: list[tuple[SearchResultItem, Path]] = []
        for item in items[: self.settings.vlm_rerank_candidate_limit]:
            image_path = self._item_image_path(item)
            if image_path is None or not image_path.exists():
                continue
            candidates.append((item, image_path))
        decisions = _map_concurrently(
            candidates,
            lambda candidate: self.rerank_image(query, candidate[1], candidate[0].attributes or {}),
            self.settings.vlm_rerank_max_workers,
        )

        reranked: list[SearchResultItem] = []
        for (item, _), decision in zip(candidates, decisions, strict=True):
            if decision.score < self.settings.vlm_rerank_min_score:
                continue
            original_score = item.original_score if item.original_score is not None else item.score
            final_score = (
                decision.score
                if self.settings.vlm_rerank_service_url
                else self._combined_score(item, decision.score)
            )
            updated = item.model_copy(
                update={
                    "original_score": original_score,
                    "score": final_score,
                    "rerank_score": decision.score,
                    "rerank_reason": self._reason(item, decision.reason, final_score),
                }
            )
            reranked.append(updated)
        reranked.sort(key=lambda result: result.score, reverse=True)
        return reranked[:limit]

    def _combined_score(self, item: SearchResultItem, vlm_score: float) -> float:
        embedding_score = item.embedding_rerank_score
        if embedding_score is None:
            embedding_score = item.original_score if item.original_score is not None else item.score
        embedding_score = self._score(embedding_score)
        total_weight = self.settings.embedding_rerank_weight + self.settings.vlm_rerank_weight
        if total_weight <= 0:
            return vlm_score
        return (
            embedding_score * self.settings.embedding_rerank_weight
            + vlm_score * self.settings.vlm_rerank_weight
        ) / total_weight

    def _reason(self, item: SearchResultItem, vlm_reason: str, final_score: float) -> str:
        if item.embedding_rerank_score is None:
            return vlm_reason
        return (
            f"{vlm_reason}；向量{item.embedding_rerank_score:.2f}，"
            f"融合{final_score:.2f}"
        )

    def rerank_image(
        self,
        query: str,
        image_path: Path,
        attributes: dict[str, Any],
    ) -> RerankDecision:
        if self.settings.vlm_rerank_service_url:
            return self._rerank_image_with_service(query, image_path, attributes)
        payload = {
            "model": self.settings.vlm_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是监控图像检索精排器。请判断候选图是否符合用户查询。"
                        "只输出合法 JSON，不要输出 Markdown。"
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"用户查询：{query}\n"
                                "候选已有结构化属性："
                                f"{json.dumps(attributes, ensure_ascii=False)[:2000]}\n"
                                "请按 0 到 1 打分，1 表示完全符合。"
                                '输出 JSON：{"score":0.0,"matched":false,'
                                '"reason":"20字以内中文理由"}'
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": self._image_data_url(image_path)},
                        },
                    ],
                },
            ],
            "temperature": 0,
            "max_tokens": 256,
            "response_format": {"type": "json_object"},
        }
        data = self._request_chat_completion(payload)
        content = VLMCaptionService(self.settings)._extract_message_content(data)
        parsed = self._parse_json_content(content)
        return RerankDecision(
            score=self._score(parsed.get("score")),
            matched=bool(parsed.get("matched")),
            reason=str(parsed.get("reason") or ""),
        )

    def _rerank_image_with_service(
        self,
        query: str,
        image_path: Path,
        attributes: dict[str, Any],
    ) -> RerankDecision:
        endpoint_url = self._rerank_service_endpoint_url()
        payload = {
            "query": query,
            "image_base64": self._image_base64(image_path),
            "image_filename": image_path.name,
            "attributes": attributes,
        }
        headers = {"Content-Type": "application/json"}
        if self.settings.vlm_rerank_service_api_key:
            headers["Authorization"] = f"Bearer {self.settings.vlm_rerank_service_api_key}"
        req = request.Request(
            endpoint_url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with request.urlopen(
                req,
                timeout=self.settings.vlm_rerank_timeout_seconds,
            ) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (OSError, error.URLError, json.JSONDecodeError) as exc:
            raise VLMRuntimeError(f"VLM rerank service request failed: {exc}") from exc
        if not isinstance(data, dict):
            raise VLMRuntimeError("VLM rerank service response is invalid")
        score = self._score(data.get("score"))
        return RerankDecision(
            score=score,
            matched=bool(data.get("matched", score > 0)),
            reason=str(data.get("reason") or "reranker score"),
        )

    def _rerank_service_endpoint_url(self) -> str:
        service_url = self.settings.vlm_rerank_service_url
        if not service_url:
            raise VLMRuntimeError("VLM_RERANK_SERVICE_URL is required")
        base = service_url.rstrip("/")
        if base.endswith("/api/embeddings/rerank") or base.endswith("/embeddings/rerank"):
            return base
        if base.endswith("/rerank") or base.endswith("/score"):
            return base
        return base + "/api/embeddings/rerank"

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
            with request.urlopen(
                req,
                timeout=self.settings.vlm_rerank_timeout_seconds,
            ) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (OSError, error.URLError, json.JSONDecodeError) as exc:
            raise VLMRuntimeError(f"VLM rerank request failed: {exc}") from exc
        if not isinstance(data, dict):
            raise VLMRuntimeError("VLM rerank response is invalid")
        return data

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
                raise VLMRuntimeError("VLM rerank did not return JSON") from None
            try:
                parsed = json.loads(text[start : end + 1])
            except json.JSONDecodeError as exc:
                raise VLMRuntimeError(f"VLM rerank JSON is invalid: {exc}") from exc
        if not isinstance(parsed, dict):
            raise VLMRuntimeError("VLM rerank JSON root must be an object")
        return parsed

    def _score(self, value: object) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(score, 1.0))

    def _item_image_path(self, item: SearchResultItem) -> Path | None:
        url = item.crop_url or item.image_url
        if not url or not url.startswith("/data/"):
            return None
        return self.settings.data_dir / url.removeprefix("/data/")

    def _image_data_url(self, image_path: Path) -> str:
        mime_type = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
        return f"data:{mime_type};base64,{self._image_base64(image_path)}"

    def _image_base64(self, image_path: Path) -> str:
        try:
            encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        except OSError as exc:
            raise VLMRuntimeError(f"Could not read image for VLM rerank: {exc}") from exc
        return encoded


class VisualRerankerService:
    provider_names = {"qwen3_vl_reranker", "qwen3-vl-reranker", "qwen3_vl"}

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def is_enabled(self) -> bool:
        return self.settings.vlm_rerank_provider.lower() in self.provider_names

    def rerank_image(
        self,
        query: str,
        image_path: Path,
        attributes: dict[str, Any] | None = None,
    ) -> RerankDecision:
        if not self.is_enabled():
            raise VLMRuntimeError("Visual reranker is not configured")
        return _Qwen3VLRerankerRuntime(
            model_name_or_path=self.settings.vlm_rerank_model,
            device=self.settings.vlm_rerank_device,
        ).rerank_image(query, image_path, attributes or {})


class _Qwen3VLRerankerRuntime:
    def __init__(self, model_name_or_path: str, device: str | None = None) -> None:
        self.model_name_or_path = model_name_or_path
        self.device = device

    def rerank_image(
        self,
        query: str,
        image_path: Path,
        attributes: dict[str, Any],
    ) -> RerankDecision:
        if "mlx-community/" not in self.model_name_or_path.lower():
            return self._rerank_with_qwen_script(query, image_path, attributes)
        return self._rerank_with_mlx(query, image_path, attributes)

    def _rerank_with_qwen_script(
        self,
        query: str,
        image_path: Path,
        attributes: dict[str, Any],
    ) -> RerankDecision:
        model = _cached_qwen3_vl_script_reranker(self.model_name_or_path)
        instruction = str(
            attributes.get("instruction")
            or "Retrieve person crop images relevant to the user's query."
        )
        payload = {
            "instruction": instruction,
            "query": {"text": query},
            "documents": [{"image": str(image_path)}],
        }
        try:
            scores = model.process(payload)
        except Exception as exc:
            raise VLMRuntimeError(f"Qwen3-VL reranker failed: {exc}") from exc
        score = self._score_from_predict_result(scores)
        return RerankDecision(score=score, matched=score > 0, reason="reranker score")

    def _rerank_with_cross_encoder(self, query: str, image_path: Path) -> RerankDecision:
        model = _cached_qwen3_vl_cross_encoder(self.model_name_or_path, self.device or "")
        try:
            if hasattr(model, "rank"):
                ranked = model.rank(query, [str(image_path)])
                score = self._score_from_rank_result(ranked)
            else:
                score = self._score_from_predict_result(model.predict([(query, str(image_path))]))
        except Exception as exc:
            raise VLMRuntimeError(f"Qwen3-VL reranker failed: {exc}") from exc
        return RerankDecision(score=score, matched=score > 0, reason="reranker score")

    def _rerank_with_mlx(
        self,
        query: str,
        image_path: Path,
        attributes: dict[str, Any],
    ) -> RerankDecision:
        try:
            from mlx_vlm import apply_chat_template, generate, load
            from mlx_vlm.utils import load_config
        except Exception as exc:
            raise VLMRuntimeError(
                "Qwen3-VL reranker dependencies are not installed. "
                "Install mlx-vlm in the embedding service environment, or use "
                "Qwen/Qwen3-VL-Reranker-2B with the sentence-transformers runtime."
            ) from exc
        try:
            model, processor = load(self.model_name_or_path)
            config = load_config(self.model_name_or_path)
            prompt = (
                "Judge whether the image matches the query. "
                f"Query: {query}\n"
                f"Attributes: {json.dumps(attributes, ensure_ascii=False)[:1000]}\n"
                "Return only a relevance score from 0 to 1."
            )
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(image_path)},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            formatted = apply_chat_template(processor, config, messages)
            output = generate(
                model,
                processor,
                formatted,
                max_tokens=16,
                temperature=0,
                verbose=False,
            )
        except Exception as exc:
            raise VLMRuntimeError(f"Qwen3-VL reranker failed: {exc}") from exc
        score = self._parse_score(str(output))
        return RerankDecision(score=score, matched=score > 0, reason="reranker score")

    def _score_from_rank_result(self, ranked: object) -> float:
        if isinstance(ranked, list) and ranked:
            first = ranked[0]
            if isinstance(first, dict):
                for key in ("score", "relevance_score", "similarity", "logit"):
                    if key in first:
                        return self._normalize_score(first[key])
            return self._normalize_score(first)
        return self._normalize_score(ranked)

    def _score_from_predict_result(self, predicted: object) -> float:
        if hasattr(predicted, "detach"):
            predicted = predicted.detach().cpu()
        if hasattr(predicted, "tolist"):
            predicted = predicted.tolist()
        if isinstance(predicted, list) and predicted:
            return self._normalize_score(predicted[0])
        return self._normalize_score(predicted)

    def _normalize_score(self, value: object) -> float:
        if isinstance(value, dict):
            for key in ("score", "relevance_score", "similarity", "logit"):
                if key in value:
                    return self._normalize_score(value[key])
            return 0.0
        try:
            score = float(value)
        except (TypeError, ValueError):
            return 0.0
        if score < 0.0 or score > 1.0:
            score = 1.0 / (1.0 + math.exp(-score))
        return max(0.0, min(score, 1.0))

    def _parse_score(self, text: str) -> float:
        match = re.search(r"(?<!\d)(?:0(?:\.\d+)?|1(?:\.0+)?)(?!\d)", text)
        if not match:
            return 0.0
        return max(0.0, min(float(match.group(0)), 1.0))


_QWEN3_VL_RERANKER_LOCK = threading.Lock()


@lru_cache(maxsize=2)
def _cached_qwen3_vl_script_reranker(model_name_or_path: str) -> Any:
    _ensure_torchvision_import_compat()
    model_path = Path(model_name_or_path)
    scripts_path = model_path / "scripts"
    if scripts_path.is_dir() and str(scripts_path) not in sys.path:
        sys.path.insert(0, str(scripts_path))
    try:
        import torch
        from qwen3_vl_reranker import Qwen3VLReranker
    except Exception as exc:
        raise VLMRuntimeError(
            "Qwen3-VL reranker runtime is not importable. Install torch, transformers, "
            "qwen-vl-utils, and use a model directory containing scripts/qwen3_vl_reranker.py."
        ) from exc

    kwargs: dict[str, Any] = {}
    if torch.cuda.is_available():
        kwargs["torch_dtype"] = torch.bfloat16
    else:
        kwargs["torch_dtype"] = torch.float32
    try:
        with _QWEN3_VL_RERANKER_LOCK:
            model = Qwen3VLReranker(model_name_or_path=model_name_or_path, **kwargs)
            _patch_qwen3_vl_reranker_token_truncation(model)
            return model
    except Exception as exc:
        raise VLMRuntimeError(
            f"Could not load Qwen3-VL reranker model {model_name_or_path}: {exc}"
        ) from exc


def _patch_qwen3_vl_reranker_token_truncation(model: Any) -> None:
    original_tokenize = model.tokenize

    def truncate_tokens_optimized(
        tokens: list[int],
        max_length: int,
        special_tokens: list[int],
    ) -> list[int]:
        if len(tokens) <= max_length:
            return tokens
        special_tokens_set = set(special_tokens)
        num_special = sum(1 for token in tokens if token in special_tokens_set)
        num_non_special_to_keep = max(0, max_length - num_special)
        final_tokens: list[int] = []
        non_special_kept_count = 0
        for token in tokens:
            if token in special_tokens_set:
                final_tokens.append(token)
            elif non_special_kept_count < num_non_special_to_keep:
                final_tokens.append(token)
                non_special_kept_count += 1
        return final_tokens

    def tokenize_with_tensor_types(pairs: list[Any], **kwargs: Any) -> Any:
        inputs = original_tokenize(pairs, **kwargs)
        token_type_ids = inputs.get("mm_token_type_ids")
        if isinstance(token_type_ids, list):
            try:
                import torch

                inputs["mm_token_type_ids"] = torch.tensor(
                    token_type_ids,
                    dtype=torch.long,
                    device=inputs["input_ids"].device,
                )
            except Exception as exc:
                raise VLMRuntimeError(
                    f"Could not convert Qwen3-VL token type ids to tensor: {exc}"
                ) from exc
        return inputs

    model.truncate_tokens_optimized = truncate_tokens_optimized
    model.tokenize = tokenize_with_tensor_types


@lru_cache(maxsize=2)
def _cached_qwen3_vl_cross_encoder(model_name_or_path: str, device: str) -> Any:
    _ensure_torchvision_import_compat()
    try:
        from sentence_transformers import CrossEncoder
    except Exception as exc:
        raise VLMRuntimeError(
            "sentence-transformers is not installed. Install requirements.visual.txt "
            "or the visual-embedding extra before enabling Qwen3-VL reranker."
        ) from exc
    kwargs: dict[str, Any] = {}
    if device:
        kwargs["device"] = device
    try:
        with _QWEN3_VL_RERANKER_LOCK:
            return CrossEncoder(model_name_or_path, **kwargs)
    except Exception as exc:
        raise VLMRuntimeError(
            f"Could not load Qwen3-VL reranker model {model_name_or_path}: {exc}"
        ) from exc


class EmbeddingRerankService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.visual_embedding = VisualEmbeddingService(settings)

    def is_enabled(self) -> bool:
        return self.visual_embedding.is_enabled()

    def rerank_person_crops(
        self,
        query: str,
        items: list[SearchResultItem],
        *,
        limit: int,
    ) -> list[SearchResultItem]:
        if not self.is_enabled():
            raise EmbeddingRuntimeError("Visual embedding rerank is not configured")
        query_vector = self.visual_embedding.embed_text(query)
        candidates = list(items[: self.settings.embedding_rerank_candidate_limit])
        scores = _map_concurrently(
            candidates,
            lambda item: self._candidate_score(query_vector, item),
            self.settings.embedding_rerank_max_workers,
        )

        reranked: list[SearchResultItem] = []
        for item, score in zip(candidates, scores, strict=True):
            if score is None:
                continue
            original_score = item.original_score if item.original_score is not None else item.score
            reranked.append(
                item.model_copy(
                    update={
                        "original_score": original_score,
                        "score": score,
                        "embedding_rerank_score": score,
                    }
                )
            )
        reranked.sort(
            key=lambda result: result.embedding_rerank_score or result.score,
            reverse=True,
        )
        return reranked[:limit]

    def _candidate_score(
        self,
        query_vector: list[float],
        item: SearchResultItem,
    ) -> float | None:
        score = item.embedding_rerank_score
        if score is None and self._has_vector_score(item):
            score = self._normalize_existing_score(item.score)
        if score is not None:
            return score
        image_path = self._item_image_path(item)
        if image_path is None or not image_path.exists():
            return None
        image_vector = self.visual_embedding.embed_image(image_path)
        return self._cosine_similarity(query_vector, image_vector)

    def _has_vector_score(self, item: SearchResultItem) -> bool:
        return item.score > 0 and item.person_name not in {"smoking", "phone"}

    def _normalize_existing_score(self, score: float) -> float:
        if score < 0:
            return 0.0
        if score > 1:
            return 1.0
        return score

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        if len(left) != len(right):
            raise EmbeddingRuntimeError(
                f"Embedding rerank dimension mismatch: {len(left)} != {len(right)}"
            )
        dot = sum(a * b for a, b in zip(left, right, strict=True))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm <= 0 or right_norm <= 0:
            return 0.0
        return max(0.0, min(dot / (left_norm * right_norm), 1.0))

    def _item_image_path(self, item: SearchResultItem) -> Path | None:
        url = item.crop_url or item.image_url
        if not url or not url.startswith("/data/"):
            return None
        return self.settings.data_dir / url.removeprefix("/data/")
