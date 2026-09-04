import base64
import importlib
import json
import mimetypes
import os
import sys
import threading
import time
import types
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib import error, request

from app.config.settings import Settings


class EmbeddingRuntimeError(RuntimeError):
    pass


_TORCHVISION_STUB_LIBS: list[Any] = []
_TORCHVISION_COMPAT_LOCK = threading.Lock()
_TORCHVISION_COMPAT_READY = False
_QWEN3_VL_EMBEDDER_LOCK = threading.Lock()
_QWEN3_VL_EMBEDDER_CACHE: dict[tuple[str, str, str, str, str], Any] = {}
_QWEN3_VL_EMBEDDER_CACHE_MAX = 2
_HTTP_EMBEDDING_FAILURE_LOCK = threading.Lock()
_HTTP_EMBEDDING_FAILURE_UNTIL: dict[str, float] = {}
_HTTP_EMBEDDING_SEMAPHORES_LOCK = threading.Lock()
_HTTP_EMBEDDING_SEMAPHORES: dict[tuple[str, int], threading.BoundedSemaphore] = {}


class TextEmbeddingService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def is_enabled(self) -> bool:
        return self.settings.embedding_provider.lower() == "ollama"

    def embed_text(self, text: str) -> list[float]:
        provider = self.settings.embedding_provider.lower()
        if provider == "ollama":
            return self._embed_text_with_ollama(text)
        raise EmbeddingRuntimeError(f"Unsupported embedding provider: {provider}")

    def _embed_text_with_ollama(self, text: str) -> list[float]:
        payload = json.dumps(
            {
                "model": self.settings.ollama_embedding_model,
                "prompt": text,
            }
        ).encode("utf-8")
        url = self.settings.ollama_base_url.rstrip("/") + "/api/embeddings"
        req = request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=60) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (OSError, error.URLError, json.JSONDecodeError) as exc:
            raise EmbeddingRuntimeError(f"Ollama embedding request failed: {exc}") from exc

        embedding = data.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise EmbeddingRuntimeError("Ollama embedding response did not include an embedding")
        vector = [float(value) for value in embedding]
        if len(vector) != self.settings.embedding_dim:
            raise EmbeddingRuntimeError(
                "Embedding dimension mismatch: "
                f"expected {self.settings.embedding_dim}, got {len(vector)}"
            )
        return vector


class VisualEmbeddingService:
    """Text/image embedding runtime for visual retrieval collections.

    Providers are optional and imported lazily so the base API can run without
    heavyweight model dependencies.
    """

    sentence_transformer_providers = {"sentence_transformers", "clip"}
    qwen3_vl_providers = {"qwen3_vl", "qwen3-vl"}
    qwen3_vl_http_providers = {"qwen3_vl_http", "qwen3-vl-http"}
    dashscope_multimodal_providers = {"dashscope_multimodal", "dashscope-multimodal"}

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def is_enabled(self) -> bool:
        provider = self.settings.visual_embedding_provider.lower()
        return (
            provider
            in self.sentence_transformer_providers
            | self.qwen3_vl_providers
            | self.qwen3_vl_http_providers
            | self.dashscope_multimodal_providers
        )

    def embed_text(self, text: str) -> list[float]:
        provider = self.settings.visual_embedding_provider.lower()
        if provider in self.sentence_transformer_providers:
            vector = _SentenceTransformerVisualRuntime(
                model_name=self.settings.visual_embedding_model,
                device=self.settings.visual_embedding_device,
                instruction=self.settings.visual_embedding_instruction,
            ).embed_text(text)
        elif provider in self.qwen3_vl_providers:
            vector = _Qwen3VLVisualRuntime(
                model_name_or_path=self.settings.visual_embedding_model,
                repo_dir=self.settings.qwen3_vl_embedding_repo_dir,
                extra_pythonpath=self.settings.qwen3_vl_embedding_pythonpath,
                instruction=self.settings.visual_embedding_instruction,
                torch_dtype=self.settings.qwen3_vl_embedding_torch_dtype,
                attn_implementation=self.settings.qwen3_vl_embedding_attn_implementation,
            ).embed_text(text)
        elif provider in self.qwen3_vl_http_providers:
            vector = _Qwen3VLHTTPVisualRuntime(
                service_url=self.settings.visual_embedding_service_url,
                api_key=self.settings.visual_embedding_service_api_key,
                instruction=self.settings.visual_embedding_instruction,
                timeout_seconds=self.settings.visual_embedding_service_timeout_seconds,
                failure_cooldown_seconds=(
                    self.settings.visual_embedding_service_failure_cooldown_seconds
                ),
                max_concurrency=self.settings.visual_embedding_max_concurrency,
                queue_timeout_seconds=self.settings.visual_embedding_queue_timeout_seconds,
            ).embed_text(text)
        elif provider in self.dashscope_multimodal_providers:
            vector = _DashScopeMultimodalEmbeddingRuntime(
                model=self.settings.visual_embedding_model,
                api_key=self.settings.visual_embedding_service_api_key,
                service_url=self.settings.visual_embedding_service_url,
                timeout_seconds=self.settings.visual_embedding_service_timeout_seconds,
                dimension=self.settings.visual_embedding_dim,
            ).embed_text(text)
        else:
            raise EmbeddingRuntimeError(f"Unsupported visual embedding provider: {provider}")
        return self._validate_vector(vector)

    def embed_image(self, image_path: Path) -> list[float]:
        if not image_path.exists():
            raise EmbeddingRuntimeError(f"Image path does not exist: {image_path}")

        provider = self.settings.visual_embedding_provider.lower()
        if provider in self.sentence_transformer_providers:
            vector = _SentenceTransformerVisualRuntime(
                model_name=self.settings.visual_embedding_model,
                device=self.settings.visual_embedding_device,
                instruction=self.settings.visual_embedding_instruction,
            ).embed_image(image_path)
        elif provider in self.qwen3_vl_providers:
            vector = _Qwen3VLVisualRuntime(
                model_name_or_path=self.settings.visual_embedding_model,
                repo_dir=self.settings.qwen3_vl_embedding_repo_dir,
                extra_pythonpath=self.settings.qwen3_vl_embedding_pythonpath,
                instruction=self.settings.visual_embedding_instruction,
                torch_dtype=self.settings.qwen3_vl_embedding_torch_dtype,
                attn_implementation=self.settings.qwen3_vl_embedding_attn_implementation,
            ).embed_image(image_path)
        elif provider in self.qwen3_vl_http_providers:
            vector = _Qwen3VLHTTPVisualRuntime(
                service_url=self.settings.visual_embedding_service_url,
                api_key=self.settings.visual_embedding_service_api_key,
                instruction=self.settings.visual_embedding_instruction,
                timeout_seconds=self.settings.visual_embedding_service_timeout_seconds,
                failure_cooldown_seconds=(
                    self.settings.visual_embedding_service_failure_cooldown_seconds
                ),
                max_concurrency=self.settings.visual_embedding_max_concurrency,
                queue_timeout_seconds=self.settings.visual_embedding_queue_timeout_seconds,
            ).embed_image(image_path)
        elif provider in self.dashscope_multimodal_providers:
            vector = _DashScopeMultimodalEmbeddingRuntime(
                model=self.settings.visual_embedding_model,
                api_key=self.settings.visual_embedding_service_api_key,
                service_url=self.settings.visual_embedding_service_url,
                timeout_seconds=self.settings.visual_embedding_service_timeout_seconds,
                dimension=self.settings.visual_embedding_dim,
            ).embed_image(image_path)
        else:
            raise EmbeddingRuntimeError(f"Unsupported visual embedding provider: {provider}")
        return self._validate_vector(vector)

    def _validate_vector(self, vector: list[float]) -> list[float]:
        if len(vector) != self.settings.visual_embedding_dim:
            raise EmbeddingRuntimeError(
                "Visual embedding dimension mismatch: "
                f"expected {self.settings.visual_embedding_dim}, got {len(vector)}"
            )
        return vector


class _SentenceTransformerVisualRuntime:
    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        instruction: str | None = None,
    ) -> None:
        self.model = _cached_sentence_transformer(model_name, device or "")
        self.instruction = instruction

    def embed_text(self, text: str) -> list[float]:
        kwargs: dict[str, object] = {
            "normalize_embeddings": True,
            "show_progress_bar": False,
        }
        if self.instruction:
            kwargs["prompt"] = self.instruction
        try:
            encoded = self.model.encode(text, **kwargs)
        except TypeError:
            kwargs.pop("prompt", None)
            encoded = self.model.encode(text, **kwargs)
        return _to_float_vector(encoded)

    def embed_image(self, image_path: Path) -> list[float]:
        try:
            from PIL import Image as PILImage
        except Exception as exc:
            raise EmbeddingRuntimeError(f"Pillow is not installed: {exc}") from exc

        try:
            with PILImage.open(image_path) as image:
                rgb_image = image.convert("RGB")
                encoded = self.model.encode(
                    rgb_image,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
        except OSError as exc:
            raise EmbeddingRuntimeError(f"Could not read image for embedding: {exc}") from exc
        except Exception:
            encoded = self.model.encode(
                {"image": str(image_path)},
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return _to_float_vector(encoded)


class _Qwen3VLVisualRuntime:
    def __init__(
        self,
        model_name_or_path: str,
        repo_dir: Path | None,
        extra_pythonpath: str | None,
        instruction: str,
        torch_dtype: str | None,
        attn_implementation: str | None,
    ) -> None:
        self.instruction = instruction
        self.model = _cached_qwen3_vl_embedder(
            model_name_or_path,
            str(repo_dir) if repo_dir else "",
            extra_pythonpath or "",
            torch_dtype or "",
            attn_implementation or "",
        )

    def embed_text(self, text: str) -> list[float]:
        return self._process({"text": text, "instruction": self.instruction})

    def embed_image(self, image_path: Path) -> list[float]:
        return self._process({"image": str(image_path)})

    def _process(self, item: dict[str, str]) -> list[float]:
        try:
            embeddings = self.model.process([item])
        except Exception as exc:
            raise EmbeddingRuntimeError(f"Qwen3-VL embedding failed: {exc}") from exc
        return _to_float_vector(embeddings[0] if _is_sequence(embeddings) else embeddings)


class _Qwen3VLHTTPVisualRuntime:
    def __init__(
        self,
        service_url: str | None,
        api_key: str | None,
        instruction: str,
        timeout_seconds: int,
        failure_cooldown_seconds: int,
        max_concurrency: int,
        queue_timeout_seconds: float,
    ) -> None:
        if not service_url:
            raise EmbeddingRuntimeError(
                "VISUAL_EMBEDDING_SERVICE_URL is required for qwen3_vl_http"
            )
        self.endpoint_url = _visual_embedding_endpoint_url(service_url)
        self.api_key = api_key
        self.instruction = instruction
        self.timeout_seconds = timeout_seconds
        self.failure_cooldown_seconds = failure_cooldown_seconds
        self.queue_timeout_seconds = queue_timeout_seconds
        self.semaphore = _http_embedding_semaphore(self.endpoint_url, max_concurrency)

    def embed_text(self, text: str) -> list[float]:
        return self._request_embedding(
            {
                "text": text,
                "instruction": self.instruction,
            }
        )

    def embed_image(self, image_path: Path) -> list[float]:
        try:
            image_base64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
        except OSError as exc:
            raise EmbeddingRuntimeError(f"Could not read image for embedding: {exc}") from exc
        return self._request_embedding(
            {
                "image_base64": image_base64,
                "image_filename": image_path.name,
            }
        )

    def _request_embedding(self, payload: dict[str, str]) -> list[float]:
        _raise_if_http_embedding_in_cooldown(self.endpoint_url)
        acquired = self.semaphore.acquire(timeout=self.queue_timeout_seconds)
        if not acquired:
            raise EmbeddingRuntimeError(
                "Qwen3-VL HTTP embedding service is busy; "
                f"waited {self.queue_timeout_seconds:.1f}s for a local slot"
            )
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            headers["X-API-Key"] = self.api_key
        req = request.Request(
            self.endpoint_url,
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            try:
                with request.urlopen(req, timeout=self.timeout_seconds) as response:
                    data = json.loads(response.read().decode("utf-8"))
            except error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
                if exc.code >= 500:
                    _mark_http_embedding_failure(
                        self.endpoint_url,
                        self.failure_cooldown_seconds,
                    )
                raise EmbeddingRuntimeError(
                    f"Qwen3-VL HTTP embedding request failed with status {exc.code}: {detail}"
                ) from exc
            except (OSError, error.URLError, json.JSONDecodeError) as exc:
                _mark_http_embedding_failure(
                    self.endpoint_url,
                    self.failure_cooldown_seconds,
                )
                raise EmbeddingRuntimeError(
                    f"Qwen3-VL HTTP embedding request failed: {exc}"
                ) from exc
        finally:
            self.semaphore.release()

        if not isinstance(data, dict):
            raise EmbeddingRuntimeError("Qwen3-VL HTTP embedding response was not a JSON object")
        embedding = data.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise EmbeddingRuntimeError(
                "Qwen3-VL HTTP embedding response did not include an embedding"
            )
        return _to_float_vector(embedding)


class _DashScopeMultimodalEmbeddingRuntime:
    endpoint_url = (
        "https://dashscope.aliyuncs.com/api/v1/services/embeddings/"
        "multimodal-embedding/multimodal-embedding"
    )

    def __init__(
        self,
        model: str,
        api_key: str | None,
        service_url: str | None,
        timeout_seconds: int,
        dimension: int,
    ) -> None:
        if not api_key:
            raise EmbeddingRuntimeError(
                "VISUAL_EMBEDDING_SERVICE_API_KEY is required for dashscope_multimodal"
            )
        self.model = model
        self.api_key = api_key
        self.endpoint_url = service_url.rstrip("/") if service_url else self.endpoint_url
        self.timeout_seconds = timeout_seconds
        self.dimension = dimension

    def embed_text(self, text: str) -> list[float]:
        return self._request_embedding({"text": text})

    def embed_image(self, image_path: Path) -> list[float]:
        try:
            encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        except OSError as exc:
            raise EmbeddingRuntimeError(f"Could not read image for embedding: {exc}") from exc
        mime_type = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
        return self._request_embedding(
            {
                "image": f"data:{mime_type};base64,{encoded}",
            }
        )

    def _request_embedding(self, input_item: dict[str, str]) -> list[float]:
        parameters: dict[str, int] = {}
        if self.dimension:
            parameters["dimension"] = self.dimension
        payload = {
            "model": self.model,
            "input": {
                "contents": [input_item],
            },
            "parameters": parameters,
        }
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self.endpoint_url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise EmbeddingRuntimeError(
                f"DashScope multimodal embedding request failed with status {exc.code}: {detail}"
            ) from exc
        except (OSError, error.URLError, json.JSONDecodeError) as exc:
            raise EmbeddingRuntimeError(
                f"DashScope multimodal embedding request failed: {exc}"
            ) from exc
        return _to_float_vector(_extract_dashscope_embedding(data))


def _extract_dashscope_embedding(data: object) -> list[object]:
    if not isinstance(data, dict):
        raise EmbeddingRuntimeError("DashScope embedding response was not a JSON object")
    output = data.get("output")
    if not isinstance(output, dict):
        raise EmbeddingRuntimeError("DashScope embedding response did not include output")
    embeddings = output.get("embeddings")
    if isinstance(embeddings, list) and embeddings:
        first = embeddings[0]
        if isinstance(first, dict):
            embedding = first.get("embedding")
            if isinstance(embedding, list) and embedding:
                return embedding
    embedding = output.get("embedding")
    if isinstance(embedding, list) and embedding:
        return embedding
    raise EmbeddingRuntimeError("DashScope embedding response did not include an embedding")


def _visual_embedding_endpoint_url(service_url: str) -> str:
    url = service_url.rstrip("/")
    if url.endswith("/api/embeddings/visual") or url.endswith("/embeddings/visual"):
        return url
    return f"{url}/api/embeddings/visual"


def _http_embedding_semaphore(
    endpoint_url: str,
    max_concurrency: int,
) -> threading.BoundedSemaphore:
    key = (endpoint_url, max_concurrency)
    with _HTTP_EMBEDDING_SEMAPHORES_LOCK:
        semaphore = _HTTP_EMBEDDING_SEMAPHORES.get(key)
        if semaphore is None:
            semaphore = threading.BoundedSemaphore(max_concurrency)
            _HTTP_EMBEDDING_SEMAPHORES[key] = semaphore
        return semaphore


def _raise_if_http_embedding_in_cooldown(endpoint_url: str) -> None:
    with _HTTP_EMBEDDING_FAILURE_LOCK:
        failure_until = _HTTP_EMBEDDING_FAILURE_UNTIL.get(endpoint_url, 0.0)
        now = time.monotonic()
        if failure_until <= now:
            _HTTP_EMBEDDING_FAILURE_UNTIL.pop(endpoint_url, None)
            return
    remaining = failure_until - now
    raise EmbeddingRuntimeError(
        "Qwen3-VL HTTP embedding service is in failure cooldown "
        f"for {remaining:.1f}s"
    )


def _mark_http_embedding_failure(endpoint_url: str, cooldown_seconds: int) -> None:
    if cooldown_seconds <= 0:
        return
    with _HTTP_EMBEDDING_FAILURE_LOCK:
        _HTTP_EMBEDDING_FAILURE_UNTIL[endpoint_url] = time.monotonic() + cooldown_seconds


@lru_cache(maxsize=4)
def _cached_sentence_transformer(model_name: str, device: str) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        raise EmbeddingRuntimeError(
            "sentence-transformers is not installed. Install requirements.visual.txt "
            "or the visual-embedding extra before enabling this provider."
        ) from exc

    kwargs: dict[str, str] = {}
    if device:
        kwargs["device"] = device
    try:
        return SentenceTransformer(model_name, **kwargs)
    except Exception as exc:
        raise EmbeddingRuntimeError(
            f"Could not load visual embedding model {model_name}: {exc}"
        ) from exc


def _cached_qwen3_vl_embedder(
    model_name_or_path: str,
    repo_dir: str,
    extra_pythonpath: str,
    torch_dtype: str,
    attn_implementation: str,
) -> Any:
    cache_key = (
        model_name_or_path,
        repo_dir,
        extra_pythonpath,
        torch_dtype,
        attn_implementation,
    )
    with _QWEN3_VL_EMBEDDER_LOCK:
        cached = _QWEN3_VL_EMBEDDER_CACHE.get(cache_key)
        if cached is not None:
            return cached
        embedder = _load_qwen3_vl_embedder(
            model_name_or_path,
            repo_dir,
            extra_pythonpath,
            torch_dtype,
            attn_implementation,
        )
        if len(_QWEN3_VL_EMBEDDER_CACHE) >= _QWEN3_VL_EMBEDDER_CACHE_MAX:
            _QWEN3_VL_EMBEDDER_CACHE.pop(next(iter(_QWEN3_VL_EMBEDDER_CACHE)))
        _QWEN3_VL_EMBEDDER_CACHE[cache_key] = embedder
        return embedder


def _load_qwen3_vl_embedder(
    model_name_or_path: str,
    repo_dir: str,
    extra_pythonpath: str,
    torch_dtype: str,
    attn_implementation: str,
) -> Any:
    for item in extra_pythonpath.split(os.pathsep):
        path = item.strip()
        if path and path not in sys.path:
            sys.path.insert(0, path)

    _ensure_torchvision_import_compat()

    repo_path = Path(repo_dir) if repo_dir else None
    if repo_path and str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))
    model_scripts_path = Path(model_name_or_path) / "scripts"
    if model_scripts_path.is_dir() and str(model_scripts_path) not in sys.path:
        sys.path.insert(0, str(model_scripts_path))

    try:
        from src.models.qwen3_vl_embedding import Qwen3VLEmbedder
    except Exception:
        try:
            from qwen3_vl_embedding import Qwen3VLEmbedder
        except Exception as script_exc:
            raise EmbeddingRuntimeError(
                "Qwen3-VL embedding runtime is not importable. Clone "
                "https://github.com/QwenLM/Qwen3-VL-Embedding and set "
                "QWEN3_VL_EMBEDDING_REPO_DIR to that checkout, or use a ModelScope "
                "Qwen3-VL-Embedding model directory that contains scripts/qwen3_vl_embedding.py. "
                f"Import error: {script_exc}"
            ) from script_exc

    kwargs: dict[str, Any] = {"model_name_or_path": model_name_or_path}
    if torch_dtype:
        kwargs["torch_dtype"] = _resolve_torch_dtype(torch_dtype)
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    try:
        return Qwen3VLEmbedder(**kwargs)
    except Exception as exc:
        raise EmbeddingRuntimeError(
            f"Could not load Qwen3-VL embedding model {model_name_or_path}: {exc}"
        ) from exc


def _ensure_torchvision_import_compat() -> None:
    """Allow torchvision import on Jetson wheels that omit the compiled nms op.

    Qwen3-VL preprocessing imports torchvision through transformers/qwen-vl-utils,
    but SightIndex does not use torchvision NMS. Some Jetson PyTorch installs ship a
    CUDA torch wheel without the matching torchvision custom op, causing import-time
    registration to fail before the model can load.
    """

    global _TORCHVISION_COMPAT_READY
    if _TORCHVISION_COMPAT_READY:
        return

    with _TORCHVISION_COMPAT_LOCK:
        if _TORCHVISION_COMPAT_READY:
            return
        try:
            import torch
        except Exception:
            _TORCHVISION_COMPAT_READY = True
            return

        if isinstance(torch, types.ModuleType):
            try:
                importlib.import_module("torchvision")
                _TORCHVISION_COMPAT_READY = True
                return
            except Exception as exc:
                message = str(exc).lower()
                if (
                    "torchvision::nms" not in message
                    and "operator torchvision::nms" not in message
                ):
                    _TORCHVISION_COMPAT_READY = True
                    return
                _clear_partial_torchvision_imports()

        try:
            lib = torch.library.Library("torchvision", "DEF")
            lib.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
            _TORCHVISION_STUB_LIBS.append(lib)
        except RuntimeError as exc:
            message = str(exc).lower()
            if "already" not in message and "single torch_library" not in message:
                raise
        _TORCHVISION_COMPAT_READY = True


def _clear_partial_torchvision_imports() -> None:
    for name in list(sys.modules):
        if name == "torchvision" or name.startswith("torchvision."):
            sys.modules.pop(name, None)


def _resolve_torch_dtype(name: str) -> Any:
    try:
        import torch
    except Exception as exc:
        raise EmbeddingRuntimeError(f"PyTorch is not installed: {exc}") from exc
    if not hasattr(torch, name):
        raise EmbeddingRuntimeError(f"Unsupported torch dtype: {name}")
    return getattr(torch, name)


def _to_float_vector(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if _is_sequence(value) and value and _is_sequence(value[0]):
        value = value[0]
    if not _is_sequence(value) or not value:
        raise EmbeddingRuntimeError("Embedding runtime returned an empty vector")
    return [float(item) for item in value]


def _is_sequence(value: Any) -> bool:
    return isinstance(value, list | tuple)
