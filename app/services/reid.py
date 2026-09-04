import base64
import json
import threading
from pathlib import Path
from time import monotonic
from urllib import error, request

from app.config.settings import Settings

BATCH_LIMIT = 16

_FAILURE_LOCK = threading.Lock()
_FAILURE_UNTIL: dict[str, float] = {}
_SEMAPHORE_LOCK = threading.Lock()
_SEMAPHORES: dict[tuple[str, int], threading.BoundedSemaphore] = {}


class ReidRuntimeError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        # HTTP status when the service answered; None for transport failures. Batch callers
        # use it to tell "one bad image in the payload" from "service down, retry later".
        self.status_code = status_code


def _semaphore(endpoint: str, max_concurrency: int) -> threading.BoundedSemaphore:
    key = (endpoint, max_concurrency)
    with _SEMAPHORE_LOCK:
        semaphore = _SEMAPHORES.get(key)
        if semaphore is None:
            semaphore = threading.BoundedSemaphore(max_concurrency)
            _SEMAPHORES[key] = semaphore
        return semaphore


def _raise_if_in_cooldown(endpoint: str) -> None:
    with _FAILURE_LOCK:
        until = _FAILURE_UNTIL.get(endpoint, 0.0)
    remaining = until - monotonic()
    if remaining > 0:
        raise ReidRuntimeError(
            f"ReID service is in failure cooldown for another {remaining:.0f}s"
        )


def _mark_failure(endpoint: str, cooldown_seconds: int) -> None:
    if cooldown_seconds <= 0:
        return
    with _FAILURE_LOCK:
        _FAILURE_UNTIL[endpoint] = monotonic() + cooldown_seconds


def _clear_failure(endpoint: str) -> None:
    with _FAILURE_LOCK:
        _FAILURE_UNTIL.pop(endpoint, None)


def _endpoint(service_url: str, suffix: str) -> str:
    return f"{service_url.rstrip('/')}/{suffix}"


class ReidEmbeddingService:
    """Client for the SapiensID service in deploy/agx/reid_service.

    Vectors arrive L2-normalised, so callers can treat an inner product as cosine similarity.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def is_enabled(self) -> bool:
        return bool(self.settings.reid_enabled and self.settings.reid_service_url)

    @property
    def dim(self) -> int:
        return self.settings.reid_embedding_dim

    def probe(self, timeout_seconds: float = 3.0) -> tuple[bool, str | None]:
        """Cheap /health check for status reporting; never raises, never trips the breaker."""

        if not self.is_enabled():
            return False, "REID_ENABLED / REID_SERVICE_URL not set"
        endpoint = _endpoint(str(self.settings.reid_service_url), "ready")
        req = request.Request(endpoint, method="GET")
        try:
            with request.urlopen(req, timeout=timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (OSError, error.URLError, error.HTTPError, json.JSONDecodeError) as exc:
            return False, f"{type(exc).__name__}: {exc}"[:300]
        if not isinstance(data, dict) or data.get("status") != "ok":
            return False, f"unexpected health payload: {str(data)[:200]}"
        if data.get("loaded") is not True:
            return False, "ReID service has not loaded its model"
        if data.get("checkpoint_present") is False:
            return False, f"checkpoint missing under {data.get('checkpoint')}"
        if data.get("config_present") is False:
            return False, f"model.yaml missing under {data.get('checkpoint')}"
        mismatch = self._identity_mismatch(data)
        if mismatch:
            return False, mismatch
        return True, None

    def _identity_mismatch(self, data: dict) -> str | None:
        """The service states which model/dim/preprocessing it serves; trust that over settings.

        An index built for one identity queried through a service running another silently
        returns garbage similarities, which is far worse than refusing.
        """

        required = ("model", "checkpoint_revision", "preprocess_version")
        missing = [field for field in required if not data.get(field)]
        dim = data.get("embedding_dim", data.get("dim"))
        if dim is None:
            missing.append("embedding_dim")
        if missing:
            return f"service identity is incomplete; missing {', '.join(missing)}"

        model = data["model"]
        if model != self.settings.reid_model:
            return f"service model {model!r} != configured REID_MODEL {self.settings.reid_model!r}"
        revision = data["checkpoint_revision"]
        if revision != self.settings.reid_checkpoint_revision:
            return (
                f"service checkpoint revision {revision!r} != configured "
                f"REID_CHECKPOINT_REVISION {self.settings.reid_checkpoint_revision!r}"
            )
        preprocess = data["preprocess_version"]
        if preprocess != self.settings.reid_preprocess_version:
            return (
                f"service preprocess {preprocess!r} != configured "
                f"REID_PREPROCESS_VERSION {self.settings.reid_preprocess_version!r}"
            )
        try:
            service_dim = int(dim)
        except (TypeError, ValueError):
            return f"service embedding dimension is invalid: {dim!r}"
        if service_dim != self.settings.reid_embedding_dim:
            return (
                f"service dim {dim} != configured "
                f"REID_EMBEDDING_DIM {self.settings.reid_embedding_dim}"
            )
        return None

    def embed_image(
        self,
        image_path: Path,
        *,
        deadline: float | None = None,
    ) -> list[float]:
        return self.embed_images([image_path], deadline=deadline)[0]

    def embed_images(
        self,
        image_paths: list[Path],
        *,
        deadline: float | None = None,
    ) -> list[list[float]]:
        """Embeds in service-sized chunks; a partial failure fails the whole call.

        Batching perturbs a vector slightly - measured 1.000 alone against 0.9947 in a batch of
        8, since the pose stage letterboxes a batch to one shape. That is far inside the
        same-person margin (0.97 against 0.05), so indexed and query vectors stay comparable.
        """

        if not image_paths:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(image_paths), BATCH_LIMIT):
            chunk = image_paths[start : start + BATCH_LIMIT]
            vectors.extend(self._embed_chunk(chunk, deadline=deadline))
        return vectors

    def _embed_chunk(
        self,
        image_paths: list[Path],
        *,
        deadline: float | None = None,
    ) -> list[list[float]]:
        if not self.is_enabled():
            raise ReidRuntimeError("REID_ENABLED and REID_SERVICE_URL are required")
        self._remaining_timeout(deadline)
        images = [{"image_base64": _encode(path)} for path in image_paths]
        if len(images) == 1:
            data = self._post(
                "embed",
                {"image_base64": images[0]["image_base64"]},
                deadline=deadline,
            )
            self._validate_response_identity(data)
            return [_vector(data.get("embedding"), self.dim)]
        data = self._post(
            "embed-batch",
            {"images": images},
            deadline=deadline,
        )
        self._validate_response_identity(data)
        embeddings = data.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(images):
            raise ReidRuntimeError("ReID service returned the wrong number of embeddings")
        return [_vector(item, self.dim) for item in embeddings]

    def _validate_response_identity(self, data: dict[str, object]) -> None:
        mismatch = self._identity_mismatch(data)
        if mismatch:
            raise ReidRuntimeError(f"ReID service identity mismatch: {mismatch}")

    def _post(
        self,
        suffix: str,
        payload: dict[str, object],
        *,
        deadline: float | None = None,
    ) -> dict[str, object]:
        endpoint = _endpoint(str(self.settings.reid_service_url), suffix)
        _raise_if_in_cooldown(endpoint)
        semaphore = _semaphore(endpoint, self.settings.reid_max_concurrency)
        remaining = self._remaining_timeout(deadline)
        queue_timeout = min(
            self.settings.reid_queue_timeout_seconds,
            float(self.settings.reid_timeout_seconds),
            remaining if remaining is not None else float("inf"),
        )
        if not semaphore.acquire(timeout=queue_timeout):
            raise ReidRuntimeError(
                "ReID service is busy; waited "
                f"{queue_timeout:.1f}s for a local slot"
            )
        try:
            remaining = self._remaining_timeout(deadline)
            request_timeout = min(
                float(self.settings.reid_timeout_seconds),
                remaining if remaining is not None else float("inf"),
            )
            headers = {"Content-Type": "application/json"}
            if self.settings.reid_service_api_key:
                headers["Authorization"] = f"Bearer {self.settings.reid_service_api_key}"
            req = request.Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with request.urlopen(req, timeout=request_timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except UnicodeDecodeError as exc:
            _mark_failure(endpoint, self.settings.reid_failure_cooldown_seconds)
            raise ReidRuntimeError(f"ReID response was not valid UTF-8: {exc}") from exc
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            # 4xx is our payload's fault and will not heal on its own; do not trip the breaker.
            if exc.code >= 500:
                _mark_failure(endpoint, self.settings.reid_failure_cooldown_seconds)
            raise ReidRuntimeError(
                f"ReID request failed with status {exc.code}: {detail}",
                status_code=exc.code,
            ) from exc
        except (OSError, error.URLError, json.JSONDecodeError) as exc:
            _mark_failure(endpoint, self.settings.reid_failure_cooldown_seconds)
            raise ReidRuntimeError(f"ReID request failed: {exc}") from exc
        finally:
            semaphore.release()

        if not isinstance(data, dict):
            raise ReidRuntimeError("ReID response was not a JSON object")
        _clear_failure(endpoint)
        return data

    @staticmethod
    def _remaining_timeout(deadline: float | None) -> float | None:
        if deadline is None:
            return None
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise ReidRuntimeError("ReID operation deadline exceeded")
        return remaining


def _encode(image_path: Path) -> str:
    try:
        return base64.b64encode(image_path.read_bytes()).decode("ascii")
    except OSError as exc:
        raise ReidRuntimeError(f"Cannot read {image_path}: {exc}") from exc


def _vector(value: object, expected_dim: int) -> list[float]:
    if not isinstance(value, list) or not value:
        raise ReidRuntimeError("ReID response did not include an embedding")
    try:
        vector = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ReidRuntimeError(f"ReID embedding contained non-numeric values: {exc}") from exc
    if len(vector) != expected_dim:
        raise ReidRuntimeError(
            f"ReID embedding dimension mismatch: got {len(vector)}, "
            f"REID_EMBEDDING_DIM is {expected_dim}"
        )
    return vector
