"""Compare detector ROIs with stored display crops, without changing media or indexes.

Run on the server that owns the images. This is an input-sensitivity diagnostic,
not an identity evaluation: neither sharpness nor vector drift measures accuracy.
By default no encoder is instantiated and no HTTP requests are made. Optional
inference is restricted to a loopback endpoint, with redirects and proxies disabled.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import sqlite3
import sys
import tempfile
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib import parse, request

import cv2
import numpy as np
from numpy.typing import NDArray

ROOT_DIR = Path(__file__).resolve().parents[1]
MAX_CROPS = 10
ImageArray = NDArray[np.uint8]


class DiagnosticError(ValueError):
    """A safe, non-media-bearing reason to skip an input."""


class Encoder(Protocol):
    """Only the read-only inference capability needed by this diagnostic."""

    def embed_image(self, image_path: Path) -> list[float]: ...


@dataclass(frozen=True)
class CropInput:
    """The minimum metadata required for one paired image comparison."""

    crop_id: uuid.UUID
    frame_url: str
    crop_url: str
    bbox: object


@contextmanager
def readonly_database(database: Path) -> Iterator[sqlite3.Connection]:
    """Open an existing SQLite database read-only; never create or migrate it."""

    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        connection.execute("PRAGMA query_only = ON")
        connection.execute("PRAGMA busy_timeout = 3000")
        yield connection
    finally:
        connection.close()


def load_crop(connection: sqlite3.Connection, crop_id: uuid.UUID) -> CropInput:
    """Fetch one crop and its original frame, without ORM or cache side effects."""

    row = connection.execute(
        "SELECT pc.crop_url, pc.bbox, i.image_url "
        "FROM person_crops AS pc LEFT JOIN images AS i ON i.id = pc.image_id "
        "WHERE pc.id IN (?, ?)",
        (crop_id.hex, str(crop_id)),
    ).fetchone()
    if row is None:
        raise DiagnosticError("crop_not_found")
    try:
        bbox = json.loads(row[1])
    except (TypeError, ValueError) as exc:
        raise DiagnosticError("invalid_bbox_json") from exc
    if not isinstance(row[0], str) or not isinstance(row[2], str):
        raise DiagnosticError("missing_media_reference")
    return CropInput(crop_id, row[2], row[0], bbox)


def local_data_path(data_dir: Path, url: str) -> Path:
    """Resolve only files below the configured data directory, including symlinks."""

    if not url.startswith("/data/") or "?" in url or "#" in url or "%" in url:
        raise DiagnosticError("invalid_data_url")
    root = data_dir.resolve()
    path = (root / url.removeprefix("/data/")).resolve()
    if not path.is_relative_to(root):
        raise DiagnosticError("media_outside_data_directory")
    if not path.is_file():
        raise DiagnosticError("media_file_missing")
    return path


def read_image(path: Path) -> ImageArray:
    """Decode a color image and reject failed/empty decoding."""

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise DiagnosticError("image_decode_failed")
    return image


def detector_roi(image: ImageArray, bbox: object) -> tuple[ImageArray, float]:
    """Take an unpadded detector ROI; missing/invalid geometry never means full frame.

    Finite pixel coordinates are required. Partly clipped boxes use the actual
    intersection, not a shifted box, and report the retained fraction of box area.
    """

    if not isinstance(bbox, dict):
        raise DiagnosticError("invalid_detector_bbox")
    values: list[float] = []
    for key in ("x", "y", "width", "height"):
        value = bbox.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise DiagnosticError("invalid_detector_bbox")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise DiagnosticError("invalid_detector_bbox")
        values.append(numeric)
    x, y, width, height = values
    if width <= 0 or height <= 0:
        raise DiagnosticError("invalid_detector_bbox")
    frame_height, frame_width = image.shape[:2]
    left, top = max(0.0, x), max(0.0, y)
    right, bottom = min(float(frame_width), x + width), min(float(frame_height), y + height)
    if right <= left or bottom <= top:
        raise DiagnosticError("detector_bbox_outside_frame")
    # Cover the valid fractional box instead of silently dropping edge pixels.
    roi = image[math.floor(top) : math.ceil(bottom), math.floor(left) : math.ceil(right)]
    if roi.size == 0:
        raise DiagnosticError("empty_detector_roi")
    retained = ((right - left) / width) * ((bottom - top) / height)
    return roi.copy(), retained


def image_metrics(image: ImageArray) -> dict[str, int | float]:
    """Return non-semantic quality descriptors; resized sharpness is not lost detail."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    reference = cv2.resize(gray, (96, 192), interpolation=cv2.INTER_AREA)
    return {
        "width": int(width),
        "height": int(height),
        "aspect_ratio": round(width / height, 6),
        "brightness_mean_0_255": round(float(gray.mean()), 4),
        "brightness_p05_0_255": round(float(np.percentile(gray, 5)), 4),
        "brightness_p95_0_255": round(float(np.percentile(gray, 95)), 4),
        "dark_pixel_fraction_le_16": round(float(np.mean(gray <= 16)), 6),
        "bright_pixel_fraction_ge_239": round(float(np.mean(gray >= 239)), 6),
        "laplacian_variance_native": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 4),
        "laplacian_variance_at_96x192": round(float(cv2.Laplacian(reference, cv2.CV_64F).var()), 4),
    }


def vector_cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Compute a finite cosine defensively; no vector is returned in the report."""

    first, second = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if first.ndim != 1 or second.ndim != 1 or first.size == 0 or first.shape != second.shape:
        raise DiagnosticError("incompatible_inference_vectors")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise DiagnosticError("invalid_inference_vector")
    # Scale before norm so even malformed very large values cannot overflow it.
    first_scale, second_scale = float(np.max(np.abs(first))), float(np.max(np.abs(second)))
    if first_scale == 0 or second_scale == 0:
        raise DiagnosticError("invalid_inference_vector")
    first, second = first / first_scale, second / second_scale
    cosine = float(np.dot(first / np.linalg.norm(first), second / np.linalg.norm(second)))
    return max(-1.0, min(1.0, cosine))


def infer_pair(encoder: Encoder, roi: ImageArray, display_path: Path) -> dict[str, float]:
    """Infer individually with one encoder, cleaning the lossless temporary ROI on all exits."""

    with tempfile.TemporaryDirectory(prefix="sightindex-crop-input-") as directory:
        raw_path = Path(directory) / "detector-roi.png"
        if not cv2.imwrite(str(raw_path), roi):
            raise DiagnosticError("temporary_roi_write_failed")
        raw_vector = encoder.embed_image(raw_path)
        display_vector = encoder.embed_image(display_path)
        cosine = vector_cosine(raw_vector, display_vector)
        return {
            "raw_display_cosine": round(cosine, 6),
            "cosine_distance_drift": round(1.0 - cosine, 6),
        }


def diagnose_crop(
    crop: CropInput, data_dir: Path, encoder: Encoder | None = None
) -> dict[str, object]:
    """Compare one original-frame detector ROI and existing display crop in isolation."""

    frame = read_image(local_data_path(data_dir, crop.frame_url))
    display_path = local_data_path(data_dir, crop.crop_url)
    display = read_image(display_path)
    roi, retained = detector_roi(frame, crop.bbox)
    result: dict[str, object] = {
        "crop_id": str(crop.crop_id),
        "status": "quality_only",
        "raw_detector_roi": image_metrics(roi),
        "stored_display_crop": image_metrics(display),
        "detector_bbox_retained_area_fraction": round(retained, 6),
        "display_equals_full_frame_pixels": bool(
            display.shape == frame.shape and np.array_equal(display, frame)
        ),
        "inference": None,
    }
    if encoder is not None:
        try:
            result["inference"] = infer_pair(encoder, roi, display_path)
            result["status"] = "inferred"
        except (OSError, RuntimeError, ValueError, cv2.error) as exc:
            result["status"] = "inference_failed"
            # Service errors can contain paths or request data; never print their message.
            result["error"] = str(exc) if isinstance(exc, DiagnosticError) else type(exc).__name__
    return result


class NoRedirect(request.HTTPRedirectHandler):
    """Prevent a local encoder from redirecting private media to another endpoint."""

    def redirect_request(
        self,
        req: request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None


def validate_loopback_url(service_url: str) -> None:
    """Reject remote hosts, credentials, queries, and ambiguous endpoint URLs."""

    parsed = parse.urlsplit(service_url)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "::1", "localhost"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise DiagnosticError("inference_requires_loopback_http_endpoint")
    try:
        _port = parsed.port
    except ValueError as exc:
        raise DiagnosticError("invalid_inference_port") from exc


@dataclass
class LoopbackEncoder:
    """Read-only local encoder client; validates the configured model on every response."""

    settings: Any

    def __post_init__(self) -> None:
        validate_loopback_url(str(self.settings.reid_service_url or ""))
        if not self.settings.reid_enabled:
            raise DiagnosticError("reid_disabled")

    def embed_image(self, image_path: Path) -> list[float]:
        """Make one proxy-free, non-redirecting local inference request."""

        # This import neither initializes the app nor opens a database/index connection.
        from app.services.reid import ReidEmbeddingService

        payload = {"image_base64": base64.b64encode(image_path.read_bytes()).decode("ascii")}
        headers = {"Content-Type": "application/json"}
        if self.settings.reid_service_api_key:
            headers["Authorization"] = f"Bearer {self.settings.reid_service_api_key}"
        parsed = parse.urlsplit(str(self.settings.reid_service_url))
        # Resolve the accepted localhost spelling to a numeric loopback address so
        # even a surprising local DNS/hosts configuration cannot export media.
        host = "[::1]" if parsed.hostname == "::1" else "127.0.0.1"
        endpoint = f"http://{host}:{parsed.port or 80}/embed"
        req = request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        opener = request.build_opener(request.ProxyHandler({}), NoRedirect())
        timeout = min(float(self.settings.reid_timeout_seconds), 30.0)
        with opener.open(req, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        if not isinstance(data, dict):
            raise DiagnosticError("invalid_inference_response")
        # Do not report a mismatch message: it may include unexpected service content.
        if ReidEmbeddingService(self.settings)._identity_mismatch(data) is not None:
            raise DiagnosticError("inference_identity_mismatch")
        vector = data.get("embedding")
        if not isinstance(vector, list) or len(vector) != self.settings.reid_embedding_dim:
            raise DiagnosticError("incompatible_inference_vectors")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in vector):
            raise DiagnosticError("invalid_inference_vector")
        return [float(value) for value in vector]


def main(argv: Sequence[str] | None = None) -> int:
    """Print metadata-only JSON for at most ten explicit crop UUIDs."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True, help="Existing SQLite database")
    parser.add_argument("--data-dir", type=Path, required=True, help="Server-local media directory")
    parser.add_argument("--crop-id", action="append", type=uuid.UUID, required=True)
    parser.add_argument(
        "--allow-inference",
        action="store_true",
        help="Allow local GPU inference; never writes indexes",
    )
    args = parser.parse_args(argv)
    if not 1 <= len(args.crop_id) <= MAX_CROPS:
        parser.error(f"Provide between 1 and {MAX_CROPS} crop ids")
    crop_ids = list(dict.fromkeys(args.crop_id))
    encoder: Encoder | None = None
    identity: dict[str, object] | None = None
    if args.allow_inference:
        if str(ROOT_DIR) not in sys.path:
            sys.path.insert(0, str(ROOT_DIR))
        from app.config.settings import get_settings

        settings = get_settings()
        try:
            encoder = LoopbackEncoder(settings)
        except DiagnosticError as exc:
            parser.error(str(exc))
        identity = {
            "model": settings.reid_model,
            "checkpoint_revision": settings.reid_checkpoint_revision,
            "preprocess_version": settings.reid_preprocess_version,
            "embedding_dim": settings.reid_embedding_dim,
            "inference_batch_size": 1,
        }
    results: list[dict[str, object]] = []
    try:
        with readonly_database(args.database) as connection:
            for crop_id in crop_ids:
                try:
                    crop = load_crop(connection, crop_id)
                    results.append(diagnose_crop(crop, args.data_dir, encoder))
                except (DiagnosticError, OSError, sqlite3.Error, cv2.error) as exc:
                    reason = str(exc) if isinstance(exc, DiagnosticError) else type(exc).__name__
                    results.append({"crop_id": str(crop_id), "status": "failed", "error": reason})
    except sqlite3.Error:
        print(json.dumps({"status": "failed", "error": "database_unavailable"}))
        return 1
    report = {
        "schema_version": 1,
        "mode": "quality_and_vector_drift" if encoder is not None else "quality_only",
        "sample_count": len(results),
        "encoder_identity": identity,
        "interpretation": (
            "Input sensitivity only, not accuracy or identity evidence. Native sharpness depends "
            "on dimensions; the 96x192 metric stretches both crops to a common diagnostic grid. "
            "Stored display crops may include padding, resampling, sharpening, JPEG artifacts, "
            "or legacy preprocessing. This comparison does not isolate these factors. No DB, "
            "Milvus, settings, or historical image changes are made."
        ),
        "rows": results,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return int(any(row["status"] in {"failed", "inference_failed"} for row in results))


if __name__ == "__main__":
    raise SystemExit(main())
