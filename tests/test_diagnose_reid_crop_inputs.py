"""Synthetic-only safety regressions for the isolated crop-input diagnostic."""

from __future__ import annotations

import io
import json
import sqlite3
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib import request

import cv2
import numpy as np
import pytest

from scripts import diagnose_reid_crop_inputs as diagnostic


@pytest.fixture
def sample(tmp_path: Path) -> tuple[Path, Path, diagnostic.CropInput]:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    frame = np.zeros((120, 100, 3), dtype=np.uint8)
    frame[20:100, 20:60] = (25, 170, 220)
    frame[30:90:4, 25:55] = 255
    assert cv2.imwrite(str(data_dir / "frame.png"), frame)
    display = cv2.resize(frame[10:110, 10:70], (120, 200))
    assert cv2.imwrite(str(data_dir / "display.png"), display)
    crop_id, image_id = uuid.uuid4(), uuid.uuid4()
    bbox = {"x": 20, "y": 20, "width": 40, "height": 80}
    database = tmp_path / "db.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE images (id TEXT PRIMARY KEY, image_url TEXT)")
        connection.execute(
            "CREATE TABLE person_crops "
            "(id TEXT PRIMARY KEY, image_id TEXT, crop_url TEXT, bbox TEXT)"
        )
        connection.execute("INSERT INTO images VALUES (?, ?)", (image_id.hex, "/data/frame.png"))
        connection.execute(
            "INSERT INTO person_crops VALUES (?, ?, ?, ?)",
            (crop_id.hex, image_id.hex, "/data/display.png", json.dumps(bbox)),
        )
    return (
        database,
        data_dir,
        diagnostic.CropInput(crop_id, "/data/frame.png", "/data/display.png", bbox),
    )


def test_readonly_database_rejects_writes_and_never_creates_missing_file(tmp_path: Path) -> None:
    existing = tmp_path / "existing.sqlite"
    sqlite3.connect(existing).close()
    with diagnostic.readonly_database(existing) as connection:
        with pytest.raises(sqlite3.OperationalError):
            connection.execute("CREATE TABLE forbidden (id INTEGER)")
        assert connection.execute("PRAGMA query_only").fetchone() == (1,)
    missing = tmp_path / "missing.sqlite"
    with pytest.raises(sqlite3.OperationalError), diagnostic.readonly_database(missing):
        pass
    assert not missing.exists()


def test_load_crop_reads_only_requested_row(
    sample: tuple[Path, Path, diagnostic.CropInput],
) -> None:
    database, _data_dir, crop = sample
    with diagnostic.readonly_database(database) as connection:
        assert diagnostic.load_crop(connection, crop.crop_id) == crop
        with pytest.raises(diagnostic.DiagnosticError, match="crop_not_found"):
            diagnostic.load_crop(connection, uuid.uuid4())


@pytest.mark.parametrize("bbox", [None, "not-json"])
def test_load_rejects_malformed_bbox_json(
    sample: tuple[Path, Path, diagnostic.CropInput], bbox: object
) -> None:
    database, _data_dir, crop = sample
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE person_crops SET bbox = ?", (bbox,))
    with diagnostic.readonly_database(database) as connection:
        with pytest.raises(diagnostic.DiagnosticError, match="invalid_bbox_json"):
            diagnostic.load_crop(connection, crop.crop_id)


def test_load_rejects_missing_original_frame_reference(
    sample: tuple[Path, Path, diagnostic.CropInput],
) -> None:
    database, _data_dir, crop = sample
    with sqlite3.connect(database) as connection:
        connection.execute("DELETE FROM images")
    with diagnostic.readonly_database(database) as connection:
        with pytest.raises(diagnostic.DiagnosticError, match="missing_media_reference"):
            diagnostic.load_crop(connection, crop.crop_id)


@pytest.mark.parametrize(
    "url",
    [
        "https://external/image.jpg",
        "/etc/passwd",
        "/data/a?key=x",
        "/data/%2e%2e/secret",
        "/data/a#x",
    ],
)
def test_local_data_path_rejects_nonliteral_urls(tmp_path: Path, url: str) -> None:
    with pytest.raises(diagnostic.DiagnosticError, match="invalid_data_url"):
        diagnostic.local_data_path(tmp_path, url)


def test_local_data_path_rejects_traversal_and_symlink_escape(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    private = tmp_path / "secret"
    private.touch()
    (data_dir / "symlink").symlink_to(private)
    for url in ("/data/../secret", "/data/symlink", "/data//etc/passwd"):
        with pytest.raises(diagnostic.DiagnosticError, match="media_outside_data_directory"):
            diagnostic.local_data_path(data_dir, url)
    with pytest.raises(diagnostic.DiagnosticError, match="media_file_missing"):
        diagnostic.local_data_path(data_dir, "/data/missing")


def test_decode_failure_never_becomes_full_frame(tmp_path: Path) -> None:
    path = tmp_path / "invalid.png"
    path.touch()
    with pytest.raises(diagnostic.DiagnosticError, match="image_decode_failed"):
        diagnostic.read_image(path)


@pytest.mark.parametrize(
    "bbox",
    [
        None,
        {},
        {"x": 0, "y": 0, "width": 0, "height": 5},
        {"x": 0, "y": 0, "width": -1, "height": 5},
        {"x": True, "y": 0, "width": 2, "height": 5},
        {"x": "0", "y": 0, "width": 2, "height": 5},
        {"x": float("nan"), "y": 0, "width": 2, "height": 5},
        {"x": 0, "y": 0, "width": 2, "height": float("inf")},
        {"x": 100, "y": 0, "width": 2, "height": 5},
        {"x": -100, "y": 0, "width": 2, "height": 5},
    ],
)
def test_invalid_bbox_never_falls_back_to_original_image(bbox: object) -> None:
    with pytest.raises(diagnostic.DiagnosticError):
        diagnostic.detector_roi(np.zeros((20, 20, 3), dtype=np.uint8), bbox)


def test_partly_clipped_detector_bbox_uses_intersection_not_shifted_box() -> None:
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    roi, retained = diagnostic.detector_roi(image, {"x": -5, "y": 2, "width": 10, "height": 15})
    assert roi.shape == (15, 5, 3)
    assert retained == 0.5
    assert not np.shares_memory(roi, image)


def test_quality_report_contains_descriptors_not_media_or_vectors(
    sample: tuple[Path, Path, diagnostic.CropInput],
) -> None:
    _database, data_dir, crop = sample
    report = diagnostic.diagnose_crop(crop, data_dir)
    assert report["status"] == "quality_only"
    assert report["inference"] is None
    assert report["raw_detector_roi"]["width"] == 40
    assert report["raw_detector_roi"]["height"] == 80
    assert report["stored_display_crop"]["width"] == 120
    assert report["display_equals_full_frame_pixels"] is False
    output = json.dumps(report, allow_nan=False)
    for forbidden in ("base64", "embedding", "/data/", str(data_dir)):
        assert forbidden not in output


def test_full_frame_display_fallback_is_flagged_without_inference(
    sample: tuple[Path, Path, diagnostic.CropInput],
) -> None:
    _database, data_dir, crop = sample
    fallback_crop = diagnostic.CropInput(crop.crop_id, crop.frame_url, crop.frame_url, crop.bbox)
    assert diagnostic.diagnose_crop(fallback_crop, data_dir)["display_equals_full_frame_pixels"]


@pytest.mark.parametrize("fail_at", [None, 1, 2])
def test_temporary_roi_removed_after_success_or_each_inference_failure(
    sample: tuple[Path, Path, diagnostic.CropInput], fail_at: int | None
) -> None:
    _database, data_dir, crop = sample
    paths: list[Path] = []

    class StubEncoder:
        def embed_image(self, image_path: Path) -> list[float]:
            assert image_path.is_file()
            paths.append(image_path)
            if fail_at == len(paths):
                raise RuntimeError("private request contents must not appear")
            return [1.0, 0.0] if len(paths) == 1 else [0.6, 0.8]

    original = (data_dir / "display.png").read_bytes()
    report = diagnostic.diagnose_crop(crop, data_dir, StubEncoder())
    assert paths and not paths[0].parent.exists()
    assert (data_dir / "display.png").read_bytes() == original
    if fail_at is None:
        assert report["inference"] == {"raw_display_cosine": 0.6, "cosine_distance_drift": 0.4}
        assert report["status"] == "inferred"
    else:
        assert report["status"] == "inference_failed"
        assert report["error"] == "RuntimeError"
        assert "private" not in json.dumps(report)


def test_failed_temp_image_write_is_cleaned_and_does_not_infer(
    sample: tuple[Path, Path, diagnostic.CropInput], monkeypatch: pytest.MonkeyPatch
) -> None:
    _database, data_dir, crop = sample
    paths: list[Path] = []

    def failed_write(path: str, _image: object) -> bool:
        paths.append(Path(path))
        return False

    monkeypatch.setattr(cv2, "imwrite", failed_write)
    report = diagnostic.diagnose_crop(crop, data_dir, object())
    assert report["error"] == "temporary_roi_write_failed"
    assert not paths[0].parent.exists()


@pytest.mark.parametrize(
    "left,right",
    [
        ([], []),
        ([1], [1, 2]),
        ([0], [0]),
        ([float("nan")], [1]),
        ([1], [float("inf")]),
        ([[1]], [[1]]),
    ],
)
def test_invalid_vectors_are_not_drift_evidence(left: list[Any], right: list[Any]) -> None:
    with pytest.raises(diagnostic.DiagnosticError):
        diagnostic.vector_cosine(left, right)


def test_cosine_normalizes_vectors_without_overflow() -> None:
    assert diagnostic.vector_cosine([1e300, 0], [3, 4]) == pytest.approx(0.6)


@pytest.mark.parametrize(
    "url",
    [
        "http://198.51.100.25:18031",
        "https://127.0.0.1:18031",
        "http://127.0.0.1.evil.test:18031",
        "http://user:secret@127.0.0.1:18031",
        "http://127.0.0.1:18031?secret=1",
        "http://127.0.0.1:18031#fragment",
        "http://127.0.0.1:18031/external-relay",
        "http://127.0.0.1:notaport",
        "",
    ],
)
def test_inference_cannot_send_images_to_remote_or_ambiguous_endpoint(url: str) -> None:
    with pytest.raises(diagnostic.DiagnosticError):
        diagnostic.validate_loopback_url(url)


@pytest.mark.parametrize(
    "url", ["http://127.0.0.1:18031", "http://[::1]:18031/", "http://localhost:18031"]
)
def test_loopback_endpoint_is_allowed(url: str) -> None:
    diagnostic.validate_loopback_url(url)


def test_redirect_is_not_followed() -> None:
    assert (
        diagnostic.NoRedirect().redirect_request(
            request.Request("http://127.0.0.1/embed"), None, 307, "redirect", {}, "http://external/"
        )
        is None
    )


def test_cli_default_does_not_initialize_encoder_or_change_database(
    sample: tuple[Path, Path, diagnostic.CropInput],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database, data_dir, crop = sample

    def forbidden_encoder(*_args: object) -> None:
        pytest.fail("Quality-only mode must not instantiate any encoder")

    monkeypatch.setattr(diagnostic, "LoopbackEncoder", forbidden_encoder)
    before = database.read_bytes()
    assert (
        diagnostic.main(
            [
                "--database",
                str(database),
                "--data-dir",
                str(data_dir),
                "--crop-id",
                str(crop.crop_id),
            ]
        )
        == 0
    )
    assert database.read_bytes() == before
    output = json.loads(capsys.readouterr().out)
    assert output["mode"] == "quality_only"
    assert output["sample_count"] == 1
    assert output["encoder_identity"] is None
    assert "not accuracy" in output["interpretation"]


def test_cli_reports_partial_failure_without_leaking_paths(
    sample: tuple[Path, Path, diagnostic.CropInput], capsys: pytest.CaptureFixture[str]
) -> None:
    database, data_dir, crop = sample
    assert (
        diagnostic.main(
            [
                "--database",
                str(database),
                "--data-dir",
                str(data_dir),
                "--crop-id",
                str(crop.crop_id),
                "--crop-id",
                str(uuid.uuid4()),
            ]
        )
        == 1
    )
    output = capsys.readouterr().out
    report = json.loads(output)
    assert report["rows"][1]["error"] == "crop_not_found"
    assert str(database) not in output
    assert str(data_dir) not in output


def test_cli_missing_database_is_not_created(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    database = tmp_path / "missing.sqlite"
    assert (
        diagnostic.main(
            [
                "--database",
                str(database),
                "--data-dir",
                str(tmp_path),
                "--crop-id",
                str(uuid.uuid4()),
            ]
        )
        == 1
    )
    assert not database.exists()
    assert json.loads(capsys.readouterr().out)["error"] == "database_unavailable"


def test_cli_limits_total_requested_ids_before_io(tmp_path: Path) -> None:
    args = ["--database", str(tmp_path / "none"), "--data-dir", str(tmp_path)]
    for _ in range(11):
        args.extend(["--crop-id", str(uuid.uuid4())])
    with pytest.raises(SystemExit) as error:
        diagnostic.main(args)
    assert error.value.code == 2


def test_encoder_rejects_disabled_model() -> None:
    with pytest.raises(diagnostic.DiagnosticError, match="reid_disabled"):
        diagnostic.LoopbackEncoder(
            SimpleNamespace(reid_service_url="http://127.0.0.1:18031", reid_enabled=False)
        )


@pytest.fixture
def encoder_settings() -> SimpleNamespace:
    return SimpleNamespace(
        reid_service_url="http://localhost:18031",
        reid_enabled=True,
        reid_service_api_key="synthetic-secret",
        reid_timeout_seconds=100,
        reid_model="test-model",
        reid_checkpoint_revision="test-revision",
        reid_preprocess_version="test-preprocess",
        reid_embedding_dim=2,
    )


@pytest.mark.parametrize(
    "changes,error_code",
    [
        ({}, None),
        ({"model": "another-model"}, "inference_identity_mismatch"),
        ({"preprocess_version": "another-preprocess"}, "inference_identity_mismatch"),
        ({"embedding": [1.0]}, "incompatible_inference_vectors"),
        ({"embedding": [True, 0]}, "invalid_inference_vector"),
        ({"embedding": ["1", 0]}, "invalid_inference_vector"),
    ],
)
def test_encoder_disables_proxies_redirects_and_validates_each_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    encoder_settings: SimpleNamespace,
    changes: dict[str, object],
    error_code: str | None,
) -> None:
    path = tmp_path / "synthetic.png"
    assert cv2.imwrite(str(path), np.zeros((10, 10, 3), dtype=np.uint8))
    payload = {
        "model": "test-model",
        "checkpoint_revision": "test-revision",
        "preprocess_version": "test-preprocess",
        "embedding_dim": 2,
        "embedding": [1, 0],
        **changes,
    }

    class StubOpener:
        def open(self, req: request.Request, timeout: float) -> io.BytesIO:
            assert req.full_url == "http://127.0.0.1:18031/embed"
            assert timeout == 30.0
            assert req.get_header("Authorization") == "Bearer synthetic-secret"
            assert set(json.loads(req.data)) == {"image_base64"}
            return io.BytesIO(json.dumps(payload).encode())

    def build_opener(*handlers: object) -> StubOpener:
        assert len(handlers) == 2
        assert isinstance(handlers[0], request.ProxyHandler)
        assert handlers[0].proxies == {}
        assert isinstance(handlers[1], diagnostic.NoRedirect)
        return StubOpener()

    monkeypatch.setattr(diagnostic.request, "build_opener", build_opener)
    encoder = diagnostic.LoopbackEncoder(encoder_settings)
    if error_code:
        with pytest.raises(diagnostic.DiagnosticError, match=error_code):
            encoder.embed_image(path)
    else:
        assert encoder.embed_image(path) == [1.0, 0.0]


def test_cli_explicit_inference_records_identity_and_sanitized_drift_only(
    sample: tuple[Path, Path, diagnostic.CropInput],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    encoder_settings: SimpleNamespace,
) -> None:
    from app.config import settings

    database, data_dir, crop = sample
    calls: list[Path] = []

    class StubEncoder:
        def embed_image(self, image_path: Path) -> list[float]:
            calls.append(image_path)
            return [0, 1]

    monkeypatch.setattr(settings, "get_settings", lambda: encoder_settings)
    monkeypatch.setattr(diagnostic, "LoopbackEncoder", lambda _settings: StubEncoder())
    assert (
        diagnostic.main(
            [
                "--database",
                str(database),
                "--data-dir",
                str(data_dir),
                "--crop-id",
                str(crop.crop_id),
                "--allow-inference",
            ]
        )
        == 0
    )
    assert len(calls) == 2
    output = capsys.readouterr().out
    report = json.loads(output)
    assert report["mode"] == "quality_and_vector_drift"
    assert report["encoder_identity"]["inference_batch_size"] == 1
    assert report["rows"][0]["inference"]["cosine_distance_drift"] == 0.0
    assert "synthetic-secret" not in output
    assert "localhost" not in output
    assert "image_base64" not in output
    assert not calls[0].parent.exists()
