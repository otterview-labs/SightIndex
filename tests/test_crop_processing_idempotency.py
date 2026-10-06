"""Technical crop failures stay retryable across actual image-processing requests."""

import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_api_smoke import load_app


@pytest.mark.parametrize("failure", ["false", "undecodable", "exception"])
def test_crop_failure_keeps_api_retry_open_then_success_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sample_jpeg: bytes,
    failure: str,
) -> None:
    """A transient encoder failure must not permanently complete a person frame."""

    for name, value in {
        "MILVUS_ENABLED": "false",
        "REID_ENABLED": "false",
        "VECTOR_INDEX_ON_INGEST": "false",
        "VLM_STRUCTURED_ON_INGEST": "false",
        "APPEARANCE_TONE_ON_INGEST": "false",
        "STREAM_AUTOSTART_RUNNING": "false",
        "APP_BASIC_AUTH_USERNAME": "",
        "APP_BASIC_AUTH_PASSWORD": "",
    }.items():
        monkeypatch.setenv(name, value)
    main = load_app(monkeypatch, tmp_path, f"crop-retry-{failure}")

    from app.db.session import SessionLocal
    from app.models.media import Image, PersonCrop
    from app.models.vectors import VectorIndexJob
    from app.services.frame_processing import Detection, FrameProcessingService

    original_writer = FrameProcessingService._try_crop_with_cv2
    attempted: list[Path] = []

    def failed_writer(
        _service: FrameProcessingService,
        _source: Path,
        target: Path,
        _detection: Detection,
    ) -> bool:
        attempted.append(target)
        target.write_bytes(b"partial undecodable encoder output")
        if failure == "exception":
            raise OSError("transient encoder failure")
        return failure == "undecodable"

    monkeypatch.setattr(FrameProcessingService, "_try_crop_with_cv2", failed_writer)
    with TestClient(main.create_app()) as client:
        uploaded = client.post(
            "/api/images/upload", files={"file": ("person.jpg", sample_jpeg, "image/jpeg")}
        )
        assert uploaded.status_code == 200
        image_id = uuid.UUID(uploaded.json()["id"])
        endpoint = f"/api/images/{image_id}/process"
        failed = client.post(endpoint)
        assert failed.status_code == 200
        assert failed.json() == []
        assert attempted and all(not path.exists() for path in attempted)
        assert list(main.get_settings().thumbnails_dir.iterdir()) == []
        with SessionLocal() as db:
            image = db.get(Image, image_id)
            assert image is not None and image.processed_at is None
            assert image.thumbnail_url is None
            assert db.query(PersonCrop).count() == 0
            assert db.query(VectorIndexJob).count() == 0

        monkeypatch.setattr(FrameProcessingService, "_try_crop_with_cv2", original_writer)
        successful = client.post(endpoint)
        assert successful.status_code == 200
        assert len(successful.json()) == 1
        files = {path for path in main.get_settings().data_dir.rglob("*") if path.is_file()}
        repeated = client.post(endpoint)
        assert repeated.status_code == 200
        assert [crop["id"] for crop in repeated.json()] == [successful.json()[0]["id"]]
        assert {path for path in main.get_settings().data_dir.rglob("*") if path.is_file()} == files
        with SessionLocal() as db:
            image = db.get(Image, image_id)
            assert image is not None and image.processed_at is not None
            assert db.query(PersonCrop).count() == 1
