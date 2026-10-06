"""Crop failures must abstain without turning scenes into indexed person crops."""

from __future__ import annotations

import sys
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import Mock

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from app.config.settings import Settings
from app.db.session import Base
from app.models import events, persons  # noqa: F401
from app.models.media import Image, PersonCrop
from app.models.vectors import VectorIndexJob
from app.services import frame_processing
from app.services.frame_processing import (
    Detection,
    FrameProcessingService,
    WholeFramePersonDetector,
)


@dataclass
class CropFixture:
    service: FrameProcessingService
    db: Session
    source: Path
    image: Image
    detection: Detection


@pytest.fixture
def crop_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[CropFixture]:
    """Keep persistence real and image decoding/model inference outside these write tests."""

    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    source = tmp_path / "source.jpg"
    source.write_bytes(b"original source must never become a fallback person crop")
    settings = Settings(
        data_dir=tmp_path,
        appearance_tone_on_ingest=True,
        vlm_structured_on_ingest=True,
        vlm_structured_background=False,
        vector_index_on_ingest=True,
    )
    with Session(engine) as db:
        image = Image(image_url="/data/source.jpg", source_type="stream_frame")
        db.add(image)
        db.commit()
        service = FrameProcessingService(db, settings, detector=WholeFramePersonDetector())
        monkeypatch.setattr(service, "_read_image_size", lambda _url: (400, 400))
        monkeypatch.setattr(service, "_create_annotated_frame_file", lambda *_args: None)
        for method in (
            "_try_recognize_faces",
            "_try_read_clothing_tone",
            "_try_analyze_crop_attributes",
            "_try_index_crops",
        ):
            monkeypatch.setattr(service, method, Mock())
        yield CropFixture(
            service,
            db,
            source,
            image,
            Detection({"x": 30, "y": 40, "width": 80, "height": 160}, 0.95),
        )
    engine.dispose()


@pytest.mark.parametrize("partial", [False, True])
def test_failed_writer_never_copies_source_and_removes_only_its_output(
    crop_fixture: CropFixture, monkeypatch: pytest.MonkeyPatch, partial: bool
) -> None:
    service, source = crop_fixture.service, crop_fixture.source
    service.settings.crops_dir.mkdir()
    previous = service.settings.crops_dir / "previous.jpg"
    previous.write_bytes(b"existing historical crop")
    attempted: list[Path] = []

    def fail(_source: Path, target: Path, _detection: Detection) -> bool:
        attempted.append(target)
        if partial:
            target.write_bytes(b"partial output")
        return False

    monkeypatch.setattr(service, "_try_crop_with_cv2", fail)
    original = source.read_bytes()
    assert service._create_crop_file(source, crop_fixture.detection) is None
    assert len(attempted) == 1 and not attempted[0].exists()
    assert source.read_bytes() == original
    assert previous.read_bytes() == b"existing historical crop"
    assert list(service.settings.crops_dir.iterdir()) == [previous]


@pytest.mark.parametrize("failure", [OSError("disk failure"), RuntimeError("writer failure")])
def test_partial_crop_is_cleaned_when_writer_raises(
    crop_fixture: CropFixture, monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    service = crop_fixture.service
    attempted: list[Path] = []

    def fail(_source: Path, target: Path, _detection: Detection) -> bool:
        attempted.append(target)
        target.write_bytes(b"partial output")
        raise failure

    monkeypatch.setattr(service, "_try_crop_with_cv2", fail)
    assert service._create_crop_file(crop_fixture.source, crop_fixture.detection) is None
    assert not attempted[0].exists()


def test_empty_successful_write_is_rejected(
    crop_fixture: CropFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = crop_fixture.service
    monkeypatch.setattr(service, "_try_crop_with_cv2", lambda *_args: True)
    assert service._create_crop_file(crop_fixture.source, crop_fixture.detection) is None
    assert list(service.settings.crops_dir.iterdir()) == []


def test_target_collision_preserves_existing_file(
    crop_fixture: CropFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = crop_fixture.service
    existing_id = uuid.uuid4()
    service.settings.crops_dir.mkdir()
    existing = service.settings.crops_dir / f"{existing_id}.jpg"
    existing.write_bytes(b"existing historical crop")
    monkeypatch.setattr(frame_processing.uuid, "uuid4", lambda: existing_id)
    writer = Mock(return_value=False)
    monkeypatch.setattr(service, "_try_crop_with_cv2", writer)
    assert service._create_crop_file(crop_fixture.source, crop_fixture.detection) is None
    writer.assert_not_called()
    assert existing.read_bytes() == b"existing historical crop"


def test_missing_vision_backend_skips_crop_without_full_frame_fallback(
    crop_fixture: CropFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "cv2", None)
    service = crop_fixture.service
    assert service._create_crop_file(crop_fixture.source, crop_fixture.detection) is None
    assert list(service.settings.crops_dir.iterdir()) == []


def test_failed_image_decode_skips_crop(
    crop_fixture: CropFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    cv2 = pytest.importorskip("cv2")
    monkeypatch.setattr(cv2, "imread", lambda _path: None)
    service = crop_fixture.service
    assert service._create_crop_file(crop_fixture.source, crop_fixture.detection) is None
    assert list(service.settings.crops_dir.iterdir()) == []


def test_opencv_exception_does_not_escape_or_leave_partial_crop(
    crop_fixture: CropFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    service = crop_fixture.service
    monkeypatch.setattr(cv2, "imread", lambda _path: np.zeros((400, 400, 3), dtype=np.uint8))

    def fail(_cv2: object, target: Path, _image: object, _quality: int) -> bool:
        target.write_bytes(b"partial output")
        raise cv2.error("synthetic encoder error")

    monkeypatch.setattr(service, "_write_jpeg", fail)
    assert service._create_crop_file(crop_fixture.source, crop_fixture.detection) is None
    assert list(service.settings.crops_dir.iterdir()) == []


def test_all_crops_failed_creates_no_rows_jobs_or_face_inference(
    crop_fixture: CropFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, db = crop_fixture.service, crop_fixture.db
    monkeypatch.setattr(service, "_try_crop_with_cv2", lambda *_args: False)
    enqueue = Mock(side_effect=AssertionError("failed crops must not enqueue any jobs"))
    monkeypatch.setattr(service, "_enqueue_index_jobs", enqueue)

    assert service.process_image(crop_fixture.image, [crop_fixture.detection]) == []

    enqueue.assert_not_called()
    assert db.scalar(select(func.count()).select_from(PersonCrop)) == 0
    assert db.scalar(select(func.count()).select_from(VectorIndexJob)) == 0
    for method in (
        "_try_recognize_faces",
        "_try_read_clothing_tone",
        "_try_analyze_crop_attributes",
        "_try_index_crops",
    ):
        getattr(service, method).assert_not_called()
    assert crop_fixture.source.exists()
    assert list(service.settings.crops_dir.iterdir()) == []
    assert crop_fixture.image.processed_at is None


def test_written_but_undecodable_crop_is_removed_without_rows_or_jobs(
    crop_fixture: CropFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, db = crop_fixture.service, crop_fixture.db

    def writer(_source: Path, target: Path, _detection: Detection) -> bool:
        target.write_bytes(b"not a decodable crop despite writer success")
        return True

    monkeypatch.setattr(service, "_try_crop_with_cv2", writer)
    monkeypatch.setattr(
        service,
        "_read_image_size",
        lambda url: (400, 400) if url == crop_fixture.image.image_url else (None, None),
    )
    enqueue = Mock(return_value=False)
    monkeypatch.setattr(service, "_enqueue_index_jobs", enqueue)
    assert service.process_image(crop_fixture.image, [crop_fixture.detection]) == []
    assert db.scalar(select(func.count()).select_from(PersonCrop)) == 0
    enqueue.assert_not_called()
    service._try_recognize_faces.assert_not_called()
    assert list(service.settings.crops_dir.iterdir()) == []
    assert crop_fixture.image.processed_at is None


def test_crop_directory_failure_does_not_remove_existing_path(
    crop_fixture: CropFixture,
) -> None:
    service = crop_fixture.service
    service.settings.crops_dir.write_bytes(b"unrelated existing file")
    assert service._create_crop_file(crop_fixture.source, crop_fixture.detection) is None
    assert service.settings.crops_dir.read_bytes() == b"unrelated existing file"


@pytest.mark.parametrize("failed_first", [True, False])
def test_one_crop_failure_does_not_lose_successful_detection_or_change_source_index(
    crop_fixture: CropFixture, monkeypatch: pytest.MonkeyPatch, failed_first: bool
) -> None:
    service, db = crop_fixture.service, crop_fixture.db
    failed = Detection(crop_fixture.detection.bbox, 0.95, source_index=3)
    success = Detection(
        {"x": 160, "y": 40, "width": 80, "height": 160, "source_detection_index": 999},
        0.95,
        source_index=8,
    )
    attempted: dict[int, Path] = {}

    def writer(_source: Path, target: Path, detection: Detection) -> bool:
        assert detection.source_index is not None
        attempted[detection.source_index] = target
        target.write_bytes(b"partial" if detection is failed else b"successful synthetic crop")
        return detection is success

    enqueued_ids: list[uuid.UUID] = []

    def enqueue(_image: Image, crops: list[PersonCrop]) -> bool:
        db.flush()
        for crop in crops:
            enqueued_ids.append(crop.id)
            db.add(VectorIndexJob(target="reid_person_crop", object_id=crop.id))
        return False  # Do not start background workers in this transaction test.

    monkeypatch.setattr(service, "_try_crop_with_cv2", writer)
    monkeypatch.setattr(service, "_enqueue_index_jobs", enqueue)
    detections = [failed, success] if failed_first else [success, failed]
    crops = service.process_image(crop_fixture.image, detections)

    assert len(crops) == 1
    assert crops[0].bbox["source_detection_index"] == 8
    assert crops[0].bbox["x"] == 160
    assert db.scalar(select(func.count()).select_from(PersonCrop)) == 1
    assert db.scalar(select(func.count()).select_from(VectorIndexJob)) == 1
    assert enqueued_ids == [crops[0].id]
    assert not attempted[3].exists()
    assert attempted[8].read_bytes() == b"successful synthetic crop"
    service._try_recognize_faces.assert_called_once_with(crop_fixture.image, crops)
    assert crop_fixture.source.exists()
    assert crop_fixture.image.processed_at is not None

    # A partial success has already committed its output and must never duplicate it.
    completed_at = crop_fixture.image.processed_at
    repeated = service.process_image(crop_fixture.image, detections)
    assert [crop.id for crop in repeated] == [crops[0].id]
    assert crop_fixture.image.processed_at == completed_at
    assert db.scalar(select(func.count()).select_from(PersonCrop)) == 1
    assert db.scalar(select(func.count()).select_from(VectorIndexJob)) == 1
    assert enqueued_ids == [crops[0].id]
    service._try_recognize_faces.assert_called_once_with(crop_fixture.image, crops)


def test_detector_payload_cannot_inject_source_index_without_explicit_association(
    crop_fixture: CropFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = crop_fixture.service

    def writer(_source: Path, target: Path, _detection: Detection) -> bool:
        target.write_bytes(b"successful synthetic crop")
        return True

    monkeypatch.setattr(service, "_try_crop_with_cv2", writer)
    monkeypatch.setattr(service, "_enqueue_index_jobs", lambda *_args: False)
    detection = Detection({**crop_fixture.detection.bbox, "source_detection_index": 999}, 0.95)
    crops = service.process_image(crop_fixture.image, [detection])
    assert len(crops) == 1
    assert "source_detection_index" not in crops[0].bbox


def test_genuinely_empty_detection_result_keeps_existing_image_only_enqueue_path(
    crop_fixture: CropFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    enqueue = Mock(return_value=False)
    monkeypatch.setattr(crop_fixture.service, "_enqueue_index_jobs", enqueue)
    assert crop_fixture.service.process_image(crop_fixture.image, []) == []
    enqueue.assert_called_once_with(crop_fixture.image, [])
    assert crop_fixture.image.processed_at is not None
    assert crop_fixture.service.process_image(crop_fixture.image, []) == []
    enqueue.assert_called_once_with(crop_fixture.image, [])


def test_annotation_target_collision_preserves_existing_file(
    crop_fixture: CropFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    service = crop_fixture.service
    existing_id = uuid.uuid4()
    service.settings.thumbnails_dir.mkdir()
    existing = service.settings.thumbnails_dir / f"{existing_id}.jpg"
    existing.write_bytes(b"existing historical thumbnail")
    monkeypatch.setattr(frame_processing.uuid, "uuid4", lambda: existing_id)
    monkeypatch.setattr(cv2, "imread", lambda _path: np.zeros((400, 400, 3), dtype=np.uint8))
    writer = Mock(return_value=False)
    monkeypatch.setattr(service, "_write_jpeg", writer)
    with pytest.raises(FileExistsError):
        # Invoke the real implementation; the fixture disables annotations for crop-only tests.
        FrameProcessingService._create_annotated_frame_file(
            service, crop_fixture.source, [crop_fixture.detection]
        )
    writer.assert_not_called()
    assert existing.read_bytes() == b"existing historical thumbnail"
