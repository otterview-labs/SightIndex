"""Zero-crop results must explain themselves, and a reprocess must be opt-in per image.

A photo whose person runs off the frame used to look exactly like a broken pipeline: the
console said "0 个裁剪候选" and nothing else, and the row was already marked processed. These
tests pin the stored reason to the gate rule that actually ran, and pin the rule that a
reprocess recovers that result without ever duplicating existing crops.
"""

from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

from app.config.settings import Settings
from app.db.session import Base
from app.models import persons  # noqa: F401
from app.models.media import Image, PersonCrop
from app.services.frame_processing import Detection, FrameProcessingService


class _StubDetector:
    def __init__(self, detections: list[Detection]) -> None:
        self.detections = detections
        self.calls = 0

    def detect(self, image_path):  # noqa: ANN001 - mirrors PersonDetector.detect
        self.calls += 1
        return list(self.detections)


# Frame is 300x400 (w x h) and the edge margin is 6px.
CLIPPED = Detection(bbox={"x": 0, "y": 40, "width": 300, "height": 360}, confidence=0.93)
WHOLE_BODY = Detection(bbox={"x": 60, "y": 20, "width": 120, "height": 340}, confidence=0.93)


@pytest.fixture()
def env(tmp_path):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    engine = create_engine(f"sqlite:///{tmp_path / 'crop-reason.db'}")
    Base.metadata.create_all(bind=engine)
    db = sessionmaker(bind=engine)()
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    path = uploads / "frame.jpg"
    assert cv2.imwrite(str(path), np.zeros((400, 300, 3), dtype=np.uint8))
    settings = Settings(
        data_dir=tmp_path,
        person_detector="whole_frame",
        person_crop_min_confidence=0.0,
        person_crop_min_bbox_width=0,
        person_crop_min_bbox_height=0,
        person_crop_require_whole_body=True,
        person_crop_edge_margin=6,
        person_crop_upscale_min_width=0,
        person_crop_upscale_min_height=0,
        person_crop_sharpen_amount=0.0,
    )
    image = Image(image_url="/data/uploads/frame.jpg", source_type="upload")
    db.add(image)
    db.commit()
    db.refresh(image)
    try:
        yield db, settings, image, path
    finally:
        db.close()


def test_edge_clipped_person_records_the_rule_that_dropped_it(env):
    db, settings, image, _path = env

    crops = FrameProcessingService(
        db, settings, detector=_StubDetector([CLIPPED])
    ).process_image(image)

    assert crops == []
    db.refresh(image)
    reason = image.crop_reason
    assert reason["reason"] == "quality_filtered"
    assert reason["raw_total"] == 1
    assert reason["rejected"] == {"touches_frame_edge": 1}
    assert reason["thresholds"]["require_whole_body"] is True
    assert reason["thresholds"]["edge_margin"] == 6
    assert reason["frame"] == {"width": 300, "height": 400}
    gaps = reason["samples"][0]["edge_gaps_px"]
    assert gaps["left"] == 0
    assert gaps["bottom"] == 0


def test_nothing_detected_is_reported_as_such(env):
    db, settings, image, _path = env

    crops = FrameProcessingService(db, settings, detector=_StubDetector([])).process_image(image)

    assert crops == []
    db.refresh(image)
    assert image.crop_reason["reason"] == "no_raw_detections"
    assert image.crop_reason["raw_total"] == 0


def test_a_zero_result_is_sticky_until_forced(env):
    db, settings, image, _path = env
    first = _StubDetector([CLIPPED])
    FrameProcessingService(db, settings, detector=first).process_image(image)
    assert first.calls == 1

    second = _StubDetector([WHOLE_BODY])
    crops = FrameProcessingService(db, settings, detector=second).process_image(image)

    assert crops == []
    assert second.calls == 0  # idempotent: no second detection run
    db.refresh(image)
    assert image.crop_reason["reason"] == "quality_filtered"


def test_force_reruns_and_recovers_crops(env):
    db, settings, image, _path = env
    FrameProcessingService(db, settings, detector=_StubDetector([CLIPPED])).process_image(image)

    crops = FrameProcessingService(
        db, settings, detector=_StubDetector([WHOLE_BODY])
    ).process_image(image, force=True)

    assert len(crops) == 1
    db.refresh(image)
    assert image.crop_reason is None  # a crop exists; there is nothing to explain
    assert db.scalar(select(func.count()).select_from(PersonCrop)) == 1


def test_force_never_duplicates_existing_crops(env):
    db, settings, image, _path = env
    FrameProcessingService(db, settings, detector=_StubDetector([WHOLE_BODY])).process_image(image)

    again = _StubDetector([WHOLE_BODY])
    crops = FrameProcessingService(db, settings, detector=again).process_image(image, force=True)

    # Recognition events, the person index and vector jobs already point at these crops, so a
    # forced call must return them rather than insert a second set.
    assert len(crops) == 1
    assert again.calls == 0
    assert db.scalar(select(func.count()).select_from(PersonCrop)) == 1


def test_missing_source_file_is_explained(env):
    db, settings, image, path = env
    path.unlink()

    crops = FrameProcessingService(
        db, settings, detector=_StubDetector([WHOLE_BODY])
    ).process_image(image)

    assert crops == []
    db.refresh(image)
    assert image.crop_reason == {"reason": "source_missing"}


def test_legacy_processed_row_gets_a_reprocess_hint(env):
    db, settings, image, _path = env
    image.processed_at = datetime.now(timezone.utc)
    db.add(image)
    db.commit()

    crops = FrameProcessingService(
        db, settings, detector=_StubDetector([WHOLE_BODY])
    ).process_image(image)

    assert crops == []
    db.refresh(image)
    assert image.crop_reason == {"reason": "already_processed", "reprocessable": True}
