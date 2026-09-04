import json
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import create_engine

from app.config.settings import Settings
from app.db.session import Base
from app.models import persons  # noqa: F401
from app.models.media import Image
from app.services import frame_processing
from app.services.frame_processing import (
    Detection,
    FrameProcessingService,
    WholeFramePersonDetector,
    YoloServicePersonDetector,
)
from app.services.time_utils import local_now


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_yolo_service_detector_parses_service_detections(monkeypatch):
    captured: dict[str, Any] = {}

    def fake_urlopen(req: Any, timeout: int) -> _FakeResponse:
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse(
            {
                "image_size": {"width": 100, "height": 200},
                "detections": [
                    {"label": "person", "score": 0.8, "bbox": [0.1, 0.2, 0.4, 0.6]},
                    {"label": "car", "score": 0.9, "bbox": [0.0, 0.0, 1.0, 1.0]},
                    {"class_id": 0, "confidence": "0.6", "bbox": [10, 20, 30, 70]},
                    {"name": "person", "score": "bad", "bbox": {"x": 5, "y": 6, "w": 10, "h": 12}},
                ],
            }
        )

    monkeypatch.setattr(frame_processing.request, "urlopen", fake_urlopen)
    detector = YoloServicePersonDetector(
        base_url="http://detector.local/",
        confidence=0.25,
        iou=0.5,
        image_size=1280,
        max_det=20,
        timeout_seconds=12,
    )

    detections = detector.detect(Path("/tmp/frame.jpg"))

    assert captured["url"] == "http://detector.local/predict"
    assert captured["timeout"] == 12
    assert captured["payload"] == {
        "task": "det",
        "image_path": "/tmp/frame.jpg",
        "conf": 0.25,
        "iou": 0.5,
        "imgsz": 1280,
        "max_det": 20,
    }
    assert [d.confidence for d in detections] == [0.8, 0.6, 0.0]
    assert [d.bbox for d in detections] == [
        {"x": 10, "y": 40, "width": 30, "height": 80},
        {"x": 10, "y": 20, "width": 20, "height": 50},
        {"x": 5, "y": 6, "width": 10, "height": 12},
    ]


def test_person_crop_bbox_padding_expands_crop_without_changing_detection_box(tmp_path):
    np = pytest.importorskip("numpy")
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    settings = Settings(
        data_dir=tmp_path,
        person_detector="whole_frame",
        person_crop_padding_x_ratio=0.5,
        person_crop_padding_top_ratio=0.25,
        person_crop_padding_bottom_ratio=0.5,
    )
    service = FrameProcessingService(
        db=None,  # type: ignore[arg-type]
        settings=settings,
        detector=WholeFramePersonDetector(),
    )
    detection = Detection(
        bbox={"x": 40, "y": 40, "width": 20, "height": 20},
        confidence=0.9,
    )

    assert service._bounded_bbox(image, detection) == (40, 40, 60, 60)
    assert service._bounded_bbox(image, detection, padded=True) == (30, 35, 70, 70)


def test_person_crop_quality_filter_and_upscale(tmp_path):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

    from sqlalchemy.orm import sessionmaker

    engine = create_engine(f"sqlite:///{tmp_path / 'quality.db'}")
    Base.metadata.create_all(bind=engine)
    db = sessionmaker(bind=engine)()
    try:
        image_array = np.zeros((320, 320, 3), dtype=np.uint8)
        source_path = tmp_path / "frame.jpg"
        assert cv2.imwrite(str(source_path), image_array)
        settings = Settings(
            data_dir=tmp_path,
            person_detector="whole_frame",
            person_crop_min_bbox_width=60,
            person_crop_min_bbox_height=120,
            person_crop_min_confidence=0.4,
            person_crop_upscale_min_width=180,
            person_crop_upscale_min_height=360,
            person_crop_upscale_max_factor=2.0,
            person_crop_sharpen_amount=0.0,
        )
        service = FrameProcessingService(db=db, settings=settings)
        image = Image(image_url="/data/frame.jpg", source_type="upload")
        db.add(image)
        db.commit()
        db.refresh(image)

        crops = service.process_image(
            image,
            detections=[
                Detection(
                    bbox={"x": 10, "y": 10, "width": 40, "height": 160},
                    confidence=0.9,
                ),
                Detection(
                    bbox={"x": 80, "y": 20, "width": 80, "height": 150},
                    confidence=0.9,
                ),
            ],
        )

        assert len(crops) == 1
        assert crops[0].bbox["width"] == 80
        assert crops[0].bbox["height"] == 150
        assert crops[0].bbox["crop_width"] >= 180
        assert crops[0].bbox["crop_height"] >= 360
        assert (tmp_path / crops[0].crop_url.removeprefix("/data/")).exists()
    finally:
        db.close()


def test_person_crop_enhancement_uses_bounded_display_size_and_preserves_flat_areas(tmp_path):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    service = FrameProcessingService(
        db=None,  # type: ignore[arg-type]
        settings=Settings(
            data_dir=tmp_path,
            person_crop_upscale_min_width=320,
            person_crop_upscale_min_height=720,
            person_crop_upscale_max_factor=2.0,
            person_crop_sharpen_amount=0.32,
            person_crop_sharpen_threshold=4,
        ),
        detector=WholeFramePersonDetector(),
    )
    crop = np.full((240, 100, 3), 64, dtype=np.uint8)

    enhanced = service._enhance_crop(cv2, crop)

    # The requested 320x720 canvas would need 3.2x, so the safety ceiling wins.
    assert enhanced.shape == (480, 200, 3)
    # Edge-aware sharpening must not amplify sensor noise across a flat wall or dark shirt.
    assert np.all(enhanced == 64)


def test_person_crop_zero_sharpen_threshold_preserves_legacy_unsharp_mask(tmp_path):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    service = FrameProcessingService(
        db=None,  # type: ignore[arg-type]
        settings=Settings(
            data_dir=tmp_path,
            person_crop_upscale_min_width=0,
            person_crop_upscale_min_height=0,
            person_crop_sharpen_amount=0.32,
            person_crop_sharpen_threshold=0,
        ),
        detector=WholeFramePersonDetector(),
    )
    crop = np.zeros((30, 30, 3), dtype=np.uint8)
    crop[:, 15:] = 180
    blurred = cv2.GaussianBlur(crop, (0, 0), 1.0)
    expected = cv2.addWeighted(crop, 1.32, blurred, -0.32, 0)

    enhanced = service._enhance_crop(cv2, crop)

    assert np.array_equal(enhanced, expected)


def test_local_now_uses_configured_timezone():
    settings = Settings(local_timezone="Asia/Shanghai")

    now = local_now(settings)

    assert now.tzinfo is not None
    assert now.utcoffset().total_seconds() == 8 * 60 * 60


def test_whole_frame_detector_survives_a_confidence_floor(tmp_path):
    """It asserts the frame is a person rather than scoring it, so 0.0 said the opposite.

    Harmless while the floor was 0; the moment PERSON_CROP_MIN_CONFIDENCE existed, every crop
    this detector produced was discarded and eight tests went red at once.
    """

    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

    path = tmp_path / "frame.jpg"
    assert cv2.imwrite(str(path), np.zeros((120, 80, 3), dtype=np.uint8))
    settings = Settings(data_dir=tmp_path)

    detections = WholeFramePersonDetector().detect(path)

    assert len(detections) == 1
    assert detections[0].confidence >= settings.person_crop_min_confidence


def test_the_confidence_floor_drops_low_scoring_detections(tmp_path):
    """Below it the detector is mostly finding floor and white objects, which match each other
    at 0.710 across cameras -- higher than any pair of real people."""

    from app.services.frame_processing import FrameProcessingService

    settings = Settings(data_dir=tmp_path, person_crop_min_confidence=0.70)
    service = FrameProcessingService(db=None, settings=settings)
    box = {"x": 0, "y": 0, "width": 200, "height": 400}

    kept = service.quality_filter_detections([
        Detection(bbox=box, confidence=0.42),
        Detection(bbox=box, confidence=0.93),
    ])

    assert [d.confidence for d in kept] == [0.93]
