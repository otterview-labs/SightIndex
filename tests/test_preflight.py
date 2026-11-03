from pathlib import Path

import pytest

from app import preflight
from app.config.settings import Settings


def test_hog_setup_explains_missing_labels_without_claiming_readiness(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight, "_installed", lambda module: module == "cv2")
    monkeypatch.setattr(preflight, "_hog_supported", lambda: True)
    report = preflight.inspect_setup(Settings(
        _env_file=None, person_detector="hog", appearance_pose_weights=tmp_path / "missing.pt",
    ))
    assert report["ok"] is True
    assert report["warnings"] == 1
    labels = report["checks"][1]
    assert labels["status"] == "warning"
    assert "APPEARANCE_POSE_WEIGHTS" in labels["message"]


@pytest.mark.parametrize("runtime", [False, True])
def test_pose_file_is_not_an_inference_readiness_claim(tmp_path, monkeypatch, runtime):
    checkpoint = tmp_path / "pose.pt"
    checkpoint.write_bytes(b"not actual weights")
    monkeypatch.setattr(preflight, "_installed", lambda module: module == "cv2" or runtime)
    monkeypatch.setattr(preflight, "_hog_supported", lambda: True)
    report = preflight.inspect_setup(Settings(
        _env_file=None, person_detector="hog", appearance_pose_weights=checkpoint,
    ))
    labels = report["checks"][1]
    assert labels["status"] == ("ok" if runtime else "warning")
    if runtime:
        assert "not tested" in labels["message"]


def test_yolo_preflight_requires_a_local_file(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight, "_installed", lambda module: True)
    report = preflight.inspect_setup(Settings(
        _env_file=None, person_detector="yolo", yolo_model=str(tmp_path / "missing.pt"),
        appearance_tone_on_ingest=False,
    ))
    assert report["ok"] is False
    assert report["checks"][0]["status"] == "error"
    assert report["checks"][1]["status"] == "disabled"


def test_custom_pose_path_reaches_ingest_and_backfill(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from test_reid import load_app

    main = load_app(monkeypatch, tmp_path, "pose-setting", APPEARANCE_POSE_WEIGHTS="~/custom.pt")
    from app.services import appearance_attributes
    from app.services.frame_processing import FrameProcessingService

    received = []

    def record(**kwargs):
        received.append(kwargs["pose_weights"])
        return object()

    monkeypatch.setattr(appearance_attributes, "AppearanceAttributeService", record)
    # The backfill router imported the class directly; patch that same consumer too.
    from app.api import attributes

    monkeypatch.setattr(attributes, "AppearanceAttributeService", record)
    with TestClient(main.create_app()) as client:
        assert client.post("/api/attributes/person-crops/tone-backfill").status_code == 200
        FrameProcessingService(None, main.get_settings())._try_read_clothing_tone([])
    assert received == [str(Path("~/custom.pt").expanduser())] * 2


def test_preflight_rejects_installed_opencv_without_hog(monkeypatch):
    monkeypatch.setattr(preflight, "_installed", lambda module: True)
    monkeypatch.setattr(preflight, "_hog_supported", lambda: False)
    report = preflight.inspect_setup(Settings(
        _env_file=None, person_detector="hog", appearance_tone_on_ingest=False,
    ))
    assert report["ok"] is False
    assert "OpenCV 4" in report["checks"][0]["message"]


def test_cpu_hog_can_process_the_public_sample(tmp_path):
    from app.services.frame_processing import OpenCVHogPersonDetector

    sample = Path(__file__).resolve().parents[1] / "samples" / "nasa-walk.jpg"
    detector = OpenCVHogPersonDetector()
    detections = detector.detect(sample)
    assert detections
    assert all(item.bbox["width"] > 0 and item.bbox["height"] > 0 for item in detections)
