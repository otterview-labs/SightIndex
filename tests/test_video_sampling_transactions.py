"""Exercise the uploaded-video loop using synthetic capture and real SQLite events."""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker


@dataclass(frozen=True)
class SyntheticFrame:
    index: int
    shape: tuple[int, int, int] = (100, 100, 3)


def _run_video(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    centers: list[list[float]],
    media_times: list[float],
    interval: float = 1.0,
    fps: float = 0.0,
    successful_sources: set[int] | None = None,
) -> SimpleNamespace:
    from app.config.settings import Settings
    from app.db.session import Base
    from app.models.events import CountingEvent
    from app.models.media import PersonCrop
    from app.services import video_processing
    from app.services.frame_processing import Detection
    from app.services.time_utils import database_datetime
    from app.services.video_processing import CountingLine, VideoFrameFile, VideoProcessingService

    for name in ("persons", "events", "media", "vectors", "chat", "reid"):
        importlib.import_module(f"app.models.{name}")
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        person_detector="whole_frame",
        line_crossing_track_idle_seconds=6.0,
        line_crossing_track_max_missed_frames=2,
        line_crossing_match_distance=0.5,
        line_crossing_point="center",
        vector_index_on_ingest=False,
        reid_enabled=False,
        milvus_enabled=False,
        face_recognition_on_ingest=False,
    )
    engine = create_engine(f"sqlite:///{tmp_path / 'video.sqlite'}")
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    base_time = datetime(2026, 10, 6, tzinfo=UTC)
    calls: list[tuple[int, list[int | None]]] = []
    captured_times: list[datetime] = []

    class Capture:
        index = -1
        released = False

        def isOpened(self) -> bool:
            return True

        def read(self) -> tuple[bool, SyntheticFrame | None]:
            self.index += 1
            if self.index >= len(centers):
                return False, None
            return True, SyntheticFrame(self.index)

        def get(self, prop: int) -> float:
            return fps if prop == 1 else media_times[self.index] * 1000.0

        def release(self) -> None:
            self.released = True

    capture = Capture()
    monkeypatch.setitem(
        sys.modules,
        "cv2",
        SimpleNamespace(VideoCapture=lambda _path: capture, CAP_PROP_FPS=1, CAP_PROP_POS_MSEC=2),
    )
    # Wall time is constant while media time advances, so the lifecycle must use media time.
    monkeypatch.setattr(video_processing.time, "monotonic", lambda: 0.0)

    class Processor:
        def __init__(self, db: Any) -> None:
            self.db = db

        def detect_image_path(self, _path: Path) -> list[Detection]:
            return [
                Detection(
                    bbox={"x": center * 100 - 5, "y": 45, "width": 10, "height": 10},
                    confidence=1.0,
                )
                for center in centers[capture.index]
            ]

        def quality_filter_detections(
            self, detections: list[Detection], _width: int, _height: int
        ) -> list[Detection]:
            return detections

        def process_image(self, image: Any, detections: list[Detection]) -> list[PersonCrop]:
            calls.append((capture.index, [detection.source_index for detection in detections]))
            crops = []
            for detection in detections:
                if (
                    successful_sources is not None
                    and detection.source_index not in successful_sources
                ):
                    continue
                crop = PersonCrop(
                    image_id=image.id,
                    crop_url=f"/data/crops/{capture.index}-{detection.source_index}.jpg",
                    bbox={**detection.bbox, "source_detection_index": detection.source_index},
                    captured_at=image.captured_at,
                )
                self.db.add(crop)
                crops.append(crop)
            self.db.flush()
            return crops

    def write_frame(
        *, frame: SyntheticFrame, captured_at: datetime, **_kwargs: Any
    ) -> VideoFrameFile:
        path = tmp_path / f"frame-{frame.index}.jpg"
        path.touch()
        captured_times.append(captured_at)
        return VideoFrameFile(
            url=f"/data/frames/frame-{frame.index}.jpg", path=path, captured_at=captured_at
        )

    try:
        with sessions() as db:
            service = VideoProcessingService(db, settings)
            service.processor = Processor(db)
            monkeypatch.setattr(service, "_write_frame_file", write_frame)
            result = service.process_video_path(
                video_path=tmp_path / "synthetic.avi",
                video_url="/data/videos/synthetic.avi",
                frame_interval_seconds=interval,
                max_frames=len(centers),
                counting_line=CountingLine(0.5, 0.0, 0.5, 1.0),
                captured_at=base_time,
            )
            events = list(db.scalars(select(CountingEvent).order_by(CountingEvent.counted_at)))
            crops = list(db.scalars(select(PersonCrop)))
        return SimpleNamespace(
            result=result,
            events=events,
            crops=crops,
            calls=calls,
            captured_times=captured_times,
            base_time=database_datetime(base_time, settings, "sqlite"),
            released=capture.released,
        )
    finally:
        engine.dispose()


def test_slow_video_sampling_keeps_the_track_and_original_media_timestamps(monkeypatch, tmp_path):
    scenario = _run_video(
        monkeypatch,
        tmp_path,
        centers=[[0.4]] + [[]] * 9 + [[0.6]],
        media_times=[float(index) for index in range(11)],
        interval=10.0,
        fps=1.0,
    )

    assert scenario.result.frames_read == 11
    assert scenario.result.frames_sampled == 2
    assert scenario.result.counting_events_created == 1
    assert scenario.result.images_created == 1
    assert scenario.result.crops_created == 1
    assert scenario.captured_times == [
        scenario.base_time,
        scenario.base_time + timedelta(seconds=10),
    ]
    assert scenario.events[0].counted_at == scenario.base_time + timedelta(seconds=10)
    assert scenario.released


def test_empty_video_frame_expires_the_visit_using_media_time(monkeypatch, tmp_path):
    scenario = _run_video(
        monkeypatch,
        tmp_path,
        centers=[[0.4], [0.6], [], [0.4], [0.6]],
        media_times=[0.0, 1.0, 10.0, 10.1, 11.0],
    )

    assert scenario.result.frames_sampled == 5
    assert scenario.result.counting_events_created == 2
    assert scenario.result.images_created == 2
    assert scenario.result.crops_created == 2
    assert [event.counted_at for event in scenario.events] == [
        scenario.base_time + timedelta(seconds=1),
        scenario.base_time + timedelta(seconds=11),
    ]
    assert scenario.released


def test_two_empty_video_frames_expire_the_visit_before_the_idle_deadline(monkeypatch, tmp_path):
    scenario = _run_video(
        monkeypatch,
        tmp_path,
        centers=[[0.4], [0.6], [], [], [0.4], [0.6]],
        media_times=[float(index) for index in range(6)],
    )

    assert scenario.result.counting_events_created == 2
    assert scenario.result.images_created == 2
    assert scenario.result.crops_created == 2


def test_video_partial_crop_failure_keeps_the_success_on_its_original_event(monkeypatch, tmp_path):
    scenario = _run_video(
        monkeypatch,
        tmp_path,
        centers=[[0.4, 0.42], [0.6, 0.62]],
        media_times=[0.0, 1.0],
        successful_sources={1},
    )

    assert scenario.result.counting_events_created == 2
    assert scenario.result.images_created == 1
    assert scenario.result.crops_created == 1
    assert scenario.calls == [(1, [0, 1])]
    assert scenario.events[0].crop_id is None
    assert scenario.events[1].crop_id == scenario.crops[0].id
    assert scenario.crops[0].bbox["source_detection_index"] == 1
    assert scenario.released
