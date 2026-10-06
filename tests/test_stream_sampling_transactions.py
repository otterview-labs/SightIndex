"""Run the real stream loop with synthetic capture and real transactional SQLite rows."""

from __future__ import annotations

import importlib
import types
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker


@dataclass(frozen=True)
class SyntheticFrame:
    index: int
    now: float
    shape: tuple[int, int, int] = (1000, 1000, 3)


def _run_scenario(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    centers: list[list[tuple[float, float]]],
    times: list[float],
    successful_indices: dict[int, set[int]] | None = None,
    counting_line: bool = False,
    interval: float = 2.0,
) -> types.SimpleNamespace:
    """Replace capture/inference only; retain the actual loop, tracker, crossing and SQL flow."""

    from app.config.settings import Settings
    from app.db.session import Base
    from app.models.events import CountingEvent
    from app.models.media import PersonCrop, VideoStream
    from app.services import stream_runtime
    from app.services.frame_processing import Detection

    for name in ("persons", "events", "media", "vectors", "chat", "reid"):
        importlib.import_module(f"app.models.{name}")
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        person_detector="whole_frame",
        stream_warmup_frames=0,
        stream_sharpest_frame_burst_size=1,
        stream_store_empty_frames=False,
        stream_diagnostics_enabled=False,
        person_crop_dedupe_enabled=True,
        person_crop_visit_max_samples=3,
        person_crop_visit_idle_seconds=6.0,
        counting_track_idle_seconds=6.0,
        line_crossing_point="center",
        line_crossing_match_distance=0.32,
        vector_index_on_ingest=False,
        reid_enabled=False,
        milvus_enabled=False,
        face_recognition_on_ingest=False,
    )
    engine = create_engine(f"sqlite:///{tmp_path / 'stream.sqlite'}")
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine)
    with sessions() as db:
        stream = VideoStream(
            name="synthetic capture",
            stream_url="synthetic://no-camera",
            camera_id=uuid.uuid4(),
            status="stopped",
            frame_interval_seconds=interval,
            reconnect_interval_seconds=1,
            counting_line={"x1": 0.0, "y1": 0.5, "x2": 1.0, "y2": 0.5} if counting_line else None,
        )
        db.add(stream)
        db.commit()
        stream_id = stream.id
    frames = [SyntheticFrame(index, now) for index, now in enumerate(times)]
    clock = [0.0]
    read_count = [0]
    calls: list[tuple[int, list[Detection]]] = []
    trackers: list[Any] = []
    crossing_calls: list[tuple[float, int, float, int, int]] = []

    class FastStop:
        stopped = False

        def is_set(self) -> bool:
            return self.stopped

        def wait(self, _timeout: float) -> bool:
            if read_count[0] >= len(frames):
                self.stopped = True
            return self.stopped

    stop = FastStop()

    class Capture:
        released = False

        def isOpened(self) -> bool:
            return True

        def read(self) -> tuple[bool, SyntheticFrame]:
            if read_count[0] >= len(frames):
                stop.stopped = True
                raise InterruptedError("synthetic capture exhausted")
            frame = frames[read_count[0]]
            read_count[0] += 1
            clock[0] = frame.now
            return True, frame

        def release(self) -> None:
            self.released = True

    capture = Capture()

    class Processor:
        def __init__(self, db: Any, _settings: Settings) -> None:
            self.db = db

        def detect_image_path(self, _path: Path) -> list[Detection]:
            return [
                Detection(
                    bbox={"x": x * 1000 - 40, "y": y * 1000 - 80, "width": 80, "height": 160},
                    confidence=0.95,
                    source_index=index,
                )
                for index, (x, y) in enumerate(centers[read_count[0] - 1])
            ]

        def quality_filter_detections(
            self, detections: list[Detection], _width: int, _height: int
        ) -> list[Detection]:
            return detections

        def process_image(self, image: Any, detections: list[Detection]) -> list[PersonCrop]:
            frame_index = read_count[0] - 1
            calls.append((frame_index, list(detections)))
            successful = (
                successful_indices.get(frame_index, set())
                if successful_indices is not None
                else {detection.source_index for detection in detections}
            )
            crops = []
            for detection in detections:
                if detection.source_index not in successful:
                    continue
                crop = PersonCrop(
                    image_id=image.id,
                    crop_url=f"/data/crops/synthetic-{frame_index}-{detection.source_index}.jpg",
                    bbox={**detection.bbox, "source_detection_index": detection.source_index},
                    camera_id=image.camera_id,
                    captured_at=image.captured_at,
                )
                self.db.add(crop)
                crops.append(crop)
            self.db.commit()
            return crops

    tracker_type = stream_runtime.AppearanceTracker

    def tracker_factory(**kwargs: Any) -> Any:
        tracker = tracker_type(**kwargs)
        trackers.append(tracker)
        return tracker

    runtime = stream_runtime.StreamRuntime()

    def write_frame(_stream: Any, frame: SyntheticFrame, _cv2: Any) -> tuple[str, Path, datetime]:
        path = tmp_path / f"synthetic-{frame.index}.jpg"
        path.write_bytes(b"synthetic frame placeholder; never decoded or inferred")
        return (
            f"/data/{path.name}",
            path,
            datetime(2026, 10, 1, tzinfo=UTC) + timedelta(seconds=frame.now),
        )

    original_crossings = stream_runtime.VideoProcessingService._line_crossings

    def record_crossings(self: Any, **kwargs: Any) -> Any:
        before = len(kwargs["tracks"])
        result = original_crossings(self, **kwargs)
        crossing_calls.append(
            (
                kwargs["now"],
                len(kwargs["detections"]),
                kwargs["max_idle_seconds"],
                before,
                len(kwargs["tracks"]),
            )
        )
        return result

    monkeypatch.setattr(stream_runtime, "SessionLocal", sessions)
    monkeypatch.setattr(stream_runtime, "get_settings", lambda: settings)
    monkeypatch.setattr(stream_runtime, "CaptureProcess", lambda *_args: capture)
    monkeypatch.setattr(stream_runtime, "FrameProcessingService", Processor)
    monkeypatch.setattr(stream_runtime, "AppearanceTracker", tracker_factory)
    monkeypatch.setattr(stream_runtime, "measure_person_roi", lambda *_args: None)
    # Replace only this module's clock, not Python's shared time module used by pytest/SQL.
    monkeypatch.setattr(stream_runtime, "time", types.SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(runtime, "_usable_frame_reference", lambda *_args: (True, None))
    monkeypatch.setattr(runtime, "_write_frame_file", write_frame)
    monkeypatch.setattr(runtime, "_try_index_frame_image", lambda *_args: None)
    monkeypatch.setattr(stream_runtime.VideoProcessingService, "_line_crossings", record_crossings)
    try:
        runtime._run_capture_loop(stream_id, stop)
        with sessions() as db:
            crop_rows = [dict(crop.bbox) for crop in db.scalars(select(PersonCrop))]
            crop_ids = list(db.scalars(select(PersonCrop.id)))
            event_rows = [
                (event.crop_id, event.direction) for event in db.scalars(select(CountingEvent))
            ]
            final_stream = db.get(VideoStream, stream_id)
            assert final_stream is not None
            assert final_stream.last_error is None
            assert final_stream.status == "stopped"
        assert stop.stopped and capture.released
        assert read_count[0] == len(frames)
        return types.SimpleNamespace(
            calls=calls,
            crops=crop_rows,
            crop_ids=crop_ids,
            events=event_rows,
            trackers=trackers,
            crossing_calls=crossing_calls,
        )
    finally:
        engine.dispose()


def test_all_first_crops_failed_are_retried_once_then_deduped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result = _run_scenario(
        monkeypatch,
        tmp_path,
        centers=[[(0.25, 0.5)]] * 3,
        times=[0.0, 2.0, 4.0],
        successful_indices={0: set(), 1: {0}, 2: {0}},
    )
    assert [index for index, _detections in result.calls] == [0, 1]
    assert len(result.crops) == 1
    assert result.crops[0]["source_detection_index"] == 0
    first_token = result.calls[0][1][0].bbox["sampling_visit_token"]
    saved_token = result.calls[1][1][0].bbox["sampling_visit_token"]
    assert isinstance(first_token, str) and isinstance(saved_token, str)
    assert first_token != saved_token, "the failed first visit must release its retention budget"
    assert result.crops[0]["sampling_visit_token"] == saved_token
    assert result.trackers[0].samples_saved == (1,)


def test_partial_success_retries_only_failed_visit_not_successful_sibling(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result = _run_scenario(
        monkeypatch,
        tmp_path,
        centers=[[(0.2, 0.5), (0.8, 0.5)]] * 3,
        times=[0.0, 2.0, 4.0],
        successful_indices={0: {1}, 1: {0}, 2: {0, 1}},
    )
    assert [(index, [item.source_index for item in items]) for index, items in result.calls] == [
        (0, [0, 1]),
        (1, [0]),
    ]
    assert sorted(crop["source_detection_index"] for crop in result.crops) == [0, 1]
    first_saved_token = result.calls[0][1][1].bbox["sampling_visit_token"]
    second_saved_token = result.calls[1][1][0].bbox["sampling_visit_token"]
    assert {crop["sampling_visit_token"] for crop in result.crops} == {
        first_saved_token,
        second_saved_token,
    }
    assert first_saved_token != second_saved_token
    assert result.trackers[0].samples_saved == (1, 1)


def test_ten_second_sampling_keeps_track_across_default_six_second_idle_window(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result = _run_scenario(
        monkeypatch,
        tmp_path,
        centers=[[(0.5, 0.4)], [(0.5, 0.6)], [(0.5, 0.65)]],
        times=[0.0, 10.0, 20.0],
        counting_line=True,
        interval=10.0,
    )
    assert [index for index, _items in result.calls] == [1]
    assert len(result.crops) == 1
    assert result.crops[0]["source_detection_index"] == 0
    assert result.events == [(result.crop_ids[0], "a_to_b")]
    assert [call[2] for call in result.crossing_calls] == [20.0, 20.0, 20.0]


def test_empty_detection_frame_runs_crossing_expiry_before_new_person_arrives(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result = _run_scenario(
        monkeypatch,
        tmp_path,
        centers=[[(0.5, 0.4)], [], [(0.5, 0.6)]],
        times=[0.0, 7.0, 8.0],
        counting_line=True,
    )
    assert result.crossing_calls == [
        (0.0, 1, 6.0, 0, 1),
        (7.0, 0, 6.0, 1, 0),
        (8.0, 1, 6.0, 0, 1),
    ]
    assert result.calls == []
    assert result.crops == []
    assert result.events == [], "an expired track must not create a crossing for a later person"
