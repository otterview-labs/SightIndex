import logging
import shutil
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

from sqlalchemy.orm import Session

from app.config.settings import Settings, get_settings
from app.db.session import SessionLocal
from app.models.media import Image, VideoStream
from app.services.appearance_tracker import AppearanceTracker
from app.services.frame_processing import Detection, FrameProcessingService
from app.services.opencv_capture import open_video_capture
from app.services.time_utils import local_now
from app.services.vector_index_queue import VectorQueueFullError
from app.services.video_processing import CountingLine, PersonTrack, VideoProcessingService

logger = logging.getLogger("sightindex.stream_diagnostics")


class StreamRuntime:
    def __init__(self) -> None:
        self._stop_events: dict[uuid.UUID, threading.Event] = {}
        self._threads: dict[uuid.UUID, threading.Thread] = {}
        self._lock = threading.Lock()

    def is_running(self, stream_id: uuid.UUID) -> bool:
        thread = self._threads.get(stream_id)
        return thread is not None and thread.is_alive()

    def start(self, stream_id: uuid.UUID) -> str:
        with self._lock:
            if self.is_running(stream_id):
                return "already running"
            stop_event = threading.Event()
            thread = threading.Thread(
                target=self._run_capture_loop,
                args=(stream_id, stop_event),
                daemon=True,
                name=f"sightindex-stream-{stream_id}",
            )
            self._stop_events[stream_id] = stop_event
            self._threads[stream_id] = thread
            thread.start()
            return "started"

    def stop(self, stream_id: uuid.UUID) -> str:
        with self._lock:
            stop_event = self._stop_events.get(stream_id)
            if stop_event is None:
                return "not running"
            stop_event.set()
            return "stopping"

    def wait_stopped(self, stream_id: uuid.UUID, timeout_seconds: float = 10.0) -> bool:
        with self._lock:
            thread = self._threads.get(stream_id)
        if thread is None:
            return True
        thread.join(timeout=max(timeout_seconds, 0.0))
        return not thread.is_alive()

    def _run_capture_loop(self, stream_id: uuid.UUID, stop_event: threading.Event) -> None:
        try:
            import cv2  # type: ignore[import-not-found]
        except Exception as exc:
            self._mark_error(stream_id, f"OpenCV is not installed: {exc}")
            return

        while not stop_event.is_set():
            db = SessionLocal()
            stream = db.get(VideoStream, stream_id)
            if stream is None:
                db.close()
                return
            reconnect_interval = stream.reconnect_interval_seconds
            settings = get_settings()
            stream.status = "running"
            stream.last_error = None
            stream.started_at = stream.started_at or local_now(settings)
            db.commit()
            db.refresh(stream)
            counting_line_value = self._counting_line_value(stream.counting_line)
            counting_line = self._counting_line(counting_line_value)
            tracks: dict[int, PersonTrack] = {}
            next_track_id = 1
            visits = AppearanceTracker(
                match_distance=settings.person_crop_visit_match_distance,
                idle_seconds=settings.person_crop_visit_idle_seconds,
                max_seconds=settings.person_crop_visit_max_seconds,
            )
            previous_frame_reference = None
            warmup_frames_remaining = int(settings.stream_warmup_frames)
            last_diagnostic_at = 0.0
            queue_backpressure_failures = 0

            capture = open_video_capture(cv2, stream.stream_url, settings)
            if not capture.isOpened():
                self._set_stream_error(
                    db,
                    stream,
                    f"Could not open stream: {stream.stream_url}",
                    status="error",
                )
                db.close()
                time.sleep(reconnect_interval)
                continue

            try:
                while not stop_event.is_set():
                    ok, frame = capture.read()
                    if not ok:
                        self._set_stream_error(db, stream, "Failed to read frame", status="error")
                        break
                    settings = get_settings()
                    if warmup_frames_remaining > 0:
                        warmup_frames_remaining -= 1
                        time.sleep(stream.frame_interval_seconds)
                        continue
                    is_usable, frame_reference = self._usable_frame_reference(
                        frame,
                        previous_frame_reference,
                        cv2,
                        settings,
                    )
                    if not is_usable:
                        last_diagnostic_at = self._log_diagnostic(
                            settings=settings,
                            stream=stream,
                            captured_at=local_now(settings),
                            last_diagnostic_at=last_diagnostic_at,
                            raw_detections=0,
                            quality_detections=0,
                            crossings=0,
                            reason="corrupt_frame_skipped",
                            counting_line_set=counting_line is not None,
                        )
                        time.sleep(stream.frame_interval_seconds)
                        continue
                    previous_frame_reference = (
                        frame_reference if frame_reference is not None else previous_frame_reference
                    )
                    db.refresh(stream)
                    next_line_value = self._counting_line_value(stream.counting_line)
                    if next_line_value != counting_line_value:
                        counting_line_value = next_line_value
                        counting_line = self._counting_line(counting_line_value)
                        tracks = {}
                        next_track_id = 1
                    frame_url, frame_path, captured_at = self._write_frame_file(stream, frame, cv2)
                    processor = FrameProcessingService(db, settings)
                    raw_detections = processor.detect_image_path(frame_path)
                    detections = processor.quality_filter_detections(raw_detections)
                    if counting_line is not None:
                        crossings = []
                        tracks_before_crossing = {
                            track_id: PersonTrack(
                                id=track.id,
                                center=track.center,
                                side=track.side,
                                counted=track.counted,
                            )
                            for track_id, track in tracks.items()
                        }
                        next_track_id_before_crossing = next_track_id
                        if detections:
                            count_service = VideoProcessingService(db, settings)
                            crossings, next_track_id = count_service._line_crossings(
                                detections=detections,
                                frame=frame,
                                line=counting_line,
                                tracks=tracks,
                                next_track_id=next_track_id,
                            )
                        if not crossings:
                            reason = self._skip_reason(raw_detections, detections)
                            last_diagnostic_at = self._log_diagnostic(
                                settings=settings,
                                stream=stream,
                                captured_at=captured_at,
                                last_diagnostic_at=last_diagnostic_at,
                                raw_detections=len(raw_detections),
                                quality_detections=len(detections),
                                crossings=0,
                                reason=reason,
                                counting_line_set=True,
                                frame_path=frame_path,
                            )
                            frame_path.unlink(missing_ok=True)
                            time.sleep(stream.frame_interval_seconds)
                            continue

                        image = self._create_frame_image(db, stream, frame_url, captured_at)
                        crossing_detections = [
                            detections[crossing.detection_index] for crossing in crossings
                        ]
                        try:
                            crops = processor.process_image(image, detections=crossing_detections)
                        except VectorQueueFullError as exc:
                            tracks = tracks_before_crossing
                            next_track_id = next_track_id_before_crossing
                            queue_backpressure_failures += 1
                            last_diagnostic_at, backoff_seconds = (
                                self._handle_vector_queue_full(
                                    db=db,
                                    stream=stream,
                                    settings=settings,
                                    captured_at=captured_at,
                                    last_diagnostic_at=last_diagnostic_at,
                                    frame_path=frame_path,
                                    raw_detections=len(raw_detections),
                                    quality_detections=len(detections),
                                    crossings=len(crossings),
                                    failures=queue_backpressure_failures,
                                    error=exc,
                                )
                            )
                            stop_event.wait(backoff_seconds)
                            continue
                        self._try_index_frame_image(db, image, settings)
                        crops_by_detection_index = {
                            crossing.detection_index: crop
                            for crossing, crop in zip(crossings, crops, strict=False)
                        }
                        count_service._create_line_crossing_events(
                            crossings=crossings,
                            crops_by_detection_index=crops_by_detection_index,
                            counted_at=captured_at,
                            camera_id=stream.camera_id,
                            location_id=stream.location_id,
                            stream_id=stream.id,
                            image=image,
                        )
                        stream.last_frame_image_id = image.id
                        stream.status = "running"
                        stream.last_error = None
                        db.commit()
                        queue_backpressure_failures = 0
                        last_diagnostic_at = self._log_diagnostic(
                            settings=settings,
                            stream=stream,
                            captured_at=captured_at,
                            last_diagnostic_at=last_diagnostic_at,
                            raw_detections=len(raw_detections),
                            quality_detections=len(detections),
                            crossings=len(crossings),
                            reason="crossing_saved",
                            counting_line_set=True,
                            frame_path=frame_path,
                            force=True,
                        )
                    elif detections or settings.stream_store_empty_frames:
                        # With a counting line the crossing already picks the one frame worth
                        # storing; without one, every frame of a stationary person qualifies.
                        visits_before_frame = visits.snapshot()
                        if detections and settings.person_crop_dedupe_enabled:
                            detections = self._first_sightings(detections, frame, visits)
                            if not detections and not settings.stream_store_empty_frames:
                                last_diagnostic_at = self._log_diagnostic(
                                    settings=settings,
                                    stream=stream,
                                    captured_at=captured_at,
                                    last_diagnostic_at=last_diagnostic_at,
                                    raw_detections=len(raw_detections),
                                    quality_detections=0,
                                    crossings=0,
                                    reason="already_stored_this_visit",
                                    counting_line_set=False,
                                    frame_path=frame_path,
                                )
                                frame_path.unlink(missing_ok=True)
                                time.sleep(stream.frame_interval_seconds)
                                continue
                        image = self._create_frame_image(db, stream, frame_url, captured_at)
                        try:
                            crops = processor.process_image(image, detections=detections)
                        except VectorQueueFullError as exc:
                            # This frame is being dropped, so its visits were never stored.
                            visits.restore(visits_before_frame)
                            queue_backpressure_failures += 1
                            last_diagnostic_at, backoff_seconds = (
                                self._handle_vector_queue_full(
                                    db=db,
                                    stream=stream,
                                    settings=settings,
                                    captured_at=captured_at,
                                    last_diagnostic_at=last_diagnostic_at,
                                    frame_path=frame_path,
                                    raw_detections=len(raw_detections),
                                    quality_detections=len(detections),
                                    crossings=0,
                                    failures=queue_backpressure_failures,
                                    error=exc,
                                )
                            )
                            stop_event.wait(backoff_seconds)
                            continue
                        self._try_index_frame_image(db, image, settings)
                        stream.last_frame_image_id = image.id
                        stream.status = "running"
                        stream.last_error = None
                        db.commit()
                        queue_backpressure_failures = 0
                        last_diagnostic_at = self._log_diagnostic(
                            settings=settings,
                            stream=stream,
                            captured_at=captured_at,
                            last_diagnostic_at=last_diagnostic_at,
                            raw_detections=len(raw_detections),
                            quality_detections=len(detections),
                            crossings=0,
                            reason="saved_without_counting_line",
                            counting_line_set=False,
                            frame_path=frame_path,
                            force=bool(detections),
                        )
                    else:
                        last_diagnostic_at = self._log_diagnostic(
                            settings=settings,
                            stream=stream,
                            captured_at=captured_at,
                            last_diagnostic_at=last_diagnostic_at,
                            raw_detections=len(raw_detections),
                            quality_detections=len(detections),
                            crossings=0,
                            reason=self._skip_reason(raw_detections, detections),
                            counting_line_set=False,
                            frame_path=frame_path,
                        )
                        frame_path.unlink(missing_ok=True)
                    time.sleep(stream.frame_interval_seconds)
            finally:
                capture.release()
                db.close()

        self._mark_stopped(stream_id)

    @staticmethod
    def _first_sightings(
        detections: list[Detection],
        frame: object,
        visits: AppearanceTracker,
    ) -> list[Detection]:
        """Drops the detections that repeat a body already stored during this visit."""

        height, width = frame.shape[:2]
        centers = []
        for detection in detections:
            bbox = detection.bbox
            x = float(bbox.get("x", 0))
            y = float(bbox.get("y", 0))
            box_width = float(bbox.get("width", 0))
            box_height = float(bbox.get("height", 0))
            centers.append(
                ((x + box_width / 2) / max(width, 1), (y + box_height / 2) / max(height, 1))
            )
        started = visits.new_visits(centers, time.monotonic())
        return [detections[index] for index in started]

    def _write_frame_file(
        self, stream: VideoStream, frame: object, cv2: object
    ) -> tuple[str, Path, datetime]:
        settings = get_settings()
        settings.frames_dir.mkdir(parents=True, exist_ok=True)
        now = local_now(settings)
        filename = f"{stream.id}_{now.strftime('%Y%m%d%H%M%S%f')}.jpg"
        path = settings.frames_dir / filename
        cv2.imwrite(
            str(path),
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(settings.frame_jpeg_quality)],
        )
        return f"/data/frames/{filename}", path, now

    def _create_frame_image(
        self,
        db: Session,
        stream: VideoStream,
        image_url: str,
        captured_at: datetime,
    ) -> Image:
        image = Image(
            image_url=image_url,
            source_type="stream_frame",
            camera_id=stream.camera_id,
            location_id=stream.location_id,
            captured_at=captured_at,
        )
        db.add(image)
        db.flush()
        return image

    @staticmethod
    def _try_index_frame_image(db: Session, image: Image, settings: Settings) -> None:
        """Run the legacy synchronous image index only after the frame transaction commits."""

        from app.services.media import MediaService

        MediaService(db, settings)._try_index_image(image)

    def _skip_reason(self, raw_detections: list[object], quality_detections: list[object]) -> str:
        if not raw_detections:
            return "no_raw_detections"
        if not quality_detections:
            return "quality_filtered"
        return "no_line_crossing"

    def _handle_vector_queue_full(
        self,
        *,
        db: Session,
        stream: VideoStream,
        settings: Settings,
        captured_at: datetime,
        last_diagnostic_at: float,
        frame_path: Path,
        raw_detections: int,
        quality_detections: int,
        crossings: int,
        failures: int,
        error: VectorQueueFullError,
    ) -> tuple[float, float]:
        """Roll back the rejected crop transaction and keep capture alive with backoff."""

        db.rollback()
        backoff_seconds = self._queue_backoff_seconds(
            frame_interval_seconds=stream.frame_interval_seconds,
            failures=failures,
        )
        message = (
            "Vector index queue full; crop/index transaction rolled back; "
            f"retrying stream after {backoff_seconds:.1f}s: {error}"
        )
        self._set_stream_error(db, stream, message, status="running")
        logger.warning(
            "stream_backpressure stream_id=%s failures=%s backoff_seconds=%.1f error=%s",
            stream.id,
            failures,
            backoff_seconds,
            error,
        )
        diagnostic_at = self._log_diagnostic(
            settings=settings,
            stream=stream,
            captured_at=captured_at,
            last_diagnostic_at=last_diagnostic_at,
            raw_detections=raw_detections,
            quality_detections=quality_detections,
            crossings=crossings,
            reason="vector_queue_full",
            counting_line_set=stream.counting_line is not None,
            frame_path=frame_path,
            force=True,
        )
        # Diagnostics copies the frame first when configured; the rejected source frame has no
        # database Image after rollback and must not accumulate as an orphan.
        frame_path.unlink(missing_ok=True)
        return diagnostic_at, backoff_seconds

    @staticmethod
    def _queue_backoff_seconds(frame_interval_seconds: float, failures: int) -> float:
        base = max(frame_interval_seconds, 0.2)
        return min(base * (2 ** min(max(failures - 1, 0), 6)), 30.0)

    def _log_diagnostic(
        self,
        *,
        settings: Settings,
        stream: VideoStream,
        captured_at: datetime,
        last_diagnostic_at: float,
        raw_detections: int,
        quality_detections: int,
        crossings: int,
        reason: str,
        counting_line_set: bool,
        frame_path: Path | None = None,
        force: bool = False,
    ) -> float:
        if not settings.stream_diagnostics_enabled:
            return last_diagnostic_at
        now = time.monotonic()
        if (
            not force
            and now - last_diagnostic_at < settings.stream_diagnostics_interval_seconds
        ):
            return last_diagnostic_at
        logger.warning(
            "stream_diag stream_id=%s name=%r captured_at=%s raw=%s quality=%s "
            "crossings=%s reason=%s line=%s frame=%s latest=%s",
            stream.id,
            stream.name,
            captured_at.isoformat(),
            raw_detections,
            quality_detections,
            crossings,
            reason,
            "set" if counting_line_set else "none",
            frame_path.name if frame_path else "-",
            self._keep_latest_diagnostic_frame(settings, stream, frame_path),
        )
        return now

    def _keep_latest_diagnostic_frame(
        self,
        settings: Settings,
        stream: VideoStream,
        frame_path: Path | None,
    ) -> str:
        if not settings.stream_diagnostics_keep_latest_frame or frame_path is None:
            return "-"
        try:
            settings.diagnostics_dir.mkdir(parents=True, exist_ok=True)
            target = settings.diagnostics_dir / f"latest_{stream.id}.jpg"
            shutil.copyfile(frame_path, target)
            return f"/data/diagnostics/{target.name}"
        except Exception as exc:
            return f"copy_error:{exc}"

    def _set_stream_error(
        self, db: Session, stream: VideoStream, message: str, status: str = "error"
    ) -> None:
        stream.status = status
        stream.last_error = message
        db.add(stream)
        db.commit()

    def _mark_error(self, stream_id: uuid.UUID, message: str) -> None:
        db = SessionLocal()
        stream = db.get(VideoStream, stream_id)
        if stream is not None:
            self._set_stream_error(db, stream, message)
        db.close()

    def _mark_stopped(self, stream_id: uuid.UUID) -> None:
        db = SessionLocal()
        stream = db.get(VideoStream, stream_id)
        if stream is not None:
            stream.status = "stopped"
            stream.stopped_at = local_now(get_settings())
            db.add(stream)
            db.commit()
        db.close()
        with self._lock:
            self._stop_events.pop(stream_id, None)
            self._threads.pop(stream_id, None)

    def _counting_line_value(
        self,
        value: dict[str, object] | None,
    ) -> tuple[float, float, float, float] | None:
        if not value:
            return None
        try:
            return (
                float(value["x1"]),
                float(value["y1"]),
                float(value["x2"]),
                float(value["y2"]),
            )
        except (KeyError, TypeError, ValueError):
            return None

    def _counting_line(
        self,
        value: tuple[float, float, float, float] | None,
    ) -> CountingLine | None:
        if value is None:
            return None
        try:
            return VideoProcessingService.create_counting_line(*value)
        except ValueError:
            return None

    def _usable_frame_reference(
        self,
        frame: object,
        previous_reference: object | None,
        cv2: object,
        settings: Settings,
    ) -> tuple[bool, object | None]:
        threshold = settings.stream_corrupt_frame_mean_diff_threshold
        if threshold <= 0:
            return True, previous_reference
        try:
            reference = self._frame_reference(frame, cv2)
        except Exception:
            return True, previous_reference
        if previous_reference is None:
            return True, reference
        try:
            difference = cv2.absdiff(reference, previous_reference)
            mean_difference = float(cv2.mean(difference)[0])
        except Exception:
            return True, reference
        if mean_difference > threshold:
            return False, previous_reference
        return True, reference

    def _frame_reference(self, frame: object, cv2: object) -> object:
        resized = cv2.resize(frame, (160, 90))
        return cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)


stream_runtime = StreamRuntime()
