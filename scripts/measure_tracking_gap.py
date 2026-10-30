"""Measure how much the crop filter removes before the tracker ever sees it.

The line-crossing tracker used to run on the crop-storage list, so a person clipped by the frame
edge -- i.e. exactly someone walking in or out -- was deleted before it could be tracked. This
prints the size of that gap, from either source:

    # 1. What the deployment already logged. Diagnostics are OFF by default, so enable them,
    #    restart the API, let real traffic flow for a few minutes, then read the log:
    #      STREAM_DIAGNOSTICS_ENABLED=true
    .venv/bin/python scripts/measure_tracking_gap.py --log logs/api.log

    # 2. Or re-detect stored frames directly (costs GPU time, but needs no restart):
    .venv/bin/python scripts/measure_tracking_gap.py --frames data/frames --max 200

    # 3. Or sample a recorded video:
    .venv/bin/python scripts/measure_tracking_gap.py --video data/videos/door.mp4 --interval 0.5

Read "only_tracked" as the answer: those are detections a cross-camera-consistent tracker would
keep and the current counting pipeline throws away. If it is near zero on real footage, the
filter is not the problem and the sampling rate is; if it is a large share of raw, the split
pays off immediately.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config.settings import get_settings  # noqa: E402
from app.services.frame_processing import (  # noqa: E402
    FrameProcessingService,
    create_person_detector,
)

DIAGNOSTIC = re.compile(
    r"stream_diag .* raw=(?P<raw>\d+) quality=(?P<quality>\d+) "
    r"crossings=(?P<crossings>\d+) reason=(?P<reason>\S+)"
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _reason_for_crop_rejection(processor: FrameProcessingService, detection: Any, width, height):
    """Mirror the crop filter's own order so the report cannot drift from the code."""

    if detection.confidence < processor.settings.person_crop_min_confidence:
        return "below_crop_confidence"
    bbox = detection.bbox
    too_small = (
        int(bbox.get("width", 0) or 0) < processor.settings.person_crop_min_bbox_width
        or int(bbox.get("height", 0) or 0) < processor.settings.person_crop_min_bbox_height
    )
    if too_small:
        return "below_min_bbox"
    if processor.settings.person_crop_require_whole_body and width and height:
        if processor._touches_frame_edge(
            bbox,
            int(bbox.get("width", 0) or 0),
            int(bbox.get("height", 0) or 0),
            width,
            height,
        ):
            return "touches_frame_edge"
    return "other"


def _report_filter_gap(counts: Counter, reasons: Counter, frames: int) -> None:
    raw = counts["raw"]
    print(f"frames measured     {frames}")
    print(f"raw detections      {raw}")
    if not raw:
        print("\nNo detections at all: check the detector before judging the filters.")
        return
    kept = counts["crop_kept"]
    tracked = counts["track_kept"]
    only_tracked = counts["only_tracked"]
    print(f"crop filter keeps   {kept:5d}  ({kept / raw:.1%})")
    print(f"tracking keeps      {tracked:5d}  ({tracked / raw:.1%})")
    print(f"only_tracked        {only_tracked:5d}  ({only_tracked / raw:.1%})")
    print("                    ^ kept by a tracker, thrown away by the crop filter")
    print(f"neither filter      {counts['untracked']:5d}")
    if reasons:
        print("\nwhy the crop filter dropped a tracked detection:")
        for reason, count in reasons.most_common():
            print(f"  {reason:24s} {count}")


def _measure_detections(detections: list[Any], width: int, height: int, processor) -> None:
    for detection in detections:
        counts["raw"] += 1
        crop_ok = processor.is_quality_detection(detection, width, height)
        track_ok = processor.is_trackable_detection(detection, width, height)
        if crop_ok:
            counts["crop_kept"] += 1
        if track_ok:
            counts["track_kept"] += 1
        if track_ok and not crop_ok:
            counts["only_tracked"] += 1
            reasons[_reason_for_crop_rejection(processor, detection, width, height)] += 1
        if not track_ok:
            counts["untracked"] += 1


counts: Counter = Counter()
reasons: Counter = Counter()


def measure_log(path: Path) -> int:
    counters: Counter = Counter()
    reason_counts: Counter = Counter()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = DIAGNOSTIC.search(line)
        if match is None:
            continue
        counters["frames"] += 1
        counters["raw"] += int(match.group("raw"))
        counters["quality"] += int(match.group("quality"))
        counters["crossings"] += int(match.group("crossings"))
        reason_counts[match.group("reason")] += 1

    if not counters["frames"]:
        print(f"No stream_diag lines in {path}.")
        print("Diagnostics are off by default: set STREAM_DIAGNOSTICS_ENABLED=true and restart.")
        return 2

    print(f"log                 {path}")
    print(f"diagnostic frames   {counters['frames']}")
    print(f"raw detections      {counters['raw']}")
    print(f"quality detections  {counters['quality']}")
    print(f"crossings           {counters['crossings']}")
    print("\nframe reasons:")
    for reason, count in reason_counts.most_common():
        print(f"  {reason:26s} {count}")
    blank = reason_counts.get("quality_filtered", 0)
    if blank:
        print(
            f"\n{blank} frames had detections that the crop filter removed entirely. "
            "That is the population the tracker never saw."
        )
    return 0


def _iter_frame_files(directory: Path, limit: int):
    files = sorted(
        (path for path in directory.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return files[:limit]


def measure_frames(directory: Path, limit: int, detector, processor) -> int:
    cv2 = _cv2()
    measured = 0
    for path in _iter_frame_files(directory, limit):
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        height, width = frame.shape[:2]
        _measure_detections(detector.detect(path), width, height, processor)
        measured += 1
    _report_filter_gap(counts, reasons, measured)
    return 0


def measure_video(path: Path, interval: float, limit: int, detector, processor) -> int:
    cv2 = _cv2()
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        print(f"Could not open {path}")
        return 1
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    step = max(1, round(fps * interval)) if fps > 0 else 1
    frames_read = 0
    measured = 0
    scratch = Path(get_settings().frames_dir)
    scratch.mkdir(parents=True, exist_ok=True)
    scratch_frame = scratch / "_tracking_gap_probe.jpg"
    try:
        while measured < limit:
            ok, frame = capture.read()
            if not ok:
                break
            index = frames_read
            frames_read += 1
            if index % step != 0:
                continue
            height, width = frame.shape[:2]
            if not cv2.imwrite(str(scratch_frame), frame):
                continue
            _measure_detections(detector.detect(scratch_frame), width, height, processor)
            measured += 1
    finally:
        capture.release()
        scratch_frame.unlink(missing_ok=True)
    _report_filter_gap(counts, reasons, measured)
    return 0


def _cv2():
    try:
        import cv2  # type: ignore[import-not-found]
    except Exception as exc:
        raise SystemExit(f"OpenCV is not installed: {exc}") from exc
    return cv2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--log", type=Path, help="API log to read stream_diag lines from")
    source.add_argument("--frames", type=Path, help="directory of stored frames to re-detect")
    source.add_argument("--video", type=Path, help="video file to sample and re-detect")
    parser.add_argument("--max", type=int, default=200, help="frames to measure (default 200)")
    parser.add_argument("--interval", type=float, default=0.5, help="video sampling interval")
    args = parser.parse_args()

    if args.log is not None:
        return measure_log(args.log)

    settings = get_settings()
    print(f"detector            {settings.person_detector}")
    print(f"crop rules          min_conf={settings.person_crop_min_confidence} "
          f"whole_body={settings.person_crop_require_whole_body} "
          f"edge_margin={settings.person_crop_edge_margin}")
    print(f"tracking rules      min_conf={settings.tracking_min_confidence} "
          f"whole_body={settings.tracking_require_whole_body}")
    print()

    detector = create_person_detector(settings)
    processor = FrameProcessingService(db=None, settings=settings, detector=detector)

    if args.frames is not None:
        return measure_frames(args.frames, args.max, detector, processor)
    return measure_video(args.video, args.interval, args.max, detector, processor)


if __name__ == "__main__":
    raise SystemExit(main())
