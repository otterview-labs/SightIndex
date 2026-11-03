"""Read-only local setup checks: python -m app.preflight [--json] [--strict]."""

import argparse
import importlib.util
import json
from pathlib import Path

from app.config.settings import Settings


def _installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _hog_supported() -> bool:
    try:
        import cv2
    except ImportError:
        return False
    return callable(getattr(cv2, "HOGDescriptor", None)) and callable(
        getattr(cv2, "HOGDescriptor_getDefaultPeopleDetector", None)
    )


def inspect_setup(settings: Settings) -> dict[str, object]:
    """Inspect dependencies and local assets without loading models or opening services.

    File presence and installed packages are prerequisites, not inference readiness.
    No database access, downloads, network probes or background workers run here.
    """
    checks: list[dict[str, str]] = []

    def add(name: str, status: str, message: str) -> None:
        checks.append({"name": name, "status": status, "message": message})

    detector = settings.person_detector.lower()
    if detector == "hog":
        if _installed("cv2") and _hog_supported():
            add("detector", "ok", "OpenCV HOG API available; no model checkpoint required.")
        else:
            add(
                "detector", "error",
                "HOG needs a compatible OpenCV 4 build. Install requirements.cpu.txt.",
            )
    elif detector == "yolo":
        if not _installed("ultralytics"):
            add("detector", "error", "YOLO needs the optional Ultralytics runtime.")
        elif not settings.yolo_model or not Path(settings.yolo_model).expanduser().is_file():
            add("detector", "error", "Set YOLO_MODEL to an existing local checkpoint before use.")
        else:
            add("detector", "ok", "YOLO runtime and local file found; inference not tested.")
    elif detector == "yolo_service":
        add("detector", "warning", "YOLO HTTP service configured; connectivity not tested.")
    elif detector == "whole_frame":
        add("detector", "warning", "whole_frame is a test stub, not person detection.")
    else:
        add("detector", "error", "Unsupported PERSON_DETECTOR setting.")

    if not settings.appearance_tone_on_ingest:
        add("clothing_labels", "disabled", "Automatic CV clothing labels are disabled.")
    elif not settings.appearance_pose_weights.expanduser().is_file():
        add(
            "clothing_labels", "warning",
            "Local pose weights missing: detection can run, but CV clothing labels cannot. "
            "Set APPEARANCE_POSE_WEIGHTS to an existing YOLO-pose checkpoint.",
        )
    elif not _installed("ultralytics"):
        add("clothing_labels", "warning", "Pose file found; optional Ultralytics runtime missing.")
    else:
        add(
            "clothing_labels", "ok",
            "Pose runtime and local file found; inference and existing label coverage not tested.",
        )

    add(
        "external_services", "info",
        "Milvus, embeddings, VLM and ReID are not probed. Check their status and index coverage "
        "separately before using those workflows.",
    )
    return {
        "ok": not any(item["status"] == "error" for item in checks),
        "warnings": sum(item["status"] == "warning" for item in checks),
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print machine-readable results")
    parser.add_argument("--strict", action="store_true", help="Also fail on optional warnings")
    args = parser.parse_args()
    report = inspect_setup(Settings())
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for item in report["checks"]:
            print(f"[{item['status'].upper()}] {item['name']}: {item['message']}")
        print("Local prerequisites only; models were not loaded and services were not contacted.")
    return 0 if report["ok"] and (not args.strict or not report["warnings"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
