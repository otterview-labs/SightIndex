#!/usr/bin/env python3
"""Replay a bounded crop list through the real API; emit metadata, never media/vectors.

These retrieval POSTs may populate the application's normal lazy caches. They do not submit
identity feedback, rebuild an index, or evaluate accuracy without human-labelled ground truth.
Run on the application host: the endpoint is intentionally restricted to loopback.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from http.client import HTTPException
from typing import Any
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener
from uuid import UUID

ITEM_FIELDS = (
    "crop_id",
    "camera_id",
    "score",
    "frame_count",
    "first_seen",
    "last_seen",
    "face_similarity",
    "face_match",
    "face_reliability",
    "face_query_identity_verified",
    "face_candidate_identity_verified",
    "face_candidate_source_crop_id",
    "attribute_agreement",
    "attribute_comparable_count",
    "attribute_match_count",
    "attribute_conflict_count",
    "attribute_conflict_weight",
    "fusion_score",
    "evidence_level",
)
STATUS_FIELDS = (
    "ready",
    "model",
    "index_fingerprint",
    "indexed_crops",
    "pending_crops",
    "min_score",
    "min_score_cross_camera",
    "face_priority_ready",
    "face_candidate_limit",
    "face_min_quality",
    "face_strong_reliability",
    "face_rescue_min_body_score",
)
FACE_COVERAGE_FIELDS = (
    "status",
    "query_face_found",
    "query_face_quality",
    "query_identity_verified",
    "query_attempted_count",
    "candidate_attempted_count",
    "shortlist_count",
    "compared_count",
    "borrowed_candidate_count",
    "hard_match_count",
    "hard_conflict_count",
)
FACE_ABSENCE_REASONS = frozenset(
    {
        "candidate_row_missing",
        "extraction_unexplained",
        "face_outside_head",
        "inference_error",
        "invalid_bbox",
        "invalid_embedding",
        "low_quality",
        "model_incompatible",
        "multiple_faces",
        "no_face",
        "query_identity_unverified",
        "source_missing",
        "source_unreadable",
        "temp_write_failed",
        "unsupported_url",
    }
)
CANDIDATE_COVERAGE_FIELDS = (
    "raw_hit_count",
    "hit_camera_count",
    "indexed_camera_count",
    "pool_limit",
    "sql_row_missing_count",
    "possibly_truncated",
)


class RejectRedirects(HTTPRedirectHandler):
    """Never let a loopback request redirect to another endpoint or host."""

    def redirect_request(
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        # Returning None makes urllib surface HTTPError instead of following Location.
        return None


def _scalar_fields(payload: dict[str, Any], fields: Sequence[str]) -> dict[str, Any]:
    """Exclude nested data even if it appears under a formerly scalar field name."""
    return {
        key: payload[key]
        for key in fields
        if key in payload
        and (payload[key] is None or isinstance(payload[key], str | int | float | bool))
        and (not isinstance(payload[key], float) or math.isfinite(payload[key]))
    }


def _face_coverage(payload: object) -> dict[str, Any] | None:
    """Allowlist coverage fields and bounded numeric abstention counters recursively."""
    if not isinstance(payload, dict):
        return None
    result = _scalar_fields(payload, FACE_COVERAGE_FIELDS)
    for name in ("query_absence_reasons", "candidate_absence_reasons"):
        reasons = payload.get(name)
        if isinstance(reasons, dict):
            result[name] = {
                reason: count
                for reason, count in reasons.items()
                if reason in FACE_ABSENCE_REASONS
                and isinstance(count, int)
                and not isinstance(count, bool)
                and count >= 0
            }
    return result


def _candidate_coverage(payload: object) -> dict[str, Any] | None:
    """Keep only the non-sensitive camera-pool diagnostics from a links response."""
    if not isinstance(payload, dict):
        return None
    return _scalar_fields(payload, CANDIDATE_COVERAGE_FIELDS)


def summarize_response(payload: dict[str, Any], endpoint: str) -> dict[str, Any]:
    """Allowlist numeric/identity metadata so new API media fields cannot leak into reports."""
    rows = payload.get("items" if endpoint == "similar" else "links", [])
    if not isinstance(rows, list) or any(not isinstance(item, dict) for item in rows):
        raise ValueError("Unexpected retrieval result shape")
    query = _scalar_fields(payload, ("query_mode", "query_frame_count"))
    result = {
        "result_count": len(rows),
        "query_mode": query.get("query_mode"),
        "query_frame_count": query.get("query_frame_count"),
        "face_coverage": _face_coverage(payload.get("face_coverage")),
        "items": [_scalar_fields(item, ITEM_FIELDS) for item in rows],
    }
    # Older API versions do not return this field.  Omit it rather than manufacturing an
    # all-zero diagnostic, so replay comparisons remain honest across revisions.
    if "candidate_coverage" in payload:
        result["candidate_coverage"] = _candidate_coverage(payload.get("candidate_coverage"))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop-id", type=UUID, action="append", required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--label", required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args(argv)
    crop_ids = list(dict.fromkeys(args.crop_id))
    if not 1 <= len(crop_ids) <= 10 or not 1 <= args.top_k <= 50:
        parser.error("use 1–10 unique crops and top-k 1–50")
    if not 1 <= args.port <= 65535 or not 0 < args.timeout <= 180:
        parser.error("invalid port or timeout (maximum 180 seconds)")
    opener = build_opener(ProxyHandler({}), RejectRedirects())
    base = f"http://127.0.0.1:{args.port}"

    def read(path: str, *, post: bool = False) -> dict[str, Any]:
        request = Request(base + path, data=b"" if post else None)
        with opener.open(request, timeout=args.timeout) as response:
            payload = json.load(response)
        if not isinstance(payload, dict):
            raise ValueError("Unexpected API response shape")
        return payload

    report: dict[str, Any] = {
        "label": args.label,
        "started_at": datetime.now(UTC).isoformat(),
        "accuracy_evaluated": False,
        "warning": "Unlabelled live replay; caches may warm and the live dataset may grow.",
        "queries": [],
    }

    def capture_status(name: str) -> bool:
        """Preserve completed retrieval results if either bookend health read fails."""
        try:
            status = read("/api/reid/status")
            safe = _scalar_fields(status, STATUS_FIELDS)
            report[name] = {key: safe.get(key) for key in STATUS_FIELDS}
            return True
        except HTTPError as exc:
            error = f"HTTP {exc.code}"
        except (OSError, HTTPException, ValueError) as exc:
            error = type(exc).__name__
        report[name] = None
        report[f"{name}_error"] = error
        return False

    failed = not capture_status("status_before")
    for crop_id in crop_ids:
        for endpoint in ("similar", "links"):
            start = time.monotonic()
            row: dict[str, Any] = {"query_crop_id": str(crop_id), "endpoint": endpoint}
            suffix = f"?top_k={args.top_k}" if endpoint == "similar" else ""
            try:
                payload = read(f"/api/reid/crops/{crop_id}/{endpoint}{suffix}", post=True)
                row.update(summarize_response(payload, endpoint))
            except HTTPError as exc:
                row["error"] = f"HTTP {exc.code}"  # Do not log potentially sensitive bodies.
                failed = True
            except (OSError, HTTPException, ValueError) as exc:
                row["error"] = type(exc).__name__
                failed = True
            row["elapsed_seconds"] = round(time.monotonic() - start, 3)
            report["queries"].append(row)
    if not capture_status("status_after"):
        failed = True
    report["finished_at"] = datetime.now(UTC).isoformat()
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
