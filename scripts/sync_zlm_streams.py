#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from dataclasses import dataclass
from typing import Any
from urllib import error, request
from urllib.parse import urlparse

DEFAULT_SIGHTINDEX_URL = "http://127.0.0.1:8000"
DEFAULT_ZLM_MEDIA_LIST_URL = os.environ.get("ZLM_MEDIA_LIST_URL") or None
STREAM_NAME_PREFIX = "ZLM-H264-"


@dataclass(frozen=True)
class CandidateStream:
    name: str
    stream_url: str
    channel_id: str
    width: int
    height: int
    fps: float


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sync ZLMediaKit H.264 streams into SightIndex /api/streams."
    )
    parser.add_argument("--sightindex-url", default=DEFAULT_SIGHTINDEX_URL)
    parser.add_argument(
        "--zlm-media-list-url",
        default=DEFAULT_ZLM_MEDIA_LIST_URL,
        required=DEFAULT_ZLM_MEDIA_LIST_URL is None,
        help="ZLMediaKit getMediaList URL, including its operator-supplied secret",
    )
    parser.add_argument("--source-host", default="127.0.0.1:7003")
    parser.add_argument("--target-host", default="127.0.0.1:17003")
    parser.add_argument("--prefix", default=STREAM_NAME_PREFIX)
    parser.add_argument("--max-streams", type=int, default=16)
    parser.add_argument("--max-running", type=int, default=4)
    parser.add_argument("--frame-interval-seconds", type=float, default=2.0)
    parser.add_argument("--reconnect-interval-seconds", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=float, default=15.0)
    parser.add_argument("--basic-auth-user", default=os.environ.get("APP_BASIC_AUTH_USERNAME"))
    parser.add_argument("--basic-auth-password", default=os.environ.get("APP_BASIC_AUTH_PASSWORD"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    auth_header = basic_auth_header(args.basic_auth_user, args.basic_auth_password)

    candidates = discover_candidates(
        args.zlm_media_list_url,
        source_host=args.source_host,
        target_host=args.target_host,
        prefix=args.prefix,
        limit=args.max_streams,
        timeout=args.timeout_seconds,
    )
    existing = list_streams(
        args.sightindex_url,
        timeout=args.timeout_seconds,
        authorization=auth_header,
    )
    summary: dict[str, Any] = {
        "discovered": len(candidates),
        "created": [],
        "existing": [],
        "started": [],
        "skipped_start": [],
    }

    running_count = count_running_synced_streams(existing, args.prefix)
    existing_by_name = {stream.get("name"): stream for stream in existing}
    existing_by_url = {stream.get("stream_url"): stream for stream in existing}

    for candidate in candidates:
        stream = existing_by_name.get(candidate.name) or existing_by_url.get(candidate.stream_url)
        if stream is None:
            if args.dry_run:
                stream = {
                    "id": None,
                    "name": candidate.name,
                    "status": "dry-run",
                    "stream_url": candidate.stream_url,
                }
            else:
                stream = create_stream(
                    args.sightindex_url,
                    candidate,
                    frame_interval_seconds=args.frame_interval_seconds,
                    reconnect_interval_seconds=args.reconnect_interval_seconds,
                    timeout=args.timeout_seconds,
                    authorization=auth_header,
                )
            summary["created"].append(
                {
                    "id": stream.get("id"),
                    "name": candidate.name,
                    "resolution": f"{candidate.width}x{candidate.height}",
                    "fps": candidate.fps,
                }
            )
        else:
            summary["existing"].append(
                {
                    "id": stream.get("id"),
                    "name": stream.get("name"),
                    "status": stream.get("status"),
                }
            )

        stream_id = stream.get("id")
        if not stream_id:
            continue
        if stream.get("status") == "running":
            running_count += 1
            continue
        if running_count >= args.max_running:
            summary["skipped_start"].append(
                {"id": stream_id, "name": stream.get("name"), "reason": "max_running"}
            )
            continue
        if args.dry_run:
            summary["started"].append({"id": stream_id, "name": stream.get("name")})
            running_count += 1
            continue
        start_stream(
            args.sightindex_url,
            str(stream_id),
            timeout=args.timeout_seconds,
            authorization=auth_header,
        )
        summary["started"].append({"id": stream_id, "name": stream.get("name")})
        running_count += 1

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def discover_candidates(
    media_list_url: str,
    *,
    source_host: str,
    target_host: str,
    prefix: str,
    limit: int,
    timeout: float,
) -> list[CandidateStream]:
    payload = http_json("GET", media_list_url, timeout=timeout)
    items = payload.get("data", [])
    if not isinstance(items, list):
        return []

    selected: list[CandidateStream] = []
    seen_channels: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        candidate = candidate_from_media_item(
            item,
            source_host=source_host,
            target_host=target_host,
            prefix=prefix,
        )
        if candidate is None or candidate.channel_id in seen_channels:
            continue
        seen_channels.add(candidate.channel_id)
        selected.append(candidate)
        if len(selected) >= limit:
            break
    return selected


def candidate_from_media_item(
    item: dict[str, Any],
    *,
    source_host: str,
    target_host: str,
    prefix: str,
) -> CandidateStream | None:
    origin_url = item.get("originUrl")
    if not isinstance(origin_url, str):
        return None
    if source_host not in origin_url or "token=" not in origin_url:
        return None
    if "/device/" not in origin_url:
        return None
    video_track = video_track_from_item(item)
    if video_track is None or video_track.get("codec_id_name") != "H264":
        return None
    parsed = urlparse(origin_url)
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 3 or parts[0] != "device":
        return None
    channel_id = parts[2]
    return CandidateStream(
        name=f"{prefix}{channel_id}",
        stream_url=origin_url.replace(source_host, target_host, 1),
        channel_id=channel_id,
        width=int(video_track.get("width") or 0),
        height=int(video_track.get("height") or 0),
        fps=float(video_track.get("fps") or 0.0),
    )


def video_track_from_item(item: dict[str, Any]) -> dict[str, Any] | None:
    tracks = item.get("tracks")
    if not isinstance(tracks, list):
        return None
    for track in tracks:
        if isinstance(track, dict) and track.get("codec_type") == 0:
            return track
    return None


def list_streams(
    sightindex_url: str,
    *,
    timeout: float,
    authorization: str | None,
) -> list[dict[str, Any]]:
    payload = http_json(
        "GET",
        f"{sightindex_url.rstrip('/')}/api/streams?limit=200",
        timeout=timeout,
        authorization=authorization,
    )
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    items = payload.get("items", [])
    return [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []


def count_running_synced_streams(streams: list[dict[str, Any]], prefix: str) -> int:
    return sum(
        1
        for stream in streams
        if str(stream.get("name") or "").startswith(prefix) and stream.get("status") == "running"
    )


def create_stream(
    sightindex_url: str,
    candidate: CandidateStream,
    *,
    frame_interval_seconds: float,
    reconnect_interval_seconds: int,
    timeout: float,
    authorization: str | None,
) -> dict[str, Any]:
    return http_json(
        "POST",
        f"{sightindex_url.rstrip('/')}/api/streams",
        payload={
            "name": candidate.name,
            "stream_url": candidate.stream_url,
            "protocol": "rtsp",
            "frame_interval_seconds": frame_interval_seconds,
            "reconnect_interval_seconds": reconnect_interval_seconds,
        },
        timeout=timeout,
        authorization=authorization,
    )


def start_stream(
    sightindex_url: str,
    stream_id: str,
    *,
    timeout: float,
    authorization: str | None,
) -> dict[str, Any]:
    return http_json(
        "POST",
        f"{sightindex_url.rstrip('/')}/api/streams/{stream_id}/start",
        timeout=timeout,
        authorization=authorization,
    )


def http_json(
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float,
    authorization: str | None = None,
) -> Any:
    data = None
    headers = {"Accept": "application/json"}
    if authorization:
        headers["Authorization"] = authorization
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = request.Request(url, data=data, headers=headers, method=method)
    try:
        with request.urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed with HTTP {exc.code}: {detail}") from exc
    except OSError as exc:
        raise RuntimeError(f"{method} {url} failed: {exc}") from exc
    return json.loads(body) if body else {}


def basic_auth_header(username: str | None, password: str | None) -> str | None:
    if not username or not password:
        return None
    token = base64.b64encode(f"{username}:{password}".encode()).decode("ascii")
    return f"Basic {token}"


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"sync_zlm_streams failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
