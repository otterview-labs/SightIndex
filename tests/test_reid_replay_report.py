import io
import json
import math
from email.message import Message
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

import pytest

from scripts import replay_reid_queries as replay
from scripts.replay_reid_queries import summarize_response


def test_replay_report_excludes_media_vectors_and_unknown_fields() -> None:
    report = summarize_response(
        {
            "items": [
                {
                    "crop_id": "example",
                    "score": 0.4,
                    "crop_url": "/private.jpg",
                    "embedding": [1, 0],
                    "new_sensitive_field": "secret",
                }
            ],
            "face_coverage": {"status": "query_unavailable"},
        },
        "similar",
    )
    assert report["items"] == [{"crop_id": "example", "score": 0.4}]
    assert report["result_count"] == 1


def test_replay_report_keeps_missing_face_diagnostics_unknown() -> None:
    report = summarize_response({"links": [{"score": 0.0}]}, "links")
    assert report["items"] == [{"score": 0.0}]
    assert report["face_coverage"] is None


def test_replay_report_keeps_candidate_pool_coverage_without_media() -> None:
    report = summarize_response(
        {
            "links": [{"crop_id": "candidate", "crop_url": "/private.jpg"}],
            "candidate_coverage": {
                "raw_hit_count": 5000,
                "hit_camera_count": 2,
                "indexed_camera_count": 6,
                "pool_limit": 5000,
                "sql_row_missing_count": 2,
                "possibly_truncated": True,
                "embedding": [1, 0],
            },
        },
        "links",
    )
    assert report["candidate_coverage"] == {
        "raw_hit_count": 5000,
        "hit_camera_count": 2,
        "indexed_camera_count": 6,
        "pool_limit": 5000,
        "sql_row_missing_count": 2,
        "possibly_truncated": True,
    }


def test_replay_report_does_not_invent_old_server_candidate_coverage() -> None:
    report = summarize_response({"links": []}, "links")
    assert "candidate_coverage" not in report


def test_coverage_has_a_recursive_numeric_allowlist() -> None:
    report = summarize_response(
        {
            "items": [],
            "face_coverage": {
                "status": "compared",
                "compared_count": 3,
                "embedding": [1, 0],
                "crop_url": "/private.jpg",
                "query_face_quality": {"image": "private"},
                "query_absence_reasons": {
                    "no_face": 2,
                    "inference_error": True,
                    "source_missing": -1,
                    "source_unreadable": [1, 0],
                    "new_sensitive_field": "secret",
                },
                "candidate_absence_reasons": {"low_quality": 1, "embedding": [1, 0]},
            },
        },
        "similar",
    )
    assert report["face_coverage"] == {
        "status": "compared",
        "compared_count": 3,
        "query_absence_reasons": {"no_face": 2},
        "candidate_absence_reasons": {"low_quality": 1},
    }


def test_scalar_fields_cannot_become_nested_media_or_nonfinite_json() -> None:
    report = summarize_response(
        {
            "items": [{"crop_id": {"embedding": [1, 0]}, "score": math.nan}],
            "query_mode": ["private"],
            "query_frame_count": {"data": "private"},
            "face_coverage": ["private"],
        },
        "similar",
    )
    assert report["items"] == [{}]
    assert report["query_mode"] is None
    assert report["query_frame_count"] is None
    assert report["face_coverage"] is None
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("rows", [None, "unexpected", [None]])
def test_malformed_result_shapes_fail_explicitly(rows: object) -> None:
    with pytest.raises(ValueError):
        summarize_response({"items": rows}, "similar")


class FakeOpener:
    """Return bounded in-memory responses; no test performs network requests."""

    def __init__(self, failures: dict[int, Exception] | None = None) -> None:
        self.failures = failures or {}
        self.urls: list[str] = []

    def open(self, request: Request, *, timeout: float) -> io.BytesIO:
        self.urls.append(request.full_url)
        failure = self.failures.get(len(self.urls))
        if failure is not None:
            raise failure
        if request.full_url.endswith("/status"):
            payload: dict[str, Any] = {"ready": True, "indexed_crops": 20}
        elif "/similar?" in request.full_url:
            payload = {"items": [{"crop_id": "candidate", "score": 0.5}]}
        else:
            payload = {"links": [{"crop_id": "candidate", "score": 0.4}]}
        return io.BytesIO(json.dumps(payload).encode())


def run_fake_replay(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    opener: FakeOpener,
) -> tuple[int, dict[str, Any]]:
    def build(*handlers: Any) -> FakeOpener:
        assert any(isinstance(handler, replay.RejectRedirects) for handler in handlers)
        assert any(
            isinstance(handler, ProxyHandler) and not handler.proxies for handler in handlers
        )
        return opener

    monkeypatch.setattr(replay, "build_opener", build)
    exit_code = replay.main(
        [
            "--crop-id",
            "00000000-0000-0000-0000-000000000001",
            "--label",
            "synthetic",
        ]
    )
    return exit_code, json.loads(capsys.readouterr().out)


@pytest.mark.parametrize("failed_call,status_key", [(1, "status_before"), (4, "status_after")])
@pytest.mark.parametrize(
    "error",
    [
        URLError("private error body"),
        TimeoutError("private timeout"),
        HTTPError("http://127.0.0.1/", 503, "private error body", None, None),
    ],
)
def test_health_read_failures_preserve_completed_replay_and_exit_one(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failed_call: int,
    status_key: str,
    error: Exception,
) -> None:
    opener = FakeOpener({failed_call: error})
    exit_code, report = run_fake_replay(monkeypatch, capsys, opener)
    assert exit_code == 1
    assert report[status_key] is None
    assert report[f"{status_key}_error"]
    assert len(report["queries"]) == 2
    assert all(row["result_count"] == 1 for row in report["queries"])
    assert "finished_at" in report
    assert "private" not in json.dumps(report)
    assert len(opener.urls) == 4


def test_query_and_both_health_failures_still_emit_a_complete_failure_report(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    opener = FakeOpener({i: URLError("private") for i in range(1, 5)})
    exit_code, report = run_fake_replay(monkeypatch, capsys, opener)
    assert exit_code == 1
    assert report["status_before"] is None
    assert report["status_after"] is None
    assert [row["error"] for row in report["queries"]] == ["URLError", "URLError"]


def test_successful_replay_is_loopback_only_and_exits_zero(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    opener = FakeOpener()
    exit_code, report = run_fake_replay(monkeypatch, capsys, opener)
    assert exit_code == 0
    assert report["accuracy_evaluated"] is False
    assert report["status_before"]["ready"] is True
    assert report["status_after"]["ready"] is True
    assert all(url.startswith("http://127.0.0.1:8000/") for url in opener.urls)


@pytest.mark.parametrize("code", [301, 302, 303, 307, 308])
def test_redirect_is_an_error_without_following_location(
    monkeypatch: pytest.MonkeyPatch,
    code: int,
) -> None:
    headers = Message()
    headers["Location"] = "https://external.invalid/private"
    opener = build_opener(ProxyHandler({}), replay.RejectRedirects())

    def forbidden_open(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Redirect attempted to open a second endpoint")

    monkeypatch.setattr(opener, "open", forbidden_open)
    with pytest.raises(HTTPError) as raised:
        opener.error(
            "http",
            Request("http://127.0.0.1:8000/api/reid/status"),
            io.BytesIO(),
            code,
            "redirect",
            headers,
        )
    assert raised.value.code == code


def test_redirect_in_query_is_reported_and_later_queries_continue(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    error = HTTPError("http://127.0.0.1/", 302, "sensitive redirect body", None, None)
    exit_code, report = run_fake_replay(monkeypatch, capsys, FakeOpener({2: error}))
    assert exit_code == 1
    assert report["queries"][0]["error"] == "HTTP 302"
    assert report["queries"][1]["result_count"] == 1
    assert "sensitive" not in json.dumps(report)
