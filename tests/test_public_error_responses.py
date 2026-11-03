import importlib.util
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from test_reid import load_app


def test_search_hides_unexpected_exception_but_keeps_query_validation(monkeypatch, tmp_path):
    main = load_app(monkeypatch, tmp_path, "public-errors")
    from app.api import search

    private = "postgres://synthetic-secret@private-db/internal/path"

    def fail(*_args):
        raise ValueError(private)

    monkeypatch.setattr(search.VisualSearchService, "search", fail)
    with TestClient(main.create_app()) as client:
        response = client.post("/api/search/person-crops", json={"query": "dark"})
    assert response.status_code == 400
    assert private not in response.text
    assert "server logs" in response.json()["detail"]
    for message in ("同一标签包含矛盾条件，请拆分查询", "开始时间不能晚于结束时间"):
        assert search._public_label_query_error(ValueError(message)) == message


def test_backfill_does_not_publish_historical_provider_errors(monkeypatch, tmp_path):
    main = load_app(monkeypatch, tmp_path, "backfill-errors")
    from app.api import attributes

    private = "synthetic-password /private/model/config"
    progress = SimpleNamespace(attempted=1, updated=0, last_errors=[private])
    monkeypatch.setattr(
        attributes, "DurableAttributeBackfillService",
        lambda *a, **k: SimpleNamespace(
            load_progress=lambda: SimpleNamespace(attempted=0, updated=0),
            run=lambda **k: progress,
        ),
    )
    with TestClient(main.create_app()) as client:
        response = client.post("/api/attributes/person-crops/backfill")
    assert response.status_code == 200
    assert private not in response.text
    assert response.json()["seen"] == 1
    assert response.json()["errors"] == ["Attribute analysis failed; check server logs"]


def _load_reid_server_without_model_dependencies(monkeypatch):
    # Exercise the real HTTP handlers while preventing model imports/downloads and GPU use.
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setenv("REID_WARMUP_ON_START", "false")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        load=lambda *a, **k: None, cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(),
    ))
    monkeypatch.setitem(sys.modules, "omegaconf", SimpleNamespace(OmegaConf=SimpleNamespace()))
    monkeypatch.setitem(sys.modules, "src.aligners", SimpleNamespace(get_aligner=lambda: None))
    monkeypatch.setitem(sys.modules, "src.models", SimpleNamespace(get_model=lambda: None))
    path = Path(__file__).resolve().parents[1] / "deploy/agx/reid_service/server.py"
    spec = importlib.util.spec_from_file_location("_security_reid_server", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("failure", [RuntimeError, lambda s: HTTPException(503, detail=s)])
def test_reid_warmup_reports_failure_without_publishing_details(monkeypatch, caplog, failure):
    server = _load_reid_server_without_model_dependencies(monkeypatch)
    private = "synthetic-secret /private/checkpoint/path"

    def fail():
        raise failure(private)

    monkeypatch.setattr(server, "_load_runtime", fail)
    monkeypatch.setattr(server, "_warmup_started", True)
    with caplog.at_level(logging.ERROR):
        server._warm_runtime()
    with TestClient(server.app) as client:
        health = client.get("/health")
        ready = client.get("/ready")
    assert health.status_code == 200
    assert health.json()["loaded"] is False
    assert ready.status_code == 503
    assert private not in health.text + ready.text
    assert private in caplog.text


def test_reid_image_failure_does_not_disclose_local_path(monkeypatch):
    server = _load_reid_server_without_model_dependencies(monkeypatch)
    with TestClient(server.app) as client:
        response = client.post("/embed", json={"image_url": "/private/synthetic-missing.png"})
    assert response.status_code == 400
    assert response.json()["detail"] == "invalid image_url"
    assert "/private" not in response.text
