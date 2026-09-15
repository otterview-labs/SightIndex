import io
import json
import runpy
from pathlib import Path
from urllib.error import URLError

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _services() -> dict:
    compose = ROOT / "deploy/containers/compose.yaml"
    return yaml.safe_load(compose.read_text())["services"]


def test_container_infrastructure_is_not_published() -> None:
    services = _services()
    for name in ("postgres", "etcd", "minio", "milvus", "reid"):
        assert "ports" not in services[name]
        assert "network_mode" not in services[name]
    assert services["api"]["ports"] == [
        "${API_BIND:-127.0.0.1}:${API_PORT:-18030}:8000"
    ]


def test_container_defaults_are_isolated_and_authenticated() -> None:
    settings = _services()["api"]["environment"]
    assert settings["STREAM_AUTOSTART_RUNNING"] == "false"
    assert settings["APP_BASIC_AUTH_PASSWORD"].startswith("${APP_BASIC_AUTH_PASSWORD:?")
    assert settings["MILVUS_HOST"] == "milvus"
    assert settings["MILVUS_NAMESPACE_ID"] == "sightindex-bj-test"
    assert settings["REID_ENABLED"] == "${REID_ENABLED:-false}"
    assert settings["VECTOR_INDEX_BACKGROUND_BATCH_SIZE"] == "1"
    assert settings["FACE_INSIGHTFACE_ALLOW_DOWNLOAD"] == "false"
    assert settings["YOLO_DEVICE"] == "cpu"
    assert settings["FACE_EMBEDDING_DEVICE"] == "cpu"


def test_container_gpu_is_opt_in_and_models_are_read_only() -> None:
    services = _services()
    assert "deploy" not in services["api"]
    assert services["reid"]["profiles"] == ["reid"]
    assert services["reid"]["environment"]["REID_BATCH_LIMIT"] == "1"
    for name in ("api", "reid"):
        assert services[name]["read_only"] is True
        assert services[name]["cap_drop"] == ["ALL"]
        model_mounts = [
            mount for mount in services[name]["volumes"] if mount.startswith("${MODEL_DIR")
        ]
        assert model_mounts
        assert all(mount.endswith(":ro") for mount in model_mounts)


def test_container_image_does_not_copy_host_data_or_credentials() -> None:
    dockerfile = (ROOT / "deploy/containers/Dockerfile").read_text()
    assert "COPY . " not in dockerfile
    assert "COPY data" not in dockerfile
    assert "USER sightindex" in dockerfile
    assert '"--workers", "1"' in dockerfile
    assert "npm run build" in dockerfile
    assert "image-requirements.txt" in dockerfile
    assert "COPY deploy/containers/healthcheck.py healthcheck.py" in dockerfile
    assert "build-essential python3-dev libglib2.0-0 libgl1" in dockerfile
    assert "apt-get purge -y --auto-remove build-essential python3-dev" in dockerfile
    assert _services()["api"]["healthcheck"]["test"] == [
        "CMD", "python", "/opt/sightindex/healthcheck.py"
    ]


def test_wheel_base_preserves_cuda_version_without_host_python_changes() -> None:
    dockerfile = (ROOT / "deploy/containers/Dockerfile.torch-base").read_text()
    assert "https://download.pytorch.org/whl/cu128" in dockerfile
    assert "python3 -m venv /opt/venv" in dockerfile
    assert "torch==2.10.0 torchvision==0.25.0" in dockerfile
    assert "assert torch.version.cuda == '12.8'" in dockerfile
    assert "COPY " not in dockerfile


@pytest.mark.parametrize("reid_enabled", [False, True])
def test_container_healthcheck_checks_authenticated_database_and_optional_reid(
    monkeypatch, reid_enabled
):
    module = runpy.run_path(str(ROOT / "deploy/containers/healthcheck.py"))
    requests = []

    def urlopen(request, timeout):
        requests.append(request)
        payload = (
            {"ready": True}
            if request.full_url.endswith("/api/reid/status")
            else {"image_with_crops_count": 0, "person_crop_count": 0}
        )
        return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr(module["request"], "urlopen", urlopen)
    module["check_api"]({
        "APP_BASIC_AUTH_USERNAME": "viewer",
        "APP_BASIC_AUTH_PASSWORD": "secret",
        "REID_ENABLED": str(reid_enabled),
    })
    assert len(requests) == (2 if reid_enabled else 1)
    assert requests[0].full_url.endswith("/api/media/counts")
    assert all(
        request.get_header("Authorization") == "Basic dmlld2VyOnNlY3JldA=="
        for request in requests
    )


@pytest.mark.parametrize("failure", ["database", "invalid_counts", "reid"])
def test_container_healthcheck_fails_when_dependencies_are_unusable(monkeypatch, failure):
    module = runpy.run_path(str(ROOT / "deploy/containers/healthcheck.py"))

    def urlopen(request, timeout):
        if failure == "database":
            raise URLError("database endpoint unavailable")
        if request.full_url.endswith("/api/reid/status"):
            payload = {"ready": False}
        elif failure == "invalid_counts":
            payload = {"status": "ok"}
        else:
            payload = {"image_with_crops_count": 0, "person_crop_count": 0}
        return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr(module["request"], "urlopen", urlopen)
    with pytest.raises((URLError, RuntimeError)):
        module["check_api"]({
            "APP_BASIC_AUTH_USERNAME": "viewer",
            "APP_BASIC_AUTH_PASSWORD": "secret",
            "REID_ENABLED": "true",
        })
