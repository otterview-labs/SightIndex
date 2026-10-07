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
    assert services["api"]["ports"] == ["${API_BIND:-127.0.0.1}:${API_PORT:-18030}:8000"]


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
            mount
            for mount in services[name]["volumes"]
            if isinstance(mount, str) and mount.startswith("${MODEL_DIR")
        ]
        assert model_mounts
        assert all(mount.endswith(":ro") for mount in model_mounts)
    dfa_mount = next(mount for mount in services["reid"]["volumes"] if isinstance(mount, dict))
    assert dfa_mount["source"] == (
        "${MODEL_DIR:?set MODEL_DIR}/dfa_mobilenetv4_medium/mobilenetv4_Final.pth"
    )
    assert dfa_mount["target"].endswith(
        "/pretrained_models/aligners/dfa_mobilenetv4_medium/mobilenetv4_Final.pth"
    )
    assert dfa_mount["read_only"] is True
    assert dfa_mount["bind"]["create_host_path"] is False


def test_container_image_does_not_copy_host_data_or_credentials() -> None:
    dockerfile = (ROOT / "deploy/containers/Dockerfile").read_text()
    assert "COPY . " not in dockerfile
    assert "COPY data" not in dockerfile
    assert "USER sightindex" in dockerfile
    assert '"--workers", "1"' in dockerfile
    assert "npm run build" in dockerfile
    assert "image-requirements.txt" in dockerfile
    assert "COPY deploy/containers/healthcheck.py healthcheck.py" in dockerfile
    assert "COPY deploy/containers/verify.py deployment_verify.py" in dockerfile
    for script in ("reid", "search", "observations", "video-playback"):
        assert f"scripts/test-{script}.mjs" in dockerfile
    assert "RUN node --test" in dockerfile
    assert "/pretrained_models/aligners/dfa_mobilenetv4_medium" in dockerfile
    requirements = (ROOT / "deploy/containers/requirements.txt").read_text().splitlines()
    assert "httpx>=0.27,<1" in requirements
    assert "build-essential python3-dev libglib2.0-0 libgl1" in dockerfile
    assert "apt-get purge -y --auto-remove build-essential python3-dev" in dockerfile
    assert _services()["api"]["healthcheck"]["test"] == [
        "CMD",
        "python",
        "/opt/sightindex/healthcheck.py",
    ]


def test_container_opt_in_settings_are_explicit_and_keep_vector_identity() -> None:
    settings = _services()["api"]["environment"]
    passthrough = {
        "FACE_RECOGNITION_ON_INGEST": "false",
        "PERSON_CROP_VISIT_MAX_SAMPLES": "3",
        "PERSON_CROP_VISIT_SAMPLE_INTERVAL_SECONDS": "2.0",
        "PERSON_CROP_VISIT_QUALITY_IMPROVEMENT_RATIO": "0.15",
        "VLM_PROVIDER": "none",
        "VLM_BASE_URL": "http://127.0.0.1:8001/v1",
        "VLM_MODEL": "Qwen3.6-27B",
        "VLM_API_KEY": "",
        "VLM_SERVICE_API_KEY": "",
        "VLM_TIMEOUT_SECONDS": "120",
        "VLM_MAX_TOKENS": "256",
        "VLM_TEMPERATURE": "0.1",
        "VLM_CAPTION_ON_INDEX": "false",
        "VLM_STRUCTURED_ON_INGEST": "false",
        "VLM_STRUCTURED_BACKGROUND": "false",
        "VLM_STRUCTURED_MAX_TOKENS": "1200",
        "VLM_STRUCTURED_MIN_CONFIDENCE": "0.55",
    }
    for key, default in passthrough.items():
        assert settings[key] == "${" + key + ":-" + default + "}"
    for key in ("REID_CROSS_CAMERA_CALIBRATION_COEF", "REID_CROSS_CAMERA_CALIBRATION_INTERCEPT"):
        assert settings[key] is None
    assert settings["MILVUS_NAMESPACE_ID"] == "sightindex-bj-test"
    assert settings["MILVUS_COLLECTION_PREFIX"] == "sightindex_bj_test"
    assert settings["REID_CHECKPOINT_REVISION"] == (
        "${REID_CHECKPOINT_REVISION:?set REID_CHECKPOINT_REVISION}"
    )
    embedding = yaml.safe_load((ROOT / "deploy/containers/compose.embedding.yaml").read_text())[
        "services"
    ]["api"]["environment"]
    assert embedding["VISUAL_EMBEDDING_DIM"] == "2048"
    assert embedding["VISUAL_EMBEDDING_MODEL"] == "Qwen/Qwen3-VL-Embedding-2B"
    assert embedding["MILVUS_VISUAL_COLLECTION_PREFIX"] == (
        "${QWEN_VISUAL_COLLECTION_PREFIX:-sightindex_bj_test_qwen3vl2b_768p_v1}"
    )
    semantic = yaml.safe_load(
        (ROOT / "deploy/containers/compose.semantic-search.yaml").read_text()
    )["services"]["api"]["environment"]
    assert semantic == {
        "SEMANTIC_SEARCH_ENABLED": "${SEMANTIC_SEARCH_ENABLED:-false}",
        "SEMANTIC_SEARCH_MIN_SCORE": "${SEMANTIC_SEARCH_MIN_SCORE:-0.25}",
        "SEMANTIC_SEARCH_MAX_SCOPE": "${SEMANTIC_SEARCH_MAX_SCOPE:-10000}",
    }


def test_container_env_template_has_no_secrets_or_default_calibration() -> None:
    template = (ROOT / "deploy/containers/.env.example").read_text()
    values = dict(
        line.split("=", 1) for line in template.splitlines() if line and not line.startswith("#")
    )
    for key in (
        "VLM_API_KEY",
        "VLM_SERVICE_API_KEY",
        "APP_BASIC_AUTH_PASSWORD",
        "POSTGRES_PASSWORD",
        "MINIO_ROOT_PASSWORD",
        "REID_SERVICE_API_KEY",
        "QWEN_EMBEDDING_API_KEY",
    ):
        assert values[key] == ""
    for key in (
        "REID_ENABLED",
        "FACE_RECOGNITION_ON_INGEST",
        "VLM_STRUCTURED_ON_INGEST",
        "VLM_STRUCTURED_BACKGROUND",
        "SEMANTIC_SEARCH_ENABLED",
    ):
        assert values[key] == "false"
    assert values["VLM_PROVIDER"] == "none"
    for key in ("REID_CROSS_CAMERA_CALIBRATION_COEF", "REID_CROSS_CAMERA_CALIBRATION_INTERCEPT"):
        assert key not in values
        assert f"# {key}=<finite " in template


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
    module["check_api"](
        {
            "APP_BASIC_AUTH_USERNAME": "viewer",
            "APP_BASIC_AUTH_PASSWORD": "secret",
            "REID_ENABLED": str(reid_enabled),
        }
    )
    assert len(requests) == (2 if reid_enabled else 1)
    assert requests[0].full_url.endswith("/api/media/counts")
    assert all(
        request.get_header("Authorization") == "Basic dmlld2VyOnNlY3JldA==" for request in requests
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
        module["check_api"](
            {
                "APP_BASIC_AUTH_USERNAME": "viewer",
                "APP_BASIC_AUTH_PASSWORD": "secret",
                "REID_ENABLED": "true",
            }
        )
