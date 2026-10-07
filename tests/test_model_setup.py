"""Deployment model setup uses synthetic assets, private config, and offline stubs only."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import socket
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from urllib import request

import pytest

from deploy.models import assets, setup
from deploy.models.assets import Artifact, AssetReport, Manifest

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "deploy/models/setup.py"
ALL_STACKS = frozenset({"base", "reid", "embedding", "semantic"})
SECRET = "synthetic-model-setup-secret"


@dataclass
class ModelBundle:
    source: Path
    env_file: Path
    manifest_file: Path
    values: dict[str, str]
    manifest: Manifest
    layout: dict[str, Path]
    payloads: dict[str, bytes]


def _asset(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _write_env(path: Path, values: dict[str, str]) -> None:
    path.write_text(
        "".join(f"{key}={shlex.quote(value)}\n" for key, value in values.items()),
        encoding="utf-8",
    )
    path.chmod(0o600)


def _write_manifest(path: Path, manifest: Manifest) -> None:
    path.write_text(
        json.dumps(
            {
                "version": manifest.version,
                "artifacts": [
                    {
                        "id": artifact.id,
                        "path": artifact.path,
                        "sha256": artifact.sha256,
                        "size_bytes": artifact.size_bytes,
                        "model": artifact.model,
                        "revision": artifact.revision,
                        "license": artifact.license,
                        "terms_url": artifact.terms_url,
                        "source_url": artifact.source_url,
                    }
                    for artifact in manifest.artifacts
                ],
            }
        ),
        encoding="utf-8",
    )


def _replace_artifact(manifest: Manifest, identifier: str, **changes) -> Manifest:
    return replace(
        manifest,
        artifacts=tuple(
            replace(artifact, **changes) if artifact.id == identifier else artifact
            for artifact in manifest.artifacts
        ),
    )


def _without_artifact(manifest: Manifest, identifier: str) -> Manifest:
    return replace(
        manifest,
        artifacts=tuple(artifact for artifact in manifest.artifacts if artifact.id != identifier),
    )


def _pipeline_revision(manifest: Manifest, source: Path) -> str:
    hashes = {artifact.id: artifact.sha256 for artifact in manifest.artifacts}
    rows = (
        ("model.pth", hashes["reid.checkpoint"]),
        ("model.yaml", hashes["reid.config"]),
        (
            "yolo_dfa.yaml",
            hashlib.sha256((source / setup.ALIGNER_CONFIG).read_bytes()).hexdigest(),
        ),
        ("mobilenetv4_Final.pth", hashes["reid.dfa"]),
        ("yolov8n-pose.pt", hashes["reid.pose"]),
    )
    return (
        "sha256:"
        + hashlib.sha256(
            "\n".join(f"{name}:{digest}" for name, digest in rows).encode("ascii")
        ).hexdigest()
    )


def _snapshot(directory: Path) -> dict[str, bytes | None]:
    return {
        str(path.relative_to(directory)): path.read_bytes() if path.is_file() else None
        for path in directory.rglob("*")
    }


@pytest.fixture
def bundle(tmp_path: Path, monkeypatch) -> ModelBundle:
    source = tmp_path / "source"
    _asset(source / setup.ALIGNER_CONFIG, b"aligner: synthetic-reviewed-dfa\n")
    models = tmp_path / "models"
    values = {
        "MODEL_DIR": str(models),
        "QWEN_EMBEDDING_MODEL_DIR": str(models / "qwen2b"),
        "APP_BASIC_AUTH_PASSWORD": SECRET,
        "VLM_API_KEY": SECRET,
        "VLM_PROVIDER": "openai_compatible",
        "VLM_BASE_URL": "https://synthetic-vlm.test/v1",
        "REID_ENABLED": "true",
        "SEMANTIC_SEARCH_ENABLED": "true",
        "VISUAL_EMBEDDING_DIM": "2048",
        "VISUAL_EMBEDDING_MODEL_REVISION": "synthetic-reviewed-qwen-revision",
        "REID_PREPROCESS_REVISION": "squarepad-v1",
    }
    layout = setup.model_layout(values, "containers", source, ALL_STACKS)
    payloads = {identifier: f"not-a-real-model:{identifier}".encode() for identifier in layout}
    artifacts = tuple(
        Artifact(
            id=identifier,
            path=f"offline/{identifier}.data",
            sha256=hashlib.sha256(payloads[identifier]).hexdigest(),
            size_bytes=len(payloads[identifier]),
            model=setup._family_model(identifier),
            revision="synthetic-reviewed-revision",
            license="synthetic-test-only-not-a-license-grant",
            terms_url="https://synthetic-publisher.test/terms?token=" + SECRET,
            source_url="https://synthetic-publisher.test/model?token=" + SECRET,
        )
        for identifier in layout
    )
    manifest = Manifest(version=1, artifacts=artifacts)
    values["REID_CHECKPOINT_REVISION"] = _pipeline_revision(manifest, source)
    for identifier, path in layout.items():
        _asset(path, payloads[identifier])
    hashes = {artifact.id: artifact.sha256 for artifact in artifacts}
    monkeypatch.setattr(setup, "QWEN_WEIGHTS_SHA256", hashes["qwen.weights"])
    monkeypatch.setattr(setup, "QWEN_RUNTIME_SHA256", hashes["qwen.runtime"])
    env_file = tmp_path / "private.env"
    manifest_file = tmp_path / "reviewed-manifest.json"
    _write_env(env_file, values)
    _write_manifest(manifest_file, manifest)
    return ModelBundle(source, env_file, manifest_file, values, manifest, layout, payloads)


def _arguments(bundle: ModelBundle, *extra: str, stacks: str = "base") -> list[str]:
    return [
        "--source",
        str(bundle.source),
        "--env-file",
        str(bundle.env_file),
        "--manifest",
        str(bundle.manifest_file),
        "--stacks",
        stacks,
        *extra,
    ]


def _forbid_operations(monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        pytest.fail("model configuration/check attempted a write, subprocess, or network operation")

    monkeypatch.setattr(setup, "prepare_assets", forbidden)
    monkeypatch.setattr(request, "urlopen", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)


def test_container_layout_maps_roles_to_fixed_runtime_locations(bundle: ModelBundle) -> None:
    models = Path(bundle.values["MODEL_DIR"])
    qwen = Path(bundle.values["QWEN_EMBEDDING_MODEL_DIR"])
    expected = {
        "yolo.person": models / "yolo11n.pt",
        **{
            f"face.{name.removesuffix('.onnx')}": models / "insightface/models/buffalo_l" / name
            for name in setup.FACE_FILES
        },
        "reid.checkpoint": models / "sapiensid_wb12m/model.pth",
        "reid.config": models / "sapiensid_wb12m/model.yaml",
        "reid.pose": models / "yolov8n-pose.pt",
        "reid.dfa": models / "dfa_mobilenetv4_medium/mobilenetv4_Final.pth",
        "qwen.weights": qwen / "model.safetensors",
        "qwen.config": qwen / "config.json",
        "qwen.tokenizer-config": qwen / "tokenizer_config.json",
        "qwen.tokenizer": qwen / "tokenizer.json",
        "qwen.processor": qwen / "preprocessor_config.json",
        "qwen.runtime": qwen / "scripts/qwen3_vl_embedding.py",
    }
    assert bundle.layout == expected
    assert len([identifier for identifier in bundle.layout if identifier.startswith("face.")]) == 5


@pytest.mark.parametrize(
    "stacks",
    [{"base"}, {"base", "reid"}, {"base", "embedding"}, {"base", "embedding", "semantic"}],
)
def test_manifest_selects_only_requested_runtime_stacks(
    bundle: ModelBundle, stacks: set[str]
) -> None:
    layout = setup.model_layout(bundle.values, "containers", bundle.source, frozenset(stacks))
    selected = setup.select_manifest(bundle.manifest, layout, bundle.values, bundle.source)
    assert {artifact.id for artifact in selected.artifacts} == set(layout)
    assert any(artifact.id.startswith("reid.") for artifact in selected.artifacts) == (
        "reid" in stacks
    )
    assert any(artifact.id.startswith("qwen.") for artifact in selected.artifacts) == (
        "embedding" in stacks
    )


@pytest.mark.parametrize("stacks", [frozenset(), frozenset({"reid"}), frozenset({"base", "bad"})])
def test_unknown_or_missing_base_stack_is_rejected(bundle: ModelBundle, stacks) -> None:
    with pytest.raises(setup.ModelSetupError, match="known deployment stacks"):
        setup.model_layout(bundle.values, "containers", bundle.source, stacks)


def test_semantic_requires_embedding_stack(bundle: ModelBundle) -> None:
    with pytest.raises(setup.ModelSetupError, match="semantic requires embedding"):
        setup.model_layout(
            bundle.values, "containers", bundle.source, frozenset({"base", "semantic"})
        )


@pytest.mark.parametrize(
    "value",
    [
        "",
        "models",
        "/",
        "/tmp",
        "/private/tmp",
        "/var",
        "/private/var",
        "/data",
        "/opt",
        "/Users",
        "/etc",
        "/usr",
        "/bin",
        "/sbin",
        "/private",
        "/Library",
        "/Applications",
        str(Path.home()),
    ],
)
@pytest.mark.parametrize("key", ["MODEL_DIR", "QWEN_EMBEDDING_MODEL_DIR"])
def test_container_model_paths_must_be_explicit_dedicated_absolute_paths(
    bundle: ModelBundle, key: str, value: str
) -> None:
    values = {**bundle.values, key: value}
    with pytest.raises(setup.ModelSetupError, match=key):
        setup.model_layout(values, "containers", bundle.source, ALL_STACKS)


def _rtx_values(bundle: ModelBundle) -> dict[str, str]:
    return {
        **bundle.values,
        "YOLO_MODEL": "/var/lib/sightindex/models/yolo11n.pt",
        "FACE_INSIGHTFACE_ROOT": "/var/lib/sightindex/models/insightface",
        "FACE_INSIGHTFACE_MODEL": "buffalo_l",
        "REID_CHECKPOINT_DIR": "/var/lib/sightindex/models/sapiensid_wb12m",
        "REID_POSE_WEIGHTS": "/var/lib/sightindex/.cache/yolov8n-pose.pt",
    }


def _trust_synthetic_rtx_source(bundle: ModelBundle, monkeypatch) -> None:
    original = Path.resolve
    monkeypatch.setattr(
        Path,
        "resolve",
        lambda path, *args, **kwargs: (
            Path("/opt/sightindex") if path == bundle.source else original(path, *args, **kwargs)
        ),
    )


def test_rtx_layout_uses_exact_cache_and_vendored_dfa_paths(
    bundle: ModelBundle, monkeypatch
) -> None:
    _trust_synthetic_rtx_source(bundle, monkeypatch)
    layout = setup.model_layout(
        _rtx_values(bundle), "rtx5090", bundle.source, frozenset({"base", "reid"})
    )
    assert layout["reid.pose"] == Path("/var/lib/sightindex/.cache/yolov8n-pose.pt")
    assert layout["reid.dfa"] == bundle.source / setup.DFA_RELATIVE
    assert layout["reid.checkpoint"] == Path("/var/lib/sightindex/models/sapiensid_wb12m/model.pth")
    assert layout["face.genderage"] == Path(
        "/var/lib/sightindex/models/insightface/models/buffalo_l/genderage.onnx"
    )


def test_rtx_rejects_untrusted_checkout(bundle: ModelBundle) -> None:
    with pytest.raises(setup.ModelSetupError, match="trusted /opt/sightindex"):
        setup.model_layout(_rtx_values(bundle), "rtx5090", bundle.source, frozenset({"base"}))


@pytest.mark.parametrize("stacks", [{"base", "embedding"}, {"base", "embedding", "semantic"}])
def test_rtx_rejects_unmanaged_qwen_stack(bundle: ModelBundle, monkeypatch, stacks) -> None:
    _trust_synthetic_rtx_source(bundle, monkeypatch)
    with pytest.raises(setup.ModelSetupError, match="does not manage Qwen"):
        setup.model_layout(_rtx_values(bundle), "rtx5090", bundle.source, frozenset(stacks))


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("FACE_INSIGHTFACE_MODEL", "buffalo_s", "buffalo_l"),
        ("REID_POSE_WEIGHTS", "/var/lib/sightindex/models/yolov8n-pose.pt", "fixed cache"),
    ],
)
def test_rtx_rejects_runtime_incompatible_model_configuration(
    bundle: ModelBundle, monkeypatch, key: str, value: str, message: str
) -> None:
    _trust_synthetic_rtx_source(bundle, monkeypatch)
    values = {**_rtx_values(bundle), key: value}
    with pytest.raises(setup.ModelSetupError, match=message):
        setup.model_layout(values, "rtx5090", bundle.source, frozenset({"base", "reid"}))


@pytest.mark.parametrize("name", setup.FACE_FILES)
def test_every_buffalo_l_onnx_role_is_required(bundle: ModelBundle, name: str) -> None:
    identifier = "face." + name.removesuffix(".onnx")
    with pytest.raises(setup.ModelSetupError, match=identifier):
        setup.select_manifest(
            _without_artifact(bundle.manifest, identifier),
            dict(bundle.layout),
            bundle.values,
            bundle.source,
        )


@pytest.mark.parametrize("identifier", ["unknown.model", "vlm.weights", "qwen.shard.1"])
def test_unknown_manifest_roles_cannot_select_destinations(
    bundle: ModelBundle, identifier: str
) -> None:
    original = bundle.manifest.artifacts[0]
    artifact = replace(original, id=identifier, path="offline/unrecognized.data")
    manifest = replace(bundle.manifest, artifacts=(*bundle.manifest.artifacts, artifact))
    with pytest.raises(setup.ModelSetupError, match="unknown model artifact role"):
        setup.select_manifest(manifest, dict(bundle.layout), bundle.values, bundle.source)


@pytest.mark.parametrize(
    ("identifier", "wrong_model"),
    [
        ("yolo.person", "yolo26n"),
        ("face.det_10g", "buffalo_m"),
        ("reid.checkpoint", "sapiensid_wb4m"),
        ("qwen.weights", "Qwen/Qwen3-Embedding-2B"),
    ],
)
def test_manifest_model_identity_must_agree_with_role(
    bundle: ModelBundle, identifier: str, wrong_model: str
) -> None:
    with pytest.raises(setup.ModelSetupError, match="model identity"):
        setup.select_manifest(
            _replace_artifact(bundle.manifest, identifier, model=wrong_model),
            dict(bundle.layout),
            bundle.values,
            bundle.source,
        )


@pytest.mark.parametrize("field", ["license", "terms_url"])
@pytest.mark.parametrize(
    "identifier", ["yolo.person", "face.genderage", "reid.config", "qwen.config"]
)
def test_every_artifact_requires_license_and_publisher_terms(
    bundle: ModelBundle, field: str, identifier: str
) -> None:
    with pytest.raises(setup.ModelSetupError, match="license and publisher terms URL"):
        setup.select_manifest(
            _replace_artifact(bundle.manifest, identifier, **{field: None}),
            dict(bundle.layout),
            bundle.values,
            bundle.source,
        )


@pytest.mark.parametrize("identifier", ["qwen.weights", "qwen.runtime"])
def test_qwen_single_file_weights_and_runtime_must_match_shipped_pins(
    bundle: ModelBundle, identifier: str
) -> None:
    with pytest.raises(setup.ModelSetupError, match="embedding image pins"):
        setup.select_manifest(
            _replace_artifact(bundle.manifest, identifier, sha256="0" * 64),
            dict(bundle.layout),
            bundle.values,
            bundle.source,
        )


@pytest.mark.parametrize("identifier", ["qwen.weights", "qwen.runtime"])
def test_qwen_shards_cannot_replace_required_monolithic_weights_or_runtime(
    bundle: ModelBundle, identifier: str
) -> None:
    with pytest.raises(setup.ModelSetupError, match=identifier):
        setup.select_manifest(
            _without_artifact(bundle.manifest, identifier),
            dict(bundle.layout),
            bundle.values,
            bundle.source,
        )


def test_qwen_files_must_use_one_reviewed_revision(bundle: ModelBundle) -> None:
    with pytest.raises(setup.ModelSetupError, match="one reviewed revision"):
        setup.select_manifest(
            _replace_artifact(bundle.manifest, "qwen.config", revision="other-revision"),
            dict(bundle.layout),
            bundle.values,
            bundle.source,
        )


@pytest.mark.parametrize("name", sorted(setup.QWEN_EXTRA_FILES))
def test_optional_qwen_file_roles_map_only_to_whitelisted_basenames(
    bundle: ModelBundle, name: str
) -> None:
    original = next(
        artifact for artifact in bundle.manifest.artifacts if artifact.id == "qwen.config"
    )
    extra = replace(original, id="qwen.extra." + name, path="offline/extra/" + name)
    manifest = replace(bundle.manifest, artifacts=(*bundle.manifest.artifacts, extra))
    layout = dict(bundle.layout)
    selected = setup.select_manifest(manifest, layout, bundle.values, bundle.source)
    assert extra in selected.artifacts
    assert layout[extra.id] == Path(bundle.values["QWEN_EMBEDDING_MODEL_DIR"]) / name


@pytest.mark.parametrize("name", ["chat_template.jinja", "video_preprocessor_config.json"])
def test_official_qwen_processor_extras_can_be_imported_as_reviewed_fixed_basenames(
    bundle: ModelBundle, name: str
) -> None:
    original = next(
        artifact for artifact in bundle.manifest.artifacts if artifact.id == "qwen.config"
    )
    extra = replace(original, id="qwen.extra." + name, path="offline/extra/" + name)
    manifest = replace(bundle.manifest, artifacts=(*bundle.manifest.artifacts, extra))
    layout = dict(bundle.layout)
    selected = setup.select_manifest(manifest, layout, bundle.values, bundle.source)
    assert extra in selected.artifacts
    assert layout[extra.id] == Path(bundle.values["QWEN_EMBEDDING_MODEL_DIR"]) / name


@pytest.mark.parametrize(
    "name", ["unreviewed.py", "model-00001-of-00002.safetensors", "config.json"]
)
def test_unsupported_optional_qwen_roles_are_rejected(bundle: ModelBundle, name: str) -> None:
    original = next(
        artifact for artifact in bundle.manifest.artifacts if artifact.id == "qwen.config"
    )
    extra = replace(original, id="qwen.extra." + name, path="offline/extra/" + name)
    manifest = replace(bundle.manifest, artifacts=(*bundle.manifest.artifacts, extra))
    with pytest.raises(setup.ModelSetupError, match="unsupported optional Qwen file role"):
        setup.select_manifest(manifest, dict(bundle.layout), bundle.values, bundle.source)


def test_qwen_extra_files_are_not_prepared_without_embedding_stack(bundle: ModelBundle) -> None:
    original = next(
        artifact for artifact in bundle.manifest.artifacts if artifact.id == "qwen.config"
    )
    extra = replace(original, id="qwen.extra.vocab.json", path="offline/extra/vocab.json")
    manifest = replace(bundle.manifest, artifacts=(*bundle.manifest.artifacts, extra))
    layout = setup.model_layout(bundle.values, "containers", bundle.source, frozenset({"base"}))
    selected = setup.select_manifest(manifest, layout, bundle.values, bundle.source)
    assert not any(artifact.id.startswith("qwen.") for artifact in selected.artifacts)


def test_reid_fingerprint_uses_the_exact_five_runtime_inputs(bundle: ModelBundle) -> None:
    before_values = dict(bundle.values)
    selected = setup.select_manifest(
        bundle.manifest, dict(bundle.layout), bundle.values, bundle.source
    )
    assert selected.artifacts == bundle.manifest.artifacts
    assert bundle.values == before_values
    assert bundle.values["REID_CHECKPOINT_REVISION"] == _pipeline_revision(
        bundle.manifest, bundle.source
    )


@pytest.mark.parametrize("identifier", ["reid.checkpoint", "reid.config", "reid.dfa", "reid.pose"])
def test_reid_asset_drift_rejects_manifest_before_any_prepare_or_configuration_write(
    bundle: ModelBundle, monkeypatch, capsys, identifier: str
) -> None:
    changed = _replace_artifact(bundle.manifest, identifier, sha256="1" * 64)
    _write_manifest(bundle.manifest_file, changed)
    before = _snapshot(bundle.source.parent)
    _forbid_operations(monkeypatch)
    assert (
        setup.main(
            _arguments(bundle, "--acknowledge-model-terms", stacks="base reid embedding semantic")
        )
        == 1
    )
    output = capsys.readouterr()
    assert "pipeline fingerprint differs" in output.err
    assert SECRET not in output.err + output.out
    assert _snapshot(bundle.source.parent) == before


def test_reid_aligner_config_drift_cannot_automatically_change_model_identity(
    bundle: ModelBundle, monkeypatch, capsys
) -> None:
    _asset(bundle.source / setup.ALIGNER_CONFIG, b"aligner: changed\n")
    before = _snapshot(bundle.source.parent)
    _forbid_operations(monkeypatch)
    assert setup.main(_arguments(bundle, "--acknowledge-model-terms", stacks="base reid")) == 1
    assert "pipeline fingerprint differs" in capsys.readouterr().err
    assert _snapshot(bundle.source.parent) == before


@pytest.mark.parametrize("bad_revision", ["", "sha256:" + "a" * 64, "unreviewed"])
def test_reid_configuration_fingerprint_must_match_manifest(
    bundle: ModelBundle, bad_revision: str
) -> None:
    values = {**bundle.values, "REID_CHECKPOINT_REVISION": bad_revision}
    with pytest.raises(setup.ModelSetupError, match="pipeline fingerprint differs"):
        setup.select_manifest(bundle.manifest, dict(bundle.layout), values, bundle.source)


@pytest.mark.parametrize("mode", ["missing", "symlink"])
def test_reid_requires_regular_trusted_source_aligner_config(
    bundle: ModelBundle, mode: str
) -> None:
    config = bundle.source / setup.ALIGNER_CONFIG
    config.unlink()
    if mode == "symlink":
        alternate = bundle.source / "alternate-aligner.yaml"
        _asset(alternate, b"synthetic")
        config.symlink_to(alternate)
    with pytest.raises(setup.ModelSetupError, match="trusted source lacks"):
        setup.select_manifest(bundle.manifest, dict(bundle.layout), bundle.values, bundle.source)


def test_main_readonly_check_does_not_fetch_vlm_or_modify_config_models_or_fingerprints(
    bundle: ModelBundle, monkeypatch, capsys
) -> None:
    before = _snapshot(bundle.source.parent)
    _forbid_operations(monkeypatch)
    assert setup.main(_arguments(bundle, "--check", stacks="base reid embedding semantic")) == 0
    output = capsys.readouterr()
    report = json.loads(output.out)
    assert report["ok"] is True
    assert report["mode"] == "check"
    assert "external endpoint only" in report["vlm"]
    assert "no VLM weights" in report["vlm"]
    assert "does not grant commercial" in report["license_notice"]
    assert SECRET not in output.out + output.err
    assert _snapshot(bundle.source.parent) == before


@pytest.mark.parametrize("missing_name", setup.FACE_FILES)
def test_check_reports_any_missing_full_face_pack_file_without_repair(
    bundle: ModelBundle, monkeypatch, capsys, missing_name: str
) -> None:
    identifier = "face." + missing_name.removesuffix(".onnx")
    bundle.layout[identifier].unlink()
    before = _snapshot(bundle.source.parent)
    _forbid_operations(monkeypatch)
    assert setup.main(_arguments(bundle, "--check")) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is False
    assert any(
        item["id"] == identifier and item["status"] == "missing" for item in report["results"]
    )
    assert _snapshot(bundle.source.parent) == before


@pytest.mark.parametrize("option", ["--download", "--source-dir"])
def test_check_rejects_all_preparation_options_before_calling_asset_layer(
    bundle: ModelBundle, monkeypatch, capsys, option: str
) -> None:
    _forbid_operations(monkeypatch)
    monkeypatch.setattr(setup, "check_assets", lambda *args, **kwargs: pytest.fail("check called"))
    extra = (option, str(bundle.source)) if option == "--source-dir" else (option,)
    before = _snapshot(bundle.source.parent)
    assert setup.main(_arguments(bundle, "--check", *extra)) == 1
    assert "read-only" in capsys.readouterr().err
    assert _snapshot(bundle.source.parent) == before


def test_offline_prepare_requires_explicit_terms_acknowledgement(
    bundle: ModelBundle, monkeypatch, capsys
) -> None:
    _forbid_operations(monkeypatch)
    before = _snapshot(bundle.source.parent)
    assert setup.main(_arguments(bundle, "--source-dir", str(bundle.source))) == 1
    error = capsys.readouterr().err
    assert "--acknowledge-model-terms" in error
    assert "not a commercial license grant" in error
    assert _snapshot(bundle.source.parent) == before


@pytest.mark.parametrize("download", [False, True])
@pytest.mark.parametrize("ok", [False, True])
def test_prepare_passes_only_selected_fixed_destinations_and_explicit_download_authority(
    bundle: ModelBundle, monkeypatch, capsys, download: bool, ok: bool
) -> None:
    calls = []

    def fake_prepare(manifest, destination, **kwargs):
        calls.append((manifest, destination, kwargs))
        return AssetReport(mode="prepare", ok=ok)

    monkeypatch.setattr(setup, "prepare_assets", fake_prepare)
    monkeypatch.setattr(setup, "check_assets", lambda *args, **kwargs: pytest.fail("check called"))
    before = _snapshot(bundle.source.parent)
    extra = ("--download",) if download else ("--source-dir", str(bundle.source))
    assert setup.main(_arguments(bundle, "--acknowledge-model-terms", *extra)) == (0 if ok else 1)
    assert len(calls) == 1
    manifest, destination, kwargs = calls[0]
    assert destination == Path(bundle.values["MODEL_DIR"])
    assert kwargs["download"] is download
    assert kwargs["source_dir"] == (None if download else bundle.source)
    assert len(manifest.artifacts) == 6
    for artifact in manifest.artifacts:
        assert kwargs["destination_for"](artifact) == bundle.layout[artifact.id]
        assert kwargs["destination_for"](artifact) != destination / artifact.path
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is ok
    assert "does not grant commercial" in report["license_notice"]
    assert _snapshot(bundle.source.parent) == before


def test_offline_full_stack_prepare_check_and_reuse_use_real_asset_layer(
    bundle: ModelBundle, monkeypatch, capsys
) -> None:
    def forbidden(*args, **kwargs):
        pytest.fail("offline model preparation attempted network access")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    before_env = bundle.env_file.read_bytes()
    offline = bundle.source.parent / "offline-bundle"
    for artifact in bundle.manifest.artifacts:
        _asset(offline / artifact.path, bundle.payloads[artifact.id])
        bundle.layout[artifact.id].unlink()
    args = _arguments(
        bundle,
        "--source-dir",
        str(offline),
        "--acknowledge-model-terms",
        stacks="base reid embedding semantic",
    )
    assert setup.main(args) == 0
    report = json.loads(capsys.readouterr().out)
    assert len(report["results"]) == 16
    assert {result["status"] for result in report["results"]} == {"prepared"}
    for identifier, path in bundle.layout.items():
        assert path.read_bytes() == bundle.payloads[identifier]
        assert path.stat().st_mode & 0o777 == 0o644
        assert not path.with_name(path.name + ".partial").exists()
    assert setup.main(_arguments(bundle, "--check", stacks="base reid embedding semantic")) == 0
    assert {r["status"] for r in json.loads(capsys.readouterr().out)["results"]} == {"verified"}
    before = _snapshot(Path(bundle.values["MODEL_DIR"]))
    assert setup.main(args) == 0
    assert {r["status"] for r in json.loads(capsys.readouterr().out)["results"]} == {"reused"}
    assert _snapshot(Path(bundle.values["MODEL_DIR"])) == before
    assert bundle.env_file.read_bytes() == before_env


def test_invalid_cli_arguments_never_echo_secret_values(capsys) -> None:
    assert setup.main(["--target", "https://synthetic.test/?token=" + SECRET]) == 1
    output = capsys.readouterr()
    assert "invalid model setup arguments" in output.err
    assert SECRET not in output.out + output.err


@pytest.mark.parametrize("target", ["containers", "rtx5090"])
def test_deployment_unknown_arguments_never_echo_signed_model_urls(target: str) -> None:
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "deploy.sh"),
            "--target",
            target,
            "https://synthetic.test/model?token=" + SECRET,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "unknown" in result.stderr
    assert SECRET not in result.stdout + result.stderr


@pytest.mark.parametrize("operation", ["load_manifest", "check_assets", "prepare_assets"])
def test_unexpected_model_setup_errors_do_not_echo_credentials_or_exception_content(
    bundle: ModelBundle, monkeypatch, capsys, operation: str
) -> None:
    def failure(*args, **kwargs):
        raise RuntimeError("remote diagnostic " + SECRET)

    monkeypatch.setattr(setup, operation, failure)
    option = "--check" if operation != "prepare_assets" else "--acknowledge-model-terms"
    assert setup.main(_arguments(bundle, option)) == 1
    output = capsys.readouterr()
    assert "RuntimeError" in output.err
    assert SECRET not in output.err + output.out
    assert "remote diagnostic" not in output.err + output.out


def test_cli_check_is_readonly_with_secret_bearing_urls_and_external_vlm_config(
    bundle: ModelBundle,
) -> None:
    before = _snapshot(bundle.source.parent)
    result = subprocess.run(
        [sys.executable, str(HELPER), *_arguments(bundle, "--check")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["ok"] is True
    assert len(report["results"]) == 6
    assert SECRET not in result.stdout + result.stderr
    assert _snapshot(bundle.source.parent) == before


@pytest.mark.parametrize("invalid", ["VLM_API_KEY=$(synthetic-secret)", "VLM_API_KEY=duplicate"])
def test_cli_invalid_shell_or_duplicate_env_fails_without_echoing_secrets(
    bundle: ModelBundle, invalid: str
) -> None:
    bundle.env_file.write_text(bundle.env_file.read_text() + invalid + "\n", encoding="utf-8")
    before = _snapshot(bundle.source.parent)
    result = subprocess.run(
        [sys.executable, str(HELPER), *_arguments(bundle, "--check")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "VLM_API_KEY" in result.stderr
    assert SECRET not in result.stdout + result.stderr
    assert "synthetic-secret" not in result.stdout + result.stderr
    assert _snapshot(bundle.source.parent) == before


def test_cli_missing_destination_is_not_created_during_check(bundle: ModelBundle) -> None:
    values = {**bundle.values, "MODEL_DIR": str(bundle.source.parent / "absent-models")}
    _write_env(bundle.env_file, values)
    before = _snapshot(bundle.source.parent)
    result = subprocess.run(
        [sys.executable, str(HELPER), *_arguments(bundle, "--check")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert json.loads(result.stdout)["ok"] is False
    assert not Path(values["MODEL_DIR"]).exists()
    assert _snapshot(bundle.source.parent) == before


@pytest.mark.parametrize("problem", ["missing-face", "wrong-qwen-pin", "wrong-model", "no-license"])
def test_invalid_manifest_is_rejected_before_prepare_even_with_terms_acknowledged(
    bundle: ModelBundle, monkeypatch, capsys, problem: str
) -> None:
    manifest = bundle.manifest
    if problem == "missing-face":
        manifest = _without_artifact(manifest, "face.genderage")
    elif problem == "wrong-qwen-pin":
        manifest = _replace_artifact(manifest, "qwen.weights", sha256="0" * 64)
    elif problem == "wrong-model":
        manifest = _replace_artifact(manifest, "yolo.person", model="yolo26n")
    else:
        manifest = _replace_artifact(manifest, "reid.config", license=None)
    _write_manifest(bundle.manifest_file, manifest)
    before = _snapshot(bundle.source.parent)
    _forbid_operations(monkeypatch)
    assert (
        setup.main(
            _arguments(bundle, "--acknowledge-model-terms", stacks="base reid embedding semantic")
        )
        == 1
    )
    output = capsys.readouterr()
    assert output.out == ""
    assert "ERROR:" in output.err
    assert SECRET not in output.err
    assert _snapshot(bundle.source.parent) == before


@pytest.mark.parametrize("target", ["containers", "rtx5090"])
@pytest.mark.parametrize("exit_code", [0, 29])
def test_root_model_only_adapter_passes_profile_defaults_and_original_exit_without_deploying(
    bundle: ModelBundle, tmp_path: Path, target: str, exit_code: int
) -> None:
    wrapper_root = tmp_path / "wrapper checkout"
    _asset(wrapper_root / "deploy.sh", (ROOT / "deploy.sh").read_bytes())
    _asset(
        wrapper_root / "deploy/models/manage.sh", (ROOT / "deploy/models/manage.sh").read_bytes()
    )
    mock_python = tmp_path / "mock python"
    mock_python.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        "print(json.dumps(sys.argv[1:]))\n"
        f"raise SystemExit({exit_code})\n",
        encoding="utf-8",
    )
    mock_python.chmod(0o700)
    model_root = tmp_path / "model root not created"
    before = _snapshot(tmp_path)
    result = subprocess.run(
        [
            "bash",
            str(wrapper_root / "deploy.sh"),
            "--models-only",
            "--target",
            target,
            "--root",
            str(model_root),
            "--source",
            str(bundle.source),
            "--model-manifest",
            str(bundle.manifest_file),
            "--check",
        ],
        env={**os.environ, "SIGHTINDEX_DEPLOY_PYTHON": str(mock_python)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == exit_code, result.stderr
    argv = json.loads(result.stdout)
    assert argv == [
        str(bundle.source / "deploy/models/setup.py"),
        "--target",
        target,
        "--source",
        str(bundle.source),
        "--env-file",
        str(model_root / ".env" if target == "containers" else bundle.source / ".env"),
        "--stacks",
        "base" if target == "containers" else "base reid",
        "--manifest",
        str(bundle.manifest_file),
        "--check",
    ]
    assert not model_root.exists()
    assert _snapshot(tmp_path) == before


def test_model_diagnostic_uses_real_readonly_layer_and_never_changes_private_configuration(
    bundle: ModelBundle,
    monkeypatch,
    capsys,
) -> None:
    models = Path(bundle.values["MODEL_DIR"])
    lock = models / assets.LOCK_NAME
    lock.write_bytes(b"")
    before = _snapshot(bundle.source.parent)
    _forbid_operations(monkeypatch)
    assert setup.main(_arguments(bundle, "--diagnose-models", stacks="base reid")) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["mode"] == "diagnose"
    assert report["lock"]["code"] == "lock.legacy_empty"
    assert SECRET not in json.dumps(report)
    assert _snapshot(bundle.source.parent) == before


def test_model_recovery_requires_confirmation_before_any_write(
    bundle: ModelBundle,
    monkeypatch,
    capsys,
) -> None:
    _forbid_operations(monkeypatch)
    monkeypatch.setattr(setup, "recover_assets", lambda *a, **kw: pytest.fail("recovery called"))
    before = _snapshot(bundle.source.parent)
    assert setup.main(_arguments(bundle, "--recover-models")) == 1
    assert "--confirm-no-active-preparation" in capsys.readouterr().err
    assert _snapshot(bundle.source.parent) == before


@pytest.mark.parametrize("mode", ["--diagnose-models", "--recover-models", "--check"])
@pytest.mark.parametrize("extra", [["--download"], ["--source-dir", "/not/a/source"]])
def test_maintenance_and_checks_refuse_all_transfer_flags(
    bundle: ModelBundle,
    monkeypatch,
    capsys,
    mode: str,
    extra: list[str],
) -> None:
    _forbid_operations(monkeypatch)
    monkeypatch.setattr(setup, "diagnose_assets", lambda *a, **kw: pytest.fail("diagnose called"))
    monkeypatch.setattr(setup, "recover_assets", lambda *a, **kw: pytest.fail("recover called"))
    before = _snapshot(bundle.source.parent)
    assert setup.main(_arguments(bundle, mode, *extra)) == 1
    assert "cannot prepare/download" in capsys.readouterr().err
    assert _snapshot(bundle.source.parent) == before


@pytest.mark.parametrize(
    "extra",
    [
        ["--diagnose-models", "--check"],
        ["--recover-models", "--diagnose-models"],
        ["--diagnose-models", "--acknowledge-model-terms"],
        ["--confirm-no-active-preparation"],
        ["--diagnose-models", "--quarantine-partial", SECRET],
    ],
)
def test_maintenance_flags_do_not_leak_values_or_allow_ambiguous_modes(
    bundle: ModelBundle,
    capsys,
    extra: list[str],
) -> None:
    before = _snapshot(bundle.source.parent)
    assert setup.main(_arguments(bundle, *extra)) == 1
    output = capsys.readouterr()
    assert SECRET not in output.out + output.err
    assert _snapshot(bundle.source.parent) == before


def test_model_recovery_still_validates_reid_manifest_identity_before_lock_write(
    bundle: ModelBundle,
    capsys,
) -> None:
    changed = _replace_artifact(bundle.manifest, "reid.pose", sha256="0" * 64)
    _write_manifest(bundle.manifest_file, changed)
    lock = Path(bundle.values["MODEL_DIR"]) / assets.LOCK_NAME
    lock.write_bytes(b"")
    before = _snapshot(bundle.source.parent)
    assert (
        setup.main(
            _arguments(
                bundle,
                "--recover-models",
                "--confirm-no-active-preparation",
                stacks="base reid",
            )
        )
        == 1
    )
    assert "pipeline fingerprint differs" in capsys.readouterr().err
    assert _snapshot(bundle.source.parent) == before


def test_model_recovery_uses_selected_roles_only_and_does_not_modify_env_or_indexes(
    bundle: ModelBundle,
    capsys,
) -> None:
    models = Path(bundle.values["MODEL_DIR"])
    lock = models / assets.LOCK_NAME
    lock.write_bytes(b"")
    yolo = bundle.layout["yolo.person"]
    yolo.unlink()
    partial = yolo.with_name(yolo.name + ".partial")
    partial.write_bytes(b"x" * len(bundle.payloads["yolo.person"]))
    qwen = bundle.layout["qwen.weights"]
    qwen.unlink()
    unselected = qwen.with_name(qwen.name + ".partial")
    unselected.write_bytes(b"unselected Qwen must be preserved")
    before_config = bundle.env_file.read_bytes()
    before_manifest = bundle.manifest_file.read_bytes()
    assert (
        setup.main(
            _arguments(
                bundle,
                "--recover-models",
                "--confirm-no-active-preparation",
                "--quarantine-partial",
                "yolo.person",
            )
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["mode"] == "recover" and report["lock"]["code"] == "lock.legacy_quarantined"
    assert report["results"][0]["status"] == "quarantined"
    assert not partial.exists() and not yolo.exists()
    assert unselected.read_bytes() == b"unselected Qwen must be preserved"
    assert bundle.env_file.read_bytes() == before_config
    assert bundle.manifest_file.read_bytes() == before_manifest
    assert SECRET not in json.dumps(report)


@pytest.mark.parametrize("mode", ["--diagnose-models", "--recover-models"])
def test_model_only_adapter_forwards_maintenance_options_without_deployment(
    bundle: ModelBundle,
    tmp_path: Path,
    mode: str,
) -> None:
    mock_python = tmp_path / "mock-maintenance-python"
    mock_python.write_text(
        f"#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n",
        encoding="utf-8",
    )
    mock_python.chmod(0o700)
    options = [mode]
    if mode == "--recover-models":
        options.extend(["--confirm-no-active-preparation", "--quarantine-partial", "yolo.person"])
    before = _snapshot(tmp_path)
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "deploy.sh"),
            "--models-only",
            "--source",
            str(bundle.source),
            "--env-file",
            str(bundle.env_file),
            "--model-manifest",
            str(bundle.manifest_file),
            *options,
        ],
        env={**os.environ, "SIGHTINDEX_DEPLOY_PYTHON": str(mock_python)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    argv = json.loads(result.stdout)
    assert argv[-len(options) :] == options
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize("online", [False, True])
def test_model_only_adapter_translates_offline_or_explicit_online_options_without_service_actions(
    bundle: ModelBundle, tmp_path: Path, online: bool
) -> None:
    mock_python = tmp_path / "mock-python"
    mock_python.write_text(
        f"#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n",
        encoding="utf-8",
    )
    mock_python.chmod(0o700)
    extra = ["--download-models"] if online else ["--model-source", str(bundle.source)]
    before = _snapshot(tmp_path)
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "deploy/models/manage.sh"),
            "--source",
            str(bundle.source),
            "--env-file",
            str(bundle.env_file),
            "--stacks",
            "base reid embedding semantic",
            "--model-manifest",
            str(bundle.manifest_file),
            "--prepare-models",
            "--acknowledge-model-terms",
            *extra,
        ],
        env={**os.environ, "SIGHTINDEX_DEPLOY_PYTHON": str(mock_python)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    argv = json.loads(result.stdout)
    assert "--prepare-models" not in argv
    assert "--acknowledge-model-terms" in argv
    assert ("--download" in argv) is online
    assert ("--source-dir" in argv) is not online
    if not online:
        assert argv[argv.index("--source-dir") + 1] == str(bundle.source)
    assert argv[argv.index("--stacks") + 1] == "base reid embedding semantic"
    assert _snapshot(tmp_path) == before
