#!/usr/bin/env python3
"""Map a reviewed model lockfile to deployment assets without changing model identity."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

# This module is also invoked directly, before the application venv exists.
REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from deploy.containers.preflight import (  # noqa: E402
    DFA_RELATIVE,
    KNOWN_STACKS,
    PreflightError,
    read_env,
)
from deploy.models.assets import (  # noqa: E402
    Artifact,
    Manifest,
    check_assets,
    diagnose_assets,
    load_manifest,
    prepare_assets,
    recover_assets,
)

QWEN_MODEL = "Qwen/Qwen3-VL-Embedding-2B"
QWEN_WEIGHTS_SHA256 = "c73fa9caeddeb3ff831d46c085a7a5708343248ca777e90f2d486964464509c1"
QWEN_RUNTIME_SHA256 = "8ffa74a1a6bb759610c57865ea416fd4daf9936cb787520e1112a3e1d547f36a"
FACE_FILES = ("det_10g.onnx", "w600k_r50.onnx", "1k3d68.onnx", "2d106det.onnx", "genderage.onnx")
QWEN_EXTRA_FILES = frozenset(
    {
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "added_tokens.json",
        "processor_config.json",
        "generation_config.json",
        "chat_template.jinja",
        "video_preprocessor_config.json",
    }
)
ALIGNER_CONFIG = Path(
    "deploy/agx/reid_service/sapiensid/tasks/sapiensID/src/aligners/configs/yolo_dfa.yaml"
)


class ModelSetupError(ValueError):
    """A bounded model configuration diagnostic safe to print."""


class SafeParser(argparse.ArgumentParser):
    """Avoid echoing arbitrary CLI values, including signed model URLs."""

    def error(self, message: str) -> None:
        raise ModelSetupError("invalid model setup arguments; use --help")


def _path(values: dict[str, str], key: str) -> Path:
    """Require an explicit, dedicated absolute model path."""
    value = values.get(key, "")
    path = Path(value)
    broad_paths = {
        Path("/"),
        Path.home(),
        Path("/tmp"),
        Path("/private/tmp"),
        Path("/var"),
        Path("/private/var"),
        Path("/data"),
        Path("/opt"),
        Path("/Users"),
        Path("/etc"),
        Path("/usr"),
        Path("/bin"),
        Path("/sbin"),
        Path("/private"),
        Path("/Library"),
        Path("/Applications"),
    }
    if (
        not value
        or not path.is_absolute()
        or path in broad_paths
        or path.resolve() in {candidate.resolve() for candidate in broad_paths}
    ):
        raise ModelSetupError(f"configure a dedicated absolute {key}")
    return path


def model_layout(
    values: dict[str, str], target: str, source: Path, stacks: frozenset[str]
) -> dict[str, Path]:
    """Return fixed model roles; manifest paths cannot select host destinations."""
    if "base" not in stacks or stacks - KNOWN_STACKS:
        raise ModelSetupError("select base and only known deployment stacks")
    if "semantic" in stacks and "embedding" not in stacks:
        raise ModelSetupError("semantic requires embedding")
    if target == "containers":
        models = _path(values, "MODEL_DIR")
        yolo = models / "yolo11n.pt"
        face = models / "insightface/models/buffalo_l"
        reid = models / "sapiensid_wb12m"
        pose = models / "yolov8n-pose.pt"
        dfa = models / "dfa_mobilenetv4_medium/mobilenetv4_Final.pth"
    elif target == "rtx5090":
        if source.resolve() != Path("/opt/sightindex"):
            raise ModelSetupError(
                "RTX model preparation requires the trusted /opt/sightindex checkout"
            )
        if "embedding" in stacks or "semantic" in stacks:
            raise ModelSetupError(
                "RTX systemd profile does not manage Qwen; use the container profile"
            )
        yolo = _path(values, "YOLO_MODEL")
        if values.get("FACE_INSIGHTFACE_MODEL", "buffalo_l") != "buffalo_l":
            raise ModelSetupError("this model setup supports the buffalo_l face pack only")
        face = _path(values, "FACE_INSIGHTFACE_ROOT") / "models/buffalo_l"
        reid = _path(values, "REID_CHECKPOINT_DIR")
        pose = _path(values, "REID_POSE_WEIGHTS")
        if pose != Path("/var/lib/sightindex/.cache/yolov8n-pose.pt"):
            raise ModelSetupError("RTX pose weights must use the runtime's fixed cache path")
        dfa = source / DFA_RELATIVE
    else:
        raise ModelSetupError("unknown deployment target")
    result = {"yolo.person": yolo}
    result.update({f"face.{name.removesuffix('.onnx')}": face / name for name in FACE_FILES})
    if "reid" in stacks:
        result.update(
            {
                "reid.checkpoint": reid / "model.pth",
                "reid.config": reid / "model.yaml",
                "reid.pose": pose,
                "reid.dfa": dfa,
            }
        )
    if "embedding" in stacks:
        qwen = _path(values, "QWEN_EMBEDDING_MODEL_DIR")
        result.update(
            {
                "qwen.weights": qwen / "model.safetensors",
                "qwen.config": qwen / "config.json",
                "qwen.tokenizer-config": qwen / "tokenizer_config.json",
                "qwen.tokenizer": qwen / "tokenizer.json",
                "qwen.processor": qwen / "preprocessor_config.json",
                "qwen.runtime": qwen / "scripts/qwen3_vl_embedding.py",
            }
        )
    return result


def _family_model(identifier: str) -> str:
    """Return the explicit model identity expected for a role."""
    if identifier == "yolo.person":
        return "yolo11n"
    if identifier.startswith("face."):
        return "buffalo_l"
    if identifier.startswith("qwen."):
        return QWEN_MODEL
    return "sapiensid_wb12m"


def select_manifest(
    manifest: Manifest, layout: dict[str, Path], values: dict[str, str], source: Path
) -> Manifest:
    """Validate all roles and fixed inference pins before allowing asset writes."""
    by_id = {artifact.id: artifact for artifact in manifest.artifacts}
    allowed = set(layout)
    # Bundles may contain other standard roles for deployments selecting fewer stacks.
    allowed.update({"reid.checkpoint", "reid.config", "reid.pose", "reid.dfa"})
    allowed.update(
        {
            "qwen.weights",
            "qwen.config",
            "qwen.tokenizer-config",
            "qwen.tokenizer",
            "qwen.processor",
            "qwen.runtime",
        }
    )
    for artifact in manifest.artifacts:
        if artifact.id.startswith("qwen.extra."):
            name = artifact.id.removeprefix("qwen.extra.")
            if name not in QWEN_EXTRA_FILES:
                raise ModelSetupError("unsupported optional Qwen file role")
            if "qwen.weights" in layout:
                layout[artifact.id] = layout["qwen.weights"].parent / name
        elif artifact.id not in allowed:
            raise ModelSetupError("unknown model artifact role")
        if artifact.model != _family_model(artifact.id):
            raise ModelSetupError("model identity does not agree with its artifact role")
        if not artifact.license or not artifact.terms_url:
            raise ModelSetupError(
                "every model artifact must record its license and publisher terms URL"
            )
    missing = set(layout) - by_id.keys()
    if missing:
        raise ModelSetupError("model lockfile lacks required roles: " + ", ".join(sorted(missing)))
    selected = tuple(by_id[identifier] for identifier in layout)
    if "qwen.weights" in layout:
        if (
            by_id["qwen.weights"].sha256 != QWEN_WEIGHTS_SHA256
            or by_id["qwen.runtime"].sha256 != QWEN_RUNTIME_SHA256
        ):
            raise ModelSetupError(
                "Qwen weights/runtime do not match the shipped embedding image pins"
            )
        revisions = {artifact.revision for artifact in selected if artifact.id.startswith("qwen.")}
        if len(revisions) != 1:
            raise ModelSetupError("Qwen bundle files must use one reviewed revision")
    if "reid.checkpoint" in layout:
        config = source / ALIGNER_CONFIG
        if config.is_symlink() or not config.is_file():
            raise ModelSetupError("trusted source lacks the ReID aligner configuration")
        config_hash = hashlib.sha256(config.read_bytes()).hexdigest()
        roles = (
            ("model.pth", "reid.checkpoint"),
            ("model.yaml", "reid.config"),
            ("yolo_dfa.yaml", None),
            ("mobilenetv4_Final.pth", "reid.dfa"),
            ("yolov8n-pose.pt", "reid.pose"),
        )
        composite = "\n".join(
            f"{name}:{by_id[identifier].sha256 if identifier else config_hash}"
            for name, identifier in roles
        )
        revision = "sha256:" + hashlib.sha256(composite.encode("ascii")).hexdigest()
        if values.get("REID_CHECKPOINT_REVISION") != revision:
            raise ModelSetupError(
                "ReID pipeline fingerprint differs from configuration; "
                "review model/index migration rather than changing it automatically"
            )
    return Manifest(version=manifest.version, artifacts=selected)


def main(argv: list[str] | None = None) -> int:
    """Check, prepare, diagnose or recover reviewed assets without deploying services."""
    parser = SafeParser(description=__doc__)
    parser.add_argument("--target", choices=("containers", "rtx5090"), default="containers")
    parser.add_argument("--source", type=Path, default=REPOSITORY)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--stacks", default="base")
    parser.add_argument("--manifest", type=Path, required=True)
    operations = parser.add_mutually_exclusive_group()
    operations.add_argument("--check", action="store_true")
    operations.add_argument("--diagnose-models", action="store_true")
    operations.add_argument("--recover-models", action="store_true")
    preparation = parser.add_mutually_exclusive_group()
    preparation.add_argument("--source-dir", type=Path)
    preparation.add_argument("--download", action="store_true")
    parser.add_argument("--acknowledge-model-terms", action="store_true")
    parser.add_argument("--confirm-no-active-preparation", action="store_true")
    parser.add_argument("--quarantine-partial", action="append", default=[])
    try:
        args = parser.parse_args(argv)
        maintenance = args.diagnose_models or args.recover_models
        if (args.check or maintenance) and (args.download or args.source_dir):
            raise ModelSetupError("read-only checks or recovery cannot prepare/download assets")
        if maintenance and args.acknowledge_model_terms:
            raise ModelSetupError("model maintenance cannot be combined with preparation options")
        if not args.recover_models and (
            args.confirm_no_active_preparation or args.quarantine_partial
        ):
            raise ModelSetupError(
                "recovery confirmation and partial roles require --recover-models"
            )
        if args.recover_models and not args.confirm_no_active_preparation:
            raise ModelSetupError(
                "stop all preparation processes, then explicitly confirm "
                "--confirm-no-active-preparation; active kernel locks cannot be overridden"
            )
        values = read_env(args.env_file)
        layout = model_layout(values, args.target, args.source, frozenset(args.stacks.split()))
        manifest = select_manifest(load_manifest(args.manifest), layout, values, args.source)
        if not args.check and not maintenance and not args.acknowledge_model_terms:
            raise ModelSetupError(
                "model preparation requires --acknowledge-model-terms; "
                "this is not a commercial license grant"
            )
        destination = (
            _path(values, "MODEL_DIR")
            if args.target == "containers"
            else Path("/var/lib/sightindex/models")
        )

        def resolve(artifact: Artifact) -> Path:
            return layout[artifact.id]

        if args.check:
            report = check_assets(manifest, destination, destination_for=resolve)
        elif args.diagnose_models:
            report = diagnose_assets(manifest, destination, destination_for=resolve)
        elif args.recover_models:
            report = recover_assets(
                manifest,
                destination,
                destination_for=resolve,
                confirm_no_active_preparation=args.confirm_no_active_preparation,
                quarantine_partial_ids=frozenset(args.quarantine_partial),
            )
        else:
            report = prepare_assets(
                manifest,
                destination,
                destination_for=resolve,
                source_dir=args.source_dir,
                download=args.download,
            )
        output: dict[str, Any] = report.as_dict()
        output["vlm"] = "external endpoint only; no VLM weights or service were installed"
        output["license_notice"] = "acknowledgement does not grant commercial model rights"
        print(json.dumps(output, ensure_ascii=False))
        return 0 if report.ok else 1
    except (ModelSetupError, PreflightError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
    except Exception as error:
        print(
            f"ERROR: model setup failed ({type(error).__name__}); no unverified model was promoted",
            file=sys.stderr,
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
