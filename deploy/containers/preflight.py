#!/usr/bin/env python3
"""Read-only deployment checks; never source configuration or download weights."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit

KNOWN_STACKS = frozenset({"base", "reid", "embedding", "semantic"})
PUBLIC_VALUES = (
    "SIGHTINDEX_IMAGE",
    "MODEL_DIR",
    "MEDIA_DIR",
    "CACHE_DIR",
    "QWEN_EMBEDDING_IMAGE",
    "COMPOSE_PROJECT_NAME",
)
DFA_RELATIVE = Path(
    "deploy/agx/reid_service/sapiensid/tasks/sapiensID/src/aligners/"
    "keypoint_predictor/pretrained_models/aligners/"
    "dfa_mobilenetv4_medium/mobilenetv4_Final.pth"
)


class PreflightError(ValueError):
    """Invalid configuration, with a diagnostic that does not reveal values."""


def read_env(path: Path) -> dict[str, str]:
    """Parse simple dotenv assignments as data, not executable shell input."""
    if path.is_symlink() or not path.is_file():
        raise PreflightError("env file must be an existing regular, non-symlink file")
    if path.stat().st_mode & 0o027:
        raise PreflightError("env file must not be accessible to other users (use 0600/0640)")
    values: dict[str, str] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, raw = line.partition("=")
        if not separator or not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
            raise PreflightError(f"env line {number}: expected KEY=value")
        if key in values or key in {
            "BASHOPTS",
            "BASH_ENV",
            "CDPATH",
            "ENV",
            "GLOBIGNORE",
            "HOME",
            "IFS",
            "LOGNAME",
            "PATH",
            "SHELL",
            "SHELLOPTS",
            "USER",
            "CODEX_HOME",
            "PYTHONPATH",
            "LD_PRELOAD",
        }:
            raise PreflightError(f"env line {number}: duplicate or reserved key {key}")
        if any(character in raw for character in ("$", "`", "\x00", ";", "|", "&", "<", ">")):
            raise PreflightError(f"env line {number}: substitutions are forbidden for {key}")
        try:
            parts = shlex.split(raw, comments=True, posix=True)
        except ValueError as error:
            raise PreflightError(f"env line {number}: invalid quoting for {key}") from error
        if len(parts) > 1:
            raise PreflightError(f"env line {number}: quote spaces in {key}")
        values[key] = parts[0] if parts else ""
    return values


def enabled(values: dict[str, str], key: str, default: bool = False) -> bool:
    """Read a boolean without silently interpreting misspelled values."""
    value = values.get(key, "true" if default else "false").lower()
    if value not in {"true", "false", "1", "0"}:
        raise PreflightError(f"{key} must be true or false")
    return value in {"true", "1"}


def check_configuration(
    target: str, source: Path, values: dict[str, str], stacks: set[str]
) -> tuple[list[str], list[str]]:
    """Return all configuration/asset errors and the explicitly selected capabilities."""
    errors: list[str] = []
    capabilities = ["API/console, upload detection, isolated playback acceptance"]

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    if target == "containers" and values.get("COMPOSE_PROJECT_NAME"):
        require(
            re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,62}", values["COMPOSE_PROJECT_NAME"]) is not None,
            "COMPOSE_PROJECT_NAME must use lowercase letters, digits, _ or - (max 63 characters)",
        )

    def asset(path: Path, label: str) -> None:
        require(
            path.is_file() and path.stat().st_size > 0 and os.access(path, os.R_OK),
            f"missing or unreadable model asset: {label} ({path})",
        )

    def absolute(key: str, fallback: str = "", *, directory: bool = True) -> Path:
        value = values.get(key, fallback)
        require(bool(value) and Path(value).is_absolute(), f"{key} must be an absolute path")
        path = Path(value) if value else Path("/missing-deployment-asset")
        if directory:
            require(not path.exists() or path.is_dir(), f"{key} must refer to a directory")
            broad_paths = {
                Path("/"),
                Path.home(),
                Path("/tmp"),
                Path("/private/tmp"),
                Path("/data"),
                Path("/var"),
                Path("/private/var"),
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
            require(
                path not in broad_paths
                and path.resolve() not in {candidate.resolve() for candidate in broad_paths},
                f"{key} must be a dedicated directory, not a broad host path",
            )
        return path

    require("base" in stacks, "stacks must include base")
    require(not stacks - KNOWN_STACKS, "unknown stack; use base/reid/embedding/semantic")
    prepare_video = enabled(values, "VIDEO_PREPARATION_ENABLED")
    enabled(values, "VIDEO_PREPARATION_INCLUDE_AUDIO")
    if prepare_video:
        for key, fallback in (
            ("VIDEO_FFMPEG_BINARY", "ffmpeg"),
            ("VIDEO_FFPROBE_BINARY", "ffprobe"),
        ):
            binary = values.get(key, fallback)
            require(
                bool(binary) and not any(c in binary for c in "\x00\r\n"),
                f"configure a valid {key}",
            )
        capabilities.append("bounded CPU video preparation: new uploads only; history unchanged")
    require((source / "main.py").is_file(), "source must be a SightIndex checkout")
    require((source / "deploy/containers/verify.py").is_file(), "source lacks deployment verifier")
    for key in ("APP_BASIC_AUTH_USERNAME", "APP_BASIC_AUTH_PASSWORD", "REID_SERVICE_API_KEY"):
        require(bool(values.get(key)), f"configure {key} before deployment")
    for key in ("APP_BASIC_AUTH_PASSWORD", "MINIO_ROOT_PASSWORD", "REID_SERVICE_API_KEY"):
        require(
            values.get(key, "") not in {"replace-with-random-secret", "change-me"},
            f"replace placeholder {key}",
        )
    if target == "containers":
        for key in ("POSTGRES_PASSWORD", "MINIO_ROOT_PASSWORD", "REID_CHECKPOINT_REVISION"):
            require(bool(values.get(key)), f"configure {key} before deployment")
        require(
            bool(re.fullmatch(r"[A-Za-z0-9_.-]+", values.get("POSTGRES_PASSWORD", ""))),
            "POSTGRES_PASSWORD must be URL-safe (letters, digits, _, . or -)",
        )
        models = absolute("MODEL_DIR")
        absolute("MEDIA_DIR")
        absolute("CACHE_DIR")
        asset(models / "yolo11n.pt", "YOLO")
        require(
            enabled(values, "REID_ENABLED") == ("reid" in stacks),
            "REID_ENABLED must agree with selected reid stack",
        )
        require(
            "semantic" not in stacks or "embedding" in stacks, "semantic requires embedding stack"
        )
        require(
            enabled(values, "SEMANTIC_SEARCH_ENABLED") == ("semantic" in stacks),
            "SEMANTIC_SEARCH_ENABLED must agree with selected semantic stack",
        )
        face_root = models / "insightface"
        checkpoint = models / "sapiensid_wb12m"
        pose = models / "yolov8n-pose.pt"
        dfa = models / "dfa_mobilenetv4_medium/mobilenetv4_Final.pth"
    else:
        require(
            source.resolve() == Path("/opt/sightindex"),
            "RTX systemd checkout must be /opt/sightindex",
        )
        require(enabled(values, "REID_ENABLED"), "RTX profile requires REID_ENABLED=true")
        require(enabled(values, "MILVUS_ENABLED"), "RTX profile requires MILVUS_ENABLED=true")
        require(values.get("APP_HOST") == "127.0.0.1", "RTX APP_HOST must be 127.0.0.1")
        public = urlsplit(values.get("PUBLIC_BASE_URL", ""))
        require(
            public.scheme == "https"
            and bool(public.hostname)
            and public.hostname != "sightindex.example.com",
            "RTX PUBLIC_BASE_URL must be the real HTTPS gateway URL",
        )
        asset(absolute("YOLO_MODEL", directory=False), "YOLO")
        face_root = absolute("FACE_INSIGHTFACE_ROOT")
        checkpoint = absolute("REID_CHECKPOINT_DIR")
        pose = absolute("REID_POSE_WEIGHTS", directory=False)
        dfa = source / DFA_RELATIVE
    if "reid" in stacks:
        asset(checkpoint / "model.pth", "SapiensID checkpoint")
        asset(checkpoint / "model.yaml", "SapiensID config")
        asset(pose, "SapiensID pose")
        asset(dfa, "SapiensID DFA aligner")
        require(
            bool(re.fullmatch(r"sha256:[0-9a-f]{64}", values.get("REID_CHECKPOINT_REVISION", ""))),
            "REID_CHECKPOINT_REVISION must be a complete sha256 pipeline fingerprint",
        )
        capabilities.append("body ReID + Milvus + face-priority readiness")
    require(
        not enabled(values, "FACE_INSIGHTFACE_ALLOW_DOWNLOAD"),
        "automatic face-model downloads must remain disabled",
    )
    if (
        target == "containers"
        or ("reid" in stacks and enabled(values, "REID_FACE_PRIORITY_ENABLED", True))
        or enabled(values, "FACE_RECOGNITION_ON_INGEST")
    ):
        face_name = values.get("FACE_INSIGHTFACE_MODEL", "buffalo_l")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", face_name):
            errors.append("FACE_INSIGHTFACE_MODEL must be a single model directory name")
        else:
            face = face_root / "models" / face_name
            asset(face / "det_10g.onnx", "InsightFace detector")
            asset(face / "w600k_r50.onnx", "InsightFace recognition")
    if "embedding" in stacks:
        require(bool(values.get("QWEN_EMBEDDING_API_KEY")), "configure QWEN_EMBEDDING_API_KEY")
        require(bool(values.get("QWEN_EMBEDDING_IMAGE")), "configure prebuilt QWEN_EMBEDDING_IMAGE")
        directory = absolute("QWEN_EMBEDDING_MODEL_DIR")

        def qwen_asset(path: Path, label: str) -> bool:
            if not path.resolve().is_relative_to(directory.resolve()):
                errors.append(f"Qwen {label} resolves outside its model directory")
                return False
            asset(path, f"Qwen {label}")
            return path.is_file() and path.stat().st_size > 0

        for name in ("config.json", "tokenizer_config.json", "preprocessor_config.json"):
            qwen_asset(directory / name, name)
        # The shipped embedding image pins a monolithic checkpoint and an
        # upstream inference script. A generic shard manifest cannot replace them.
        qwen_asset(directory / "model.safetensors", "weights")
        qwen_asset(directory / "scripts/qwen3_vl_embedding.py", "inference script")
        tokenizer = directory / "tokenizer.json"
        if not tokenizer.is_file():
            tokenizer = directory / "tokenizer.model"
        qwen_asset(tokenizer, "tokenizer data")
        index = directory / "model.safetensors.index.json"
        if index.is_file():
            try:
                if not qwen_asset(index, "weight manifest"):
                    raise ValueError("invalid manifest file")
                manifest = json.loads(index.read_text(encoding="utf-8"))
                weights = manifest.get("weight_map") if isinstance(manifest, dict) else None
                if not isinstance(weights, dict) or not all(
                    isinstance(name, str) for name in weights.values()
                ):
                    raise ValueError("invalid manifest")
                filenames = set(weights.values())
                require(bool(filenames), "Qwen weight manifest is empty")
                for name in filenames:
                    if (
                        not isinstance(name, str)
                        or name in {"", ".", ".."}
                        or Path(name).name != name
                    ):
                        errors.append("Qwen manifest has an unsafe shard filename")
                    else:
                        qwen_asset(directory / name, "weight shard")
            except (ValueError, KeyError, TypeError, OSError):
                errors.append("Qwen weight manifest is invalid")
        capabilities.append("Qwen visual embedding (prebuilt image, offline weights)")
    if "semantic" in stacks:
        capabilities.append("semantic retrieval; historical coverage checked separately")
    calibration = ("REID_CROSS_CAMERA_CALIBRATION_COEF", "REID_CROSS_CAMERA_CALIBRATION_INTERCEPT")
    require(
        all(key in values for key in calibration) or all(key not in values for key in calibration),
        "calibration requires both coefficient and intercept, or neither",
    )
    for key in calibration:
        if key in values:
            try:
                require(
                    math.isfinite(float(values[key])), f"{key} must be a finite, non-empty number"
                )
            except ValueError:
                errors.append(f"{key} must be a finite, non-empty number")
    provider = values.get("VLM_PROVIDER", "none")
    require(
        provider in {"none", "openai_compatible"}, "VLM_PROVIDER must be none or openai_compatible"
    )
    if provider == "none":
        capabilities.append("VLM disabled (not included in deployment success)")
        # RTX ships a dormant background flag; it is inert until a provider exists.
        require(
            not enabled(values, "VLM_STRUCTURED_ON_INGEST")
            and not enabled(values, "VLM_CAPTION_ON_INDEX"),
            "VLM ingest/caption requires a configured provider",
        )
    elif provider == "openai_compatible":
        endpoint = urlsplit(values.get("VLM_BASE_URL", ""))
        require(
            endpoint.scheme in {"http", "https"}
            and bool(endpoint.hostname)
            and not endpoint.username,
            "configure a valid VLM_BASE_URL without inline credentials",
        )
        require(
            bool(values.get("VLM_MODEL")) and values.get("VLM_MODEL") != "your-vlm-model",
            "configure VLM_MODEL",
        )
        capabilities.append("external VLM configured; reachability/label accuracy not tested")
    return errors, capabilities


def check_tools(target: str, stacks: set[str], values: dict[str, str] | None = None) -> list[str]:
    """Inspect installed tools and device status without starting any services."""
    errors: list[str] = []

    def run(arguments: list[str]) -> str | None:
        try:
            result = subprocess.run(
                arguments, check=False, capture_output=True, text=True, timeout=15
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
        errors.append(f"read-only tool check failed: {arguments[0]}")
        return None

    if sys.version_info < (3, 11):  # noqa: UP036 - this host script can be launched by older Python
        errors.append("deployment Python must be 3.11 or newer")
    errors.extend(
        f"required tool missing: {tool}"
        for tool in ("docker", "git", "tar")
        if not shutil.which(tool)
    )
    if shutil.which("docker"):
        run(["docker", "compose", "version"])
        operating_system = run(["docker", "info", "--format", "{{.OSType}}"])
        if operating_system and operating_system != "linux":
            errors.append("Docker must use Linux containers")
    if target == "rtx5090":
        configured = values or {}
        if enabled(configured, "VIDEO_PREPARATION_ENABLED"):
            for key, fallback in (
                ("VIDEO_FFMPEG_BINARY", "ffmpeg"),
                ("VIDEO_FFPROBE_BINARY", "ffprobe"),
            ):
                if not shutil.which(configured.get(key, fallback)):
                    errors.append(f"required video preparation tool missing: {fallback}")
        if platform.system() != "Linux" or platform.machine() not in {"x86_64", "amd64"}:
            errors.append("RTX systemd profile requires Linux x86_64")
        errors.extend(
            f"required RTX tool missing: {tool}"
            for tool in ("systemctl", "runuser", "node", "npm", "curl")
            if not shutil.which(tool)
        )
        if shutil.which("node"):
            version = run(["node", "--version"])
            if version and tuple(int(part) for part in version.lstrip("v").split(".")[:2]) < (
                22,
                18,
            ):
                errors.append("RTX requires Node.js 22.18 or newer")
    if {"reid", "embedding"} & stacks:
        output = run(
            ["nvidia-smi", "--query-gpu=name,memory.free", "--format=csv,noheader,nounits"]
        )
        if output:
            device, separator, memory = output.splitlines()[0].rpartition(",")
            if not separator or not memory.strip().isdigit() or int(memory.strip()) < 4096:
                errors.append(
                    "GPU 0 requires at least 4096 MiB free; "
                    "combined-model peak must still be measured"
                )
            if target == "rtx5090" and "5090" not in device:
                errors.append("RTX profile requires an RTX 5090 on GPU 0")
    return errors


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; all checks are non-mutating and diagnostics omit secrets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("containers", "rtx5090"), default="containers")
    parser.add_argument("--source", type=Path, default=Path.cwd())
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--stacks", default="base")
    parser.add_argument("--check-tools", action="store_true")
    parser.add_argument("--value", choices=PUBLIC_VALUES, help=argparse.SUPPRESS)
    arguments = parser.parse_args(argv)
    try:
        values = read_env(arguments.env_file)
        if arguments.value:
            print(values.get(arguments.value, ""))
            return 0
        stacks = set(arguments.stacks.split())
        errors, capabilities = check_configuration(
            arguments.target, arguments.source, values, stacks
        )
        if arguments.check_tools:
            errors.extend(check_tools(arguments.target, stacks, values))
    except PreflightError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    except (OSError, ValueError):
        # Do not echo untrusted exceptions: they may include configuration values.
        print(
            "ERROR: invalid/unreadable deployment configuration; "
            "use simple private KEY=value assignments",
            file=sys.stderr,
        )
        return 1
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    if errors:
        return 1
    print("Read-only preflight passed. Selected capabilities:")
    for capability in capabilities:
        print(f"  - {capability}")
    print("No containers, models, directories or database records were changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
