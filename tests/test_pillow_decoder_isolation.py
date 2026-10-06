"""Uploads must never execute third-party image wrappers or their plugin installers."""

from __future__ import annotations

import base64
import importlib.util
import io
import sys
from collections.abc import Callable
from pathlib import Path
from types import FunctionType, ModuleType
from typing import Any

import pytest
from fastapi import HTTPException, UploadFile
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = {
    "storage": ROOT / "app/services/storage.py",
    "embedding": ROOT / "deploy/containers/embedding_app.py",
}


def _native_opener() -> Callable[..., Image.Image]:
    """Retrieve the installed decoder without calling the real Ultralytics wrapper."""
    opener = Image.open
    if opener.__module__ == "ultralytics.utils.patches":
        opener = opener.__globals__["_image_open"]
    assert opener.__module__ == "PIL.Image"
    assert opener.__globals__ is vars(Image)
    return opener


def _forbidden_wrapper(*_args: Any, **_kwargs: Any) -> Any:
    raise AssertionError("Third-party wrapper/plugin installation must never run")


def _wrapper(native: Callable[..., Image.Image] | None = None) -> FunctionType:
    """Match the installed wrapper's original-function storage, without any installer."""
    namespace: dict[str, Any] = {"__name__": "ultralytics.utils.patches"}
    if native is not None:
        namespace["_image_open"] = native
    return FunctionType(_forbidden_wrapper.__code__, namespace, "image_open")


def _load_module(target: str, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Import each entrypoint directly so both import orders can be tested independently."""
    name = f"pillow_isolation_{target}"
    specification = importlib.util.spec_from_file_location(name, SOURCE_PATHS[target])
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    monkeypatch.setitem(sys.modules, name, module)
    specification.loader.exec_module(module)
    return module


def _image_bytes(image_format: str | None) -> bytes:
    if image_format is None:
        return b"invalid synthetic image"
    with Image.new("RGB", (32, 64), (96, 80, 70)) as image:
        buffer = io.BytesIO()
        image.save(buffer, format=image_format)
        return buffer.getvalue()


def _decode(
    target: str, module: ModuleType, data: bytes, tmp_path: Path, **limits: int
) -> str | Image.Image:
    if target == "storage":
        settings = module.Settings(_env_file=None, data_dir=tmp_path, **limits)
        return module.StorageService(settings).save_upload(
            UploadFile(io.BytesIO(data), filename="synthetic.png")
        )
    settings = module.EmbeddingSettings(_env_file=None, **limits)
    return module.decode_image(base64.b64encode(data).decode("ascii"), settings)


def _assert_error(
    target: str,
    module: ModuleType,
    data: bytes,
    tmp_path: Path,
    status: int,
    **limits: int,
) -> None:
    error_type = module.InvalidUploadError if target == "storage" else HTTPException
    with pytest.raises(error_type) as raised:
        _decode(target, module, data, tmp_path, **limits)
    assert raised.value.status_code == status
    if target == "storage":
        assert list((tmp_path / "uploads").iterdir()) == []


@pytest.mark.parametrize("target", SOURCE_PATHS)
@pytest.mark.parametrize("patch_before_import", [False, True])
@pytest.mark.parametrize("image_format", ["JPEG", "PNG", "WEBP", "BMP", "GIF", "TIFF", None])
def test_decoders_bypass_installer_in_either_import_order_and_keep_format_allowlist(
    target: str,
    patch_before_import: bool,
    image_format: str | None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    native = _native_opener()
    data = _image_bytes(image_format)
    wrapper = _wrapper(native)
    if patch_before_import:
        monkeypatch.setattr(Image, "open", wrapper)
    module = _load_module(target, monkeypatch)
    if not patch_before_import:
        monkeypatch.setattr(Image, "open", wrapper)
    if image_format in {"TIFF", None}:
        _assert_error(target, module, data, tmp_path, 400)
    else:
        result = _decode(target, module, data, tmp_path)
        if target == "storage":
            assert isinstance(result, str) and result.endswith(".png")
            with native(tmp_path / result.removeprefix("/data/")) as image:
                image.load()
                assert image.size == (32, 64)
                assert image.info == {}
        else:
            assert isinstance(result, Image.Image)
            with result:
                assert result.mode == "RGB" and result.size == (32, 64)
    assert Image.open is wrapper


@pytest.mark.parametrize("target", SOURCE_PATHS)
@pytest.mark.parametrize("limit", ["bytes", "pixels"])
def test_bypassing_installer_preserves_byte_and_pixel_limits(
    target: str,
    limit: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    native = _native_opener()
    data = _image_bytes("PNG")
    wrapper = _wrapper(native)
    monkeypatch.setattr(Image, "open", wrapper)
    module = _load_module(target, monkeypatch)
    limits = (
        {"upload_image_max_bytes": len(data) - 1}
        if limit == "bytes"
        else {"upload_image_max_pixels": 32 * 64 - 1}
    )
    _assert_error(target, module, data, tmp_path, 413, **limits)
    assert Image.open is wrapper


@pytest.mark.parametrize("target", SOURCE_PATHS)
@pytest.mark.parametrize("untrusted", ["unknown_wrapper", "missing_native", "spoofed_native"])
def test_decoders_reject_untrusted_wrappers_without_calling_them(
    target: str,
    untrusted: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    data = _image_bytes("PNG")
    wrapper = _wrapper()
    if untrusted == "unknown_wrapper":
        wrapper.__module__ = "another_image_plugin"
    elif untrusted == "spoofed_native":
        pretender = _wrapper()
        pretender.__module__ = "PIL.Image"
        pretender.__name__ = "open"
        wrapper.__globals__["_image_open"] = pretender
    monkeypatch.setattr(Image, "open", wrapper)
    module = _load_module(target, monkeypatch)
    _assert_error(target, module, data, tmp_path, 400)
    assert Image.open is wrapper
