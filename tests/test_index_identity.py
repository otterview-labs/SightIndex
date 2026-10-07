"""Freeze existing vector identities and keep acceptance computation free of I/O."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest

from app.services import index_identity as identity

if TYPE_CHECKING:
    from app.config.settings import Settings


def _settings(**overrides: Any) -> Settings:
    """Use only synthetic scalar fields, with no Settings/environment construction."""
    values = {
        "milvus_namespace_id": None,
        "milvus_host": "LocalHost",
        "milvus_port": 19530,
        "milvus_db": "default",
        "milvus_collection_prefix": "sightindex",
        "milvus_visual_collection_prefix": "different_visual_prefix",
        "milvus_metric_type": "COSINE",
        "reid_model": "fixture_model",
        "reid_checkpoint_revision": "fixture_revision",
        "reid_embedding_dim": 32,
        "reid_preprocess_version": "fixture_preprocess",
    }
    values.update(overrides)
    return cast("Settings", SimpleNamespace(**values))


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"milvus_namespace_id": "  explicit-stable-namespace  "},
        {"milvus_host": " HOST.Example  ", "milvus_db": "  database  "},
        {"milvus_collection_prefix": "___"},
        {"milvus_collection_prefix": "__custom__", "milvus_metric_type": " cosine "},
    ],
)
def test_reid_identity_is_byte_identical_to_previous_formula(overrides: dict[str, Any]) -> None:
    settings = _settings(**overrides)
    # Independent pre-refactor formula guards separator/case/whitespace behavior.
    namespace = (settings.milvus_namespace_id or "").strip()
    if not namespace:
        material = "\0".join(
            [
                settings.milvus_host.strip().lower(),
                str(settings.milvus_port),
                settings.milvus_db.strip(),
            ]
        )
        namespace = "endpoint-" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]
    space = "\0".join(
        [
            settings.reid_model,
            settings.reid_checkpoint_revision,
            str(settings.reid_embedding_dim),
            settings.reid_preprocess_version,
            settings.milvus_metric_type.upper(),
            namespace,
        ]
    )
    digest = hashlib.sha256(space.encode("utf-8")).hexdigest()[:16]
    prefix = settings.milvus_collection_prefix.strip("_")
    collection = (f"{prefix}_" if prefix else "") + "reid_person_crops_" + digest
    fingerprint = "|".join(
        [
            settings.reid_model, settings.reid_checkpoint_revision,
            str(settings.reid_embedding_dim), settings.reid_preprocess_version,
            namespace, collection,
        ]
    )
    assert identity.milvus_namespace_identity(settings) == namespace
    assert identity.reid_space_digest(settings) == digest
    assert identity.milvus_collection_name(settings, "reid_person_crop") == collection
    assert identity.reid_index_fingerprint(settings) == fingerprint


def test_visual_prefix_does_not_retarget_face_or_reid_collections() -> None:
    settings = _settings()
    assert identity.milvus_collection_name(settings, "image") == "different_visual_prefix_vl_images"
    assert identity.milvus_collection_name(settings, "person_crop") == (
        "different_visual_prefix_vl_person_crops"
    )
    assert identity.milvus_collection_name(settings, "face_embedding") == (
        "sightindex_face_embeddings"
    )
    assert identity.milvus_collection_name(settings, "reid_person_crop").startswith(
        "sightindex_reid_person_crops_"
    )


def test_runtime_delegates_to_same_identity_without_initializing_clients() -> None:
    from app.services.reid_index import ReidIndexService
    from app.services.vector_index import MilvusVectorIndex

    settings = _settings(milvus_namespace_id="synthetic-logical-id")
    index = object.__new__(MilvusVectorIndex)
    index.settings = settings
    service = object.__new__(ReidIndexService)
    service.settings = settings
    service.index = index
    assert index.namespace_identity == identity.milvus_namespace_identity(settings)
    assert index.reid_space_digest() == identity.reid_space_digest(settings)
    assert service.fingerprint == identity.reid_index_fingerprint(settings)
    for object_type in identity.COLLECTION_SUFFIXES:
        assert index._collection_name(object_type) == identity.milvus_collection_name(
            settings, object_type
        )


def test_runtime_collection_suffix_and_digest_overrides_are_preserved() -> None:
    from app.services.vector_index import MilvusVectorIndex

    class CustomIndex(MilvusVectorIndex):
        collection_suffixes = {"reid_person_crop": "custom_person_crops"}

        def reid_space_digest(self) -> str:
            return "custom_digest"

    index = object.__new__(CustomIndex)
    index.settings = _settings()
    assert index._collection_name("reid_person_crop") == (
        "sightindex_custom_person_crops_custom_digest"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("reid_model", "other-model"),
        ("reid_checkpoint_revision", "other-revision"),
        ("reid_embedding_dim", 64),
        ("reid_preprocess_version", "other-preprocess"),
        ("milvus_namespace_id", "other-namespace"),
        ("milvus_collection_prefix", "other-prefix"),
        ("milvus_metric_type", "IP"),
    ],
)
def test_each_existing_space_axis_still_changes_fingerprint(field: str, value: Any) -> None:
    assert identity.reid_index_fingerprint(_settings(**{field: value})) != (
        identity.reid_index_fingerprint(_settings())
    )


def test_identity_module_imports_without_database_network_or_model_libraries() -> None:
    module = Path(identity.__file__).resolve()
    script = """
import importlib.util
import sys
from types import SimpleNamespace
spec = importlib.util.spec_from_file_location('isolated_identity', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert not {'sqlalchemy', 'torch', 'pymilvus', 'app.config.settings'} & sys.modules.keys()
settings = SimpleNamespace(
    milvus_namespace_id='fixture-id', milvus_collection_prefix='fixture',
    milvus_visual_collection_prefix=None, milvus_metric_type='COSINE',
    reid_model='fixture-model', reid_checkpoint_revision='fixture-revision',
    reid_embedding_dim=32, reid_preprocess_version='fixture-preprocess',
)
assert module.reid_index_fingerprint(settings).startswith('fixture-model|fixture-revision|32|')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(module)],
        capture_output=True, text=True, check=False, timeout=10,
    )
    assert result.returncode == 0, result.stderr


def test_qwen_reviewed_pins_match_existing_model_preparation_constants() -> None:
    from deploy.containers import download_embedding_model as downloader
    from deploy.models import setup

    assert identity.QWEN_MODEL_ID == downloader.MODEL_ID == setup.QWEN_MODEL
    assert identity.QWEN_WEIGHTS_SHA256 == downloader.WEIGHTS_SHA256 == setup.QWEN_WEIGHTS_SHA256
    assert identity.QWEN_RUNTIME_SHA256 == setup.QWEN_RUNTIME_SHA256
    assert identity.QWEN_MAX_PIXELS == 768 * 32 * 32
    assert identity.QWEN_MAX_LENGTH == 4096
    assert identity.QWEN_EMBEDDING_DIM == 2048
