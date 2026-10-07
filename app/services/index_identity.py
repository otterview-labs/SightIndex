"""Pure vector-space identities shared by runtime indexing and deployment checks.

This module performs no network, database, filesystem, or model initialization.
Keep existing byte formulas stable: changing these names would detach SQL markers
from their Milvus collections and must be handled as a separate index migration.
"""

from __future__ import annotations

import hashlib
from collections.abc import Collection, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.config.settings import Settings

COLLECTION_SUFFIXES = MappingProxyType({
    "image": "vl_images",
    "person_crop": "vl_person_crops",
    "face_embedding": "face_embeddings",
    "reid_person_crop": "reid_person_crops",
})
REID_OBJECT_TYPES = frozenset({"reid_person_crop"})

QWEN_MODEL_ID = "Qwen/Qwen3-VL-Embedding-2B"
QWEN_EMBEDDING_DIM = 2048
QWEN_WEIGHTS_SHA256 = "c73fa9caeddeb3ff831d46c085a7a5708343248ca777e90f2d486964464509c1"
QWEN_RUNTIME_SHA256 = "8ffa74a1a6bb759610c57865ea416fd4daf9936cb787520e1112a3e1d547f36a"
QWEN_MAX_PIXELS = 768 * 32 * 32
QWEN_MAX_LENGTH = 4096


def milvus_namespace_identity(settings: Settings) -> str:
    """Derive the existing logical namespace solely from explicit settings."""
    configured = (settings.milvus_namespace_id or "").strip()
    if configured:
        return configured
    material = "\0".join(
        [
            settings.milvus_host.strip().lower(),
            str(settings.milvus_port),
            settings.milvus_db.strip(),
        ]
    )
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]
    return f"endpoint-{digest}"


def reid_space_digest(settings: Settings, *, namespace: str | None = None) -> str:
    """Compute the unchanged ReID collection suffix without constructing clients."""
    material = "\0".join(
        [
            settings.reid_model,
            settings.reid_checkpoint_revision,
            str(settings.reid_embedding_dim),
            settings.reid_preprocess_version,
            settings.milvus_metric_type.upper(),
            milvus_namespace_identity(settings) if namespace is None else namespace,
        ]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def milvus_collection_name(
    settings: Settings,
    object_type: str,
    *,
    suffixes: Mapping[str, str] = COLLECTION_SUFFIXES,
    reid_types: Collection[str] = REID_OBJECT_TYPES,
    namespace: str | None = None,
    space_digest: str | None = None,
) -> str:
    """Preserve visual overrides and ReID names using only configuration."""
    suffix = suffixes[object_type]
    configured_prefix = settings.milvus_collection_prefix
    if object_type in {"image", "person_crop"} and settings.milvus_visual_collection_prefix:
        configured_prefix = settings.milvus_visual_collection_prefix
    prefix = configured_prefix.strip("_")
    name = f"{prefix}_{suffix}" if prefix else suffix
    if object_type in reid_types:
        digest = (
            reid_space_digest(settings, namespace=namespace)
            if space_digest is None else space_digest
        )
        name = f"{name}_{digest}"
    return name


def reid_index_fingerprint(
    settings: Settings, *, namespace: str | None = None, collection: str | None = None
) -> str:
    """Name the existing SQL marker space with no database/index/model I/O."""
    if namespace is None:
        namespace = milvus_namespace_identity(settings)
    if collection is None:
        collection = milvus_collection_name(settings, "reid_person_crop", namespace=namespace)
    return "|".join(
        [
            settings.reid_model,
            settings.reid_checkpoint_revision,
            str(settings.reid_embedding_dim),
            settings.reid_preprocess_version,
            namespace,
            collection,
        ]
    )
