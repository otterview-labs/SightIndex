import hashlib
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import monotonic

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config.settings import Settings
from app.models.media import Image, PersonCrop
from app.models.vectors import VLEmbedding
from app.services.embeddings import (
    EmbeddingRuntimeError,
    TextEmbeddingService,
    VisualEmbeddingService,
)
from app.services.observation_index import ObservationIndexService
from app.services.vlm import VLMCaptionService, VLMRuntimeError


class VectorIndexError(RuntimeError):
    pass


_MILVUS_CONNECTED_KEYS: set[tuple[str, str, str, str, str, str]] = set()
_MILVUS_COLLECTION_CACHE: dict[
    tuple[tuple[str, str, str, str, str, str], str, int, str], object
] = {}
_MILVUS_FAILURE_UNTIL: dict[tuple[str, str, str, str, str, str], float] = {}
_MILVUS_STATE_LOCK = threading.RLock()
_MILVUS_CONNECTION_LOCKS: dict[tuple[object, ...], threading.Lock] = {}
_MILVUS_COLLECTION_LOCKS: dict[tuple[object, ...], threading.Lock] = {}
_TEXT_QUERY_VECTOR_CACHE: dict[tuple[str, ...], list[float]] = {}
_TEXT_QUERY_VECTOR_CACHE_MAX = 256


@dataclass(frozen=True)
class VectorSearchHit:
    object_id: uuid.UUID
    score: float


def _hnsw_ef(top_k: int) -> int:
    """Search breadth for HNSW, which rejects any ef below the requested k.

    A fixed ef is a latent failure: small collections are scanned exhaustively and never complain,
    then the index kicks in at scale and every deep search errors out. Measured here at 1591
    vectors, where a 200-candidate search started failing with ef pinned to 64. The doubling is
    headroom -- recall at ef == k is poor, since the graph walk has no room to explore.
    """

    return min(max(64, top_k * 2), 32768)


class MilvusVectorIndex:
    collection_suffixes = {
        "image": "vl_images",
        "person_crop": "vl_person_crops",
        "face_embedding": "face_embeddings",
        # ReID vectors live apart from the VL ones: different model, different dimension, and
        # they answer a different question (who this is, not what they look like).
        "reid_person_crop": "reid_person_crops",
    }

    reid_object_types = frozenset({"reid_person_crop"})

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.text_embedding = TextEmbeddingService(settings)
        self.visual_embedding = VisualEmbeddingService(settings)
        self._connected = False
        self._collections: dict[str, object] = {}

    def is_enabled(self) -> bool:
        if not self.is_available():
            return False
        return self.visual_embedding.is_enabled() or self.text_embedding.is_enabled()

    def is_available(self) -> bool:
        """Milvus itself is configured and not cooling down after a failure.

        Raw-vector callers gate on this: they bring their own vectors, so requiring a text or
        visual embedding provider (which is_enabled does, for the paths that embed here) would
        couple ReID to configuration it never uses.
        """

        if not self.settings.milvus_enabled:
            return False
        return not self._is_in_failure_cooldown()

    def search_text(
        self,
        object_type: str,
        query: str,
        top_k: int,
    ) -> list[VectorSearchHit] | None:
        if not self.is_enabled():
            return None
        query_vector = self._embed_text_query(query)
        return self._search_vector(object_type, query_vector, top_k)

    def search_image(
        self,
        object_type: str,
        image_path: Path,
        top_k: int,
    ) -> list[VectorSearchHit] | None:
        if not self.is_enabled() or not self.visual_embedding.is_enabled():
            return None
        query_vector = self.visual_embedding.embed_image(image_path)
        return self._search_vector(object_type, query_vector, top_k)

    def search_vector(
        self,
        object_type: str,
        vector: list[float],
        top_k: int,
    ) -> list[VectorSearchHit] | None:
        """Search with a caller-supplied vector, for collections this class does not embed for."""

        # A ReID caller must be able to distinguish "no matches" from "the vector
        # store was never queried".  Text/image search remains an optional degraded
        # feature, but raw-vector identity search is correctness-sensitive.
        self._require_available("vector search")
        return self._search_vector(object_type, vector, top_k)

    def fetch_vectors(
        self,
        object_type: str,
        object_ids: list[uuid.UUID],
    ) -> dict[uuid.UUID, list[float]]:
        """Reads stored vectors back so callers can compare candidates with each other.

        A search only reports how close each candidate is to the query. Deciding whether two
        candidates are the same person needs their similarity to each other, which a search
        result cannot answer.
        """

        self._require_available("vector fetch")
        if not object_ids:
            return {}
        collection = self._collection(object_type)
        quoted = ", ".join(f'"{object_id}"' for object_id in object_ids)
        try:
            rows = collection.query(
                expr=f"object_id in [{quoted}]",
                output_fields=["object_id", "embedding"],
                limit=len(object_ids),
                timeout=self.settings.milvus_timeout_seconds,
            )
        except Exception as exc:
            # Deliberately not _mark_failure(): callers treat this read as optional and degrade
            # without it, so tripping the shared cooldown would let a nice-to-have take the
            # whole vector store offline -- including the search it was meant to improve.
            raise VectorIndexError(f"Milvus vector fetch failed: {exc}") from exc

        vectors: dict[uuid.UUID, list[float]] = {}
        for row in rows:
            object_id = row.get("object_id")
            embedding = row.get("embedding")
            if not object_id or embedding is None:
                continue
            vectors[uuid.UUID(str(object_id))] = list(embedding)
        return vectors

    def upsert_vector(
        self,
        object_type: str,
        object_id: uuid.UUID,
        vector: list[float],
        content: str = "",
        *,
        flush: bool = True,
    ) -> None:
        self._require_available("vector upsert")
        self._upsert_vector(object_type, object_id, vector, content, flush=flush)

    def _search_vector(
        self,
        object_type: str,
        vector: list[float],
        top_k: int,
    ) -> list[VectorSearchHit]:
        self._require_available("vector search")
        collection = self._collection(object_type)
        try:
            results = collection.search(
                data=[vector],
                anns_field="embedding",
                param={
                    "metric_type": self.settings.milvus_metric_type,
                    "params": {"ef": _hnsw_ef(top_k)},
                },
                limit=top_k,
                output_fields=["object_id"],
                timeout=self.settings.milvus_timeout_seconds,
            )
        except Exception as exc:
            self._mark_failure()
            raise VectorIndexError(f"Milvus search failed: {exc}") from exc

        hits: list[VectorSearchHit] = []
        for hit in results[0]:
            object_id = hit.entity.get("object_id")
            if not object_id:
                continue
            hits.append(
                VectorSearchHit(object_id=uuid.UUID(str(object_id)), score=float(hit.score))
            )
        return hits

    def upsert_text(
        self,
        object_type: str,
        object_id: uuid.UUID,
        text: str,
        *,
        flush: bool = True,
    ) -> None:
        self._require_available("text-vector upsert")
        if not (self.visual_embedding.is_enabled() or self.text_embedding.is_enabled()):
            raise VectorIndexError("No text or visual embedding provider is configured")
        vector = self._embed_text_query(text)
        self._upsert_vector(object_type, object_id, vector, text, flush=flush)

    def upsert_image(
        self,
        object_type: str,
        object_id: uuid.UUID,
        image_path: Path,
        content: str,
        *,
        flush: bool = True,
    ) -> None:
        self._require_available("image-vector upsert")
        if not (self.visual_embedding.is_enabled() or self.text_embedding.is_enabled()):
            raise VectorIndexError("No text or visual embedding provider is configured")
        if self.visual_embedding.is_enabled():
            vector = self.visual_embedding.embed_image(image_path)
        else:
            vector = self.text_embedding.embed_text(content)
        self._upsert_vector(object_type, object_id, vector, content, flush=flush)

    def flush(self, object_type: str) -> None:
        # Raw-vector callers (ReID) flush too, so this gates on Milvus alone like
        # search_vector/upsert_vector; gating on is_enabled() made a ReID-only
        # configuration silently skip the flush and lose durability.
        self._require_available("flush")
        collection = self._collection(object_type)
        try:
            collection.flush(timeout=self.settings.milvus_flush_timeout_seconds)
        except Exception as exc:
            self._mark_failure()
            raise VectorIndexError(f"Milvus flush failed: {exc}") from exc

    def _embed_text_query(self, text: str) -> list[float]:
        cache_key = self._text_query_cache_key(text)
        cached = _TEXT_QUERY_VECTOR_CACHE.get(cache_key)
        if cached is not None:
            return cached
        if self.visual_embedding.is_enabled():
            vector = self.visual_embedding.embed_text(text)
        else:
            vector = self.text_embedding.embed_text(text)
        if len(_TEXT_QUERY_VECTOR_CACHE) >= _TEXT_QUERY_VECTOR_CACHE_MAX:
            _TEXT_QUERY_VECTOR_CACHE.pop(next(iter(_TEXT_QUERY_VECTOR_CACHE)))
        _TEXT_QUERY_VECTOR_CACHE[cache_key] = vector
        return vector

    def _text_query_cache_key(self, text: str) -> tuple[str, ...]:
        if self.visual_embedding.is_enabled():
            return (
                "visual",
                self.settings.visual_embedding_provider,
                self.settings.visual_embedding_model,
                str(self.settings.visual_embedding_dim),
                self.settings.visual_embedding_instruction,
                str(self.settings.visual_embedding_device or ""),
                str(self.settings.visual_embedding_service_url or ""),
                str(self.settings.qwen3_vl_embedding_repo_dir or ""),
                str(self.settings.qwen3_vl_embedding_pythonpath or ""),
                str(self.settings.qwen3_vl_embedding_torch_dtype or ""),
                str(self.settings.qwen3_vl_embedding_attn_implementation or ""),
                text,
            )
        return (
            "text",
            self.settings.embedding_provider,
            self.settings.ollama_base_url,
            self.settings.ollama_embedding_model,
            str(self.settings.embedding_dim),
            text,
        )

    def _upsert_vector(
        self,
        object_type: str,
        object_id: uuid.UUID,
        vector: list[float],
        content: str,
        *,
        flush: bool,
    ) -> None:
        # Keep the internal primitive fail-closed too: future callers must not be
        # able to bypass the public mutation guards and then persist a SQL marker.
        self._require_available("vector upsert")
        collection = self._collection(object_type)
        object_id_text = str(object_id)
        pk = f"{object_type}:{object_id_text}"
        try:
            collection.delete(
                expr=f'object_id == "{object_id_text}"',
                timeout=self.settings.milvus_timeout_seconds,
            )
            collection.insert(
                [
                    [pk],
                    [object_id_text],
                    [content[:4096]],
                    [vector],
                ],
                timeout=self.settings.milvus_timeout_seconds,
            )
            if flush:
                collection.flush(timeout=self.settings.milvus_flush_timeout_seconds)
        except Exception as exc:
            self._mark_failure()
            raise VectorIndexError(f"Milvus upsert failed: {exc}") from exc

    def _collection(self, object_type: str) -> object:
        if object_type not in self.collection_suffixes:
            raise VectorIndexError(f"Unsupported vector object type: {object_type}")
        if object_type in self.reid_object_types and not self.reid_metric_supported:
            raise VectorIndexError(
                "ReID requires MILVUS_METRIC_TYPE=COSINE because its thresholds and "
                "gallery aggregation operate on cosine similarity"
            )
        collection_name = self._collection_name(object_type)
        if collection_name in self._collections:
            return self._collections[collection_name]
        cache_key = (
            self._connection_key(),
            collection_name,
            self._embedding_dim(object_type),
            self.settings.milvus_metric_type,
        )
        with _MILVUS_STATE_LOCK:
            cached_collection = _MILVUS_COLLECTION_CACHE.get(cache_key)
            if cached_collection is not None:
                self._collections[collection_name] = cached_collection
                return cached_collection
            setup_lock = _MILVUS_COLLECTION_LOCKS.setdefault(cache_key, threading.Lock())

        # Collection creation/loading is a one-time external side effect. Serialise it per
        # endpoint/name/schema so two queue workers cannot race to create or load the same name.
        with setup_lock:
            with _MILVUS_STATE_LOCK:
                cached_collection = _MILVUS_COLLECTION_CACHE.get(cache_key)
                if cached_collection is not None:
                    self._collections[collection_name] = cached_collection
                    return cached_collection
            self._connect()
            try:
                from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, utility
            except Exception as exc:
                raise VectorIndexError(f"pymilvus is not installed: {exc}") from exc

            try:
                return self._create_or_load_collection(
                    collection_name,
                    object_type,
                    cache_key,
                    Collection,
                    CollectionSchema,
                    DataType,
                    FieldSchema,
                    utility,
                )
            except VectorIndexError:
                raise
            except Exception as exc:
                # has_collection/create/load failures previously leaked raw pymilvus exceptions:
                # no cooldown, and callers saw a 500 instead of the degraded-index path.
                self._mark_failure()
                raise VectorIndexError(f"Milvus collection setup failed: {exc}") from exc

    def _create_or_load_collection(
        self,
        collection_name: str,
        object_type: str,
        cache_key: tuple,
        Collection,
        CollectionSchema,
        DataType,
        FieldSchema,
        utility,
    ) -> object:
        if not utility.has_collection(
            collection_name,
            using=self._alias,
            timeout=self.settings.milvus_timeout_seconds,
        ):
            fields = [
                FieldSchema(
                    name="pk",
                    dtype=DataType.VARCHAR,
                    max_length=160,
                    is_primary=True,
                ),
                FieldSchema(name="object_id", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=4096),
                FieldSchema(
                    name="embedding",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=self._embedding_dim(object_type),
                ),
            ]
            schema = CollectionSchema(fields=fields, description=f"SightIndex {object_type} index")
            collection = Collection(
                collection_name,
                schema=schema,
                using=self._alias,
                timeout=self.settings.milvus_timeout_seconds,
            )
            collection.create_index(
                field_name="embedding",
                index_params={
                    "index_type": "HNSW",
                    "metric_type": self.settings.milvus_metric_type,
                    "params": {"M": 16, "efConstruction": 128},
                },
                timeout=self.settings.milvus_timeout_seconds,
            )
        else:
            collection = Collection(
                collection_name,
                using=self._alias,
                timeout=self.settings.milvus_timeout_seconds,
            )
            self._validate_collection_dim(collection, collection_name, object_type)
        collection.load(timeout=self.settings.milvus_timeout_seconds)
        with _MILVUS_STATE_LOCK:
            _MILVUS_COLLECTION_CACHE[cache_key] = collection
        self._collections[collection_name] = collection
        return collection

    def _validate_collection_dim(
        self,
        collection: object,
        collection_name: str,
        object_type: str = "",
    ) -> None:
        schema = getattr(collection, "schema", None)
        fields = getattr(schema, "fields", []) if schema is not None else []
        embedding_field = next(
            (field for field in fields if getattr(field, "name", "") == "embedding"),
            None,
        )
        params = getattr(embedding_field, "params", {}) or {}
        collection_dim = params.get("dim")
        expected_dim = self._embedding_dim(object_type)
        if collection_dim is not None and int(collection_dim) != expected_dim:
            raise VectorIndexError(
                f"Milvus collection {collection_name} uses embedding dim {collection_dim}, "
                f"but current settings require {expected_dim}. Change MILVUS_COLLECTION_PREFIX "
                "or recreate the collection before switching embedding models."
            )

    @property
    def _alias(self) -> str:
        # pymilvus aliases are process-global. A fixed alias lets one Settings instance silently
        # repoint collections created by another instance at a different endpoint/database.
        material = "\0".join(self._connection_material(include_password=True))
        digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]
        return f"sightindex_{digest}"

    @property
    def namespace_identity(self) -> str:
        """Stable logical database identity used by SQL-side ReID markers."""

        configured = (self.settings.milvus_namespace_id or "").strip()
        if configured:
            return configured
        material = "\0".join(
            [
                self.settings.milvus_host.strip().lower(),
                str(self.settings.milvus_port),
                self.settings.milvus_db.strip(),
            ]
        )
        digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]
        return f"endpoint-{digest}"

    @property
    def reid_metric_supported(self) -> bool:
        return self.settings.milvus_metric_type.strip().upper() == "COSINE"

    def _connect(self) -> None:
        if self._connected:
            return
        connection_key = self._connection_key()
        with _MILVUS_STATE_LOCK:
            if connection_key in _MILVUS_CONNECTED_KEYS:
                self._connected = True
                return
            connection_lock = _MILVUS_CONNECTION_LOCKS.setdefault(
                connection_key,
                threading.Lock(),
            )

        # Do not hold the global state lock across a network call: an unreachable endpoint
        # must not block probes or queue workers using a different Milvus alias.
        with connection_lock:
            with _MILVUS_STATE_LOCK:
                if connection_key in _MILVUS_CONNECTED_KEYS:
                    self._connected = True
                    return
            try:
                from pymilvus import connections
            except Exception as exc:
                raise VectorIndexError(f"pymilvus is not installed: {exc}") from exc

            kwargs: dict[str, object] = {
                "alias": self._alias,
                "host": self.settings.milvus_host,
                "port": str(self.settings.milvus_port),
                "timeout": self.settings.milvus_timeout_seconds,
            }
            if self.settings.milvus_user:
                kwargs["user"] = self.settings.milvus_user
            if self.settings.milvus_password:
                kwargs["password"] = self.settings.milvus_password
            if self.settings.milvus_db:
                kwargs["db_name"] = self.settings.milvus_db
            try:
                connections.connect(**kwargs)
            except Exception as exc:
                self._mark_failure()
                raise VectorIndexError(f"Milvus connection failed: {exc}") from exc
            with _MILVUS_STATE_LOCK:
                _MILVUS_CONNECTED_KEYS.add(connection_key)
            self._connected = True

    def probe(self, timeout_seconds: float = 2.0) -> tuple[bool, str | None]:
        """Bounded connectivity check for status reporting: connect only, no collection work.

        A fresh process has never touched Milvus, so the cooldown state alone says nothing;
        without a real attempt, status would report ready against an unreachable server.
        """

        if not self.settings.milvus_enabled:
            return False, "MILVUS_ENABLED is not set"
        if self._is_in_failure_cooldown():
            return False, "Milvus is in failure cooldown after an earlier error"
        timeout = min(timeout_seconds, self.settings.milvus_timeout_seconds)
        clamped = self.settings.model_copy(update={"milvus_timeout_seconds": timeout})
        probe_index = MilvusVectorIndex(clamped)
        try:
            probe_index._connect()
            from pymilvus import utility

            # A cached alias is not a liveness result. list_collections performs a real bounded
            # RPC and catches a server that disappeared after an earlier successful connect.
            utility.list_collections(using=probe_index._alias, timeout=timeout)
        except Exception as exc:
            probe_index._mark_failure()
            return False, str(exc)[:300]
        return True, None

    def _is_in_failure_cooldown(self) -> bool:
        connection_key = self._connection_key()
        with _MILVUS_STATE_LOCK:
            failure_until = _MILVUS_FAILURE_UNTIL.get(connection_key, 0.0)
            if failure_until <= monotonic():
                _MILVUS_FAILURE_UNTIL.pop(connection_key, None)
                return False
            return True

    def _require_available(self, operation: str) -> None:
        """Reject correctness-sensitive operations instead of silently succeeding.

        Queue acknowledgement depends on every mutation and flush either completing or
        raising.  A no-op during the failure cooldown would otherwise delete the durable
        outbox job and commit an index marker for a vector that was never written.
        """

        if not self.settings.milvus_enabled:
            raise VectorIndexError(
                f"Milvus {operation} is unavailable: MILVUS_ENABLED is not set"
            )
        if self._is_in_failure_cooldown():
            raise VectorIndexError(
                f"Milvus {operation} is unavailable during failure cooldown"
            )

    def _mark_failure(self) -> None:
        cooldown_seconds = self.settings.milvus_failure_cooldown_seconds
        if cooldown_seconds <= 0:
            return
        with _MILVUS_STATE_LOCK:
            _MILVUS_FAILURE_UNTIL[self._connection_key()] = monotonic() + cooldown_seconds

    def _connection_material(self, *, include_password: bool) -> tuple[str, ...]:
        material = (
            self.settings.milvus_host,
            str(self.settings.milvus_port),
            self.settings.milvus_user or "",
            self.settings.milvus_db or "",
        )
        if include_password:
            return (*material, self.settings.milvus_password or "")
        return material

    def _connection_key(self) -> tuple[str, str, str, str, str, str]:
        return (
            self._alias,
            self.settings.milvus_host,
            str(self.settings.milvus_port),
            self.settings.milvus_user or "",
            self.settings.milvus_password or "",
            self.settings.milvus_db or "",
        )

    def _collection_name(self, object_type: str) -> str:
        suffix = self.collection_suffixes[object_type]
        configured_prefix = self.settings.milvus_collection_prefix
        if (
            object_type in {"image", "person_crop"}
            and self.settings.milvus_visual_collection_prefix
        ):
            configured_prefix = self.settings.milvus_visual_collection_prefix
        prefix = configured_prefix.strip("_")
        name = f"{prefix}_{suffix}" if prefix else suffix
        if object_type in self.reid_object_types:
            # Every vector-space and storage-namespace axis selects a fresh collection. This is
            # conservative even when a namespace points at another physical database, and keeps
            # a same-endpoint logical reset from mixing with a partially rebuilt old collection.
            name = f"{name}_{self.reid_space_digest()}"
        return name

    def reid_space_digest(self) -> str:
        material = "\0".join(
            [
                self.settings.reid_model,
                self.settings.reid_checkpoint_revision,
                str(self.settings.reid_embedding_dim),
                self.settings.reid_preprocess_version,
                self.settings.milvus_metric_type.upper(),
                self.namespace_identity,
            ]
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]

    def _embedding_dim(self, object_type: str = "") -> int:
        if object_type in self.reid_object_types:
            return self.settings.reid_embedding_dim
        if object_type == "face_embedding":
            return self.settings.face_embedding_dim
        if self.visual_embedding.is_enabled():
            return self.settings.visual_embedding_dim
        return self.settings.embedding_dim


class VectorIndexingService:
    # SQLite is also written by live capture workers. Commit each marker so an
    # offline rebuild never holds its single-writer lock across model inference.
    rebuild_commit_batch_size = 1
    rebuild_flush_batch_size = 100

    def __init__(self, db: Session, settings: Settings) -> None:
        self.db = db
        self.settings = settings
        self.index = MilvusVectorIndex(settings)
        self.captioner = VLMCaptionService(settings)

    def rebuild(self, target: str, limit: int) -> dict[str, object]:
        if target == "image":
            objects = list(
                self.db.scalars(select(Image).order_by(Image.created_at.desc()).limit(limit))
            )
            index_item = self.index_image
        elif target == "person_crop":
            objects = list(
                self.db.scalars(select(PersonCrop).order_by(PersonCrop.created_at.desc()).limit(limit))
            )
            index_item = self.index_crop
        else:
            raise ValueError("target must be image or person_crop")

        indexed = 0
        pending_commit = 0
        pending_flush = 0
        errors: list[str] = []
        for item in objects:
            try:
                index_item(item, flush=False)
                indexed += 1
                pending_commit += 1
                pending_flush += 1
            except (EmbeddingRuntimeError, VectorIndexError) as exc:
                errors.append(str(exc))
                break
            if pending_flush >= self.rebuild_flush_batch_size:
                try:
                    self.index.flush(target)
                except VectorIndexError as exc:
                    self.db.rollback()
                    errors.append(str(exc))
                    pending_commit = 0
                    pending_flush = 0
                    break
                pending_flush = 0
            if pending_commit >= self.rebuild_commit_batch_size:
                self.db.commit()
                pending_commit = 0
        if pending_flush:
            try:
                self.index.flush(target)
            except VectorIndexError as exc:
                self.db.rollback()
                errors.append(str(exc))
                pending_commit = 0
        if pending_commit:
            self.db.commit()
        return {
            "target": target,
            "requested": limit,
            "seen": len(objects),
            "indexed": indexed,
            "errors": errors,
        }

    def index_crop(self, crop: PersonCrop, *, flush: bool = True) -> None:
        """Write a crop vector and stage its SQL-side index metadata.

        Queue workers use :meth:`write_crop_vector` and :meth:`record_crop_index`
        separately so slow external I/O does not hold a SQLite write transaction and a
        job is acknowledged only after Milvus has flushed successfully.
        """

        self.write_crop_vector(crop, flush=flush)
        self.record_crop_index(crop)

    def write_crop_vector(self, crop: PersonCrop, *, flush: bool = True) -> None:
        """Write only the external Milvus vector for a crop."""

        content = self._crop_text(crop)
        crop_path = self._resolve_data_url(crop.crop_url)
        if crop_path and crop_path.exists():
            content = self._augment_with_vlm_caption(crop_path, "person_crop", content)
            self.index.upsert_image("person_crop", crop.id, crop_path, content, flush=flush)
            return
        self.index.upsert_text("person_crop", crop.id, content, flush=flush)

    def record_crop_index(self, crop: PersonCrop) -> None:
        """Stage crop marker and observation rows in the caller's transaction."""

        self._record_vl_embedding("person_crop", crop.id)
        ObservationIndexService(self.db, self.settings).upsert_crop(crop)

    def index_image(self, image: Image, *, flush: bool = True) -> None:
        """Write an image vector and stage its SQL-side index metadata."""

        self.write_image_vector(image, flush=flush)
        self.record_image_index(image)

    def write_image_vector(self, image: Image, *, flush: bool = True) -> None:
        """Write only the external Milvus vector for an image."""

        content = self._image_text(image)
        image_path = self._resolve_data_url(image.image_url)
        if image_path and image_path.exists():
            content = self._augment_with_vlm_caption(image_path, "image", content)
            self.index.upsert_image("image", image.id, image_path, content, flush=flush)
            return
        self.index.upsert_text("image", image.id, content, flush=flush)

    def record_image_index(self, image: Image) -> None:
        """Stage an image marker in the caller's transaction."""

        self._record_vl_embedding("image", image.id)

    def _record_vl_embedding(self, object_type: str, object_id: uuid.UUID) -> None:
        existing = self.db.scalar(
            select(VLEmbedding)
            .where(VLEmbedding.object_type == object_type, VLEmbedding.object_id == object_id)
            .order_by(VLEmbedding.created_at.desc())
        )
        if existing is not None:
            return
        self.db.add(
            VLEmbedding(
                object_type=object_type,
                object_id=object_id,
                embedding=None,
                embedding_model=(
                    self.settings.visual_embedding_model
                    if self.index.visual_embedding.is_enabled()
                    else self.settings.ollama_embedding_model
                ),
                embedding_dim=self.index._embedding_dim(),
            )
        )

    def _augment_with_vlm_caption(self, image_path: Path, object_type: str, content: str) -> str:
        try:
            caption = self.captioner.caption_image(image_path, object_type, content)
        except VLMRuntimeError as exc:
            return self._join_text([content, f"vlm caption error: {exc}"])
        if not caption:
            return content
        return self._join_text([content, f"vlm caption: {caption}"])

    def _resolve_data_url(self, url: str) -> Path | None:
        prefix = "/data/"
        if not url.startswith(prefix):
            return None
        return self.settings.data_dir / url.removeprefix(prefix)

    def _image_text(self, image: Image) -> str:
        return self._join_text(
            [
                "image",
                "monitoring frame",
                "监控画面",
                f"source type: {image.source_type}",
                self._time_text(image.captured_at or image.created_at),
                f"camera: {image.camera_id}" if image.camera_id else "",
                f"location: {image.location_id}" if image.location_id else "",
                f"url: {image.image_url}",
            ]
        )

    def _crop_text(self, crop: PersonCrop) -> str:
        bbox = crop.bbox or {}
        return self._join_text(
            [
                "person crop",
                "person",
                "人物裁剪",
                "行人",
                f"label: {bbox.get('label', 'person')}",
                (
                    f"confidence: {bbox.get('confidence')}"
                    if bbox.get("confidence") is not None
                    else ""
                ),
                (
                    "bbox: "
                    f"x={bbox.get('x')}, y={bbox.get('y')}, "
                    f"width={bbox.get('width')}, height={bbox.get('height')}"
                ),
                self._time_text(crop.captured_at or crop.created_at),
                f"camera: {crop.camera_id}" if crop.camera_id else "",
                f"location: {crop.location_id}" if crop.location_id else "",
                f"person: {crop.person_id}" if crop.person_id else "unknown person",
                f"url: {crop.crop_url}",
            ]
        )

    def _time_text(self, value: datetime | None) -> str:
        return f"time: {value.isoformat()}" if value else ""

    def _join_text(self, parts: list[str]) -> str:
        return " | ".join(part for part in parts if part)
