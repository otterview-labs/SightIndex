"""ReID-only Milvus behaviour through the real call chain, with pymilvus faked at the module
boundary (sys.modules) rather than by monkeypatching our own methods. Mocked, not a live Milvus.
"""

import importlib
import sys
import types
import uuid

import pytest


def load_app(monkeypatch, tmp_path, name: str, **env: str):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / f'{name}.db'}")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PERSON_DETECTOR", "whole_frame")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    for module_name in list(sys.modules):
        if module_name == "main" or module_name.startswith("app."):
            sys.modules.pop(module_name)
    return importlib.import_module("main")


REID_ONLY_ENV = {
    "REID_ENABLED": "true",
    "REID_SERVICE_URL": "http://reid.local",
    "MILVUS_ENABLED": "true",
    # Explicitly no text/visual embedding provider: the raw-vector path must not need them.
    "EMBEDDING_PROVIDER": "none",
    "VISUAL_EMBEDDING_PROVIDER": "none",
}


class FakeHits:
    def __init__(self, rows):
        self._rows = rows

    def __getitem__(self, index):
        assert index == 0
        return self._rows


class FakeHit:
    def __init__(self, object_id, score):
        self.entity = {"object_id": str(object_id)}
        self.score = score


class FakeCollection:
    def __init__(self, fail=None):
        self.fail = fail or set()
        self.inserted = []
        self.flushed = 0
        self.searches = 0

    def search(self, **kwargs):
        if "search" in self.fail:
            raise RuntimeError("milvus search boom")
        self.searches += 1
        return FakeHits([FakeHit(uuid.uuid4(), 0.9)])

    def delete(self, **kwargs):
        if "delete" in self.fail:
            raise RuntimeError("milvus delete boom")

    def insert(self, rows, **kwargs):
        if "insert" in self.fail:
            raise RuntimeError("milvus insert boom")
        self.inserted.append(rows)

    def flush(self, **kwargs):
        if "flush" in self.fail:
            raise RuntimeError("milvus flush boom")
        self.flushed += 1

    def create_index(self, **kwargs):
        if "create_index" in self.fail:
            raise RuntimeError("milvus create_index boom")

    def load(self, **kwargs):
        if "load" in self.fail:
            raise RuntimeError("milvus load boom")

    @property
    def schema(self):
        return None


def install_fake_pymilvus(monkeypatch, *, fail=None, collection=None):
    """A minimal pymilvus that records created collections and fails on demand."""

    fail = fail or set()
    created: dict[str, FakeCollection] = {}
    module = types.ModuleType("pymilvus")

    class _Connections:
        @staticmethod
        def connect(**kwargs):
            if "connect" in fail:
                raise RuntimeError("milvus connect boom")

    def _collection_factory(name, *args, **kwargs):
        if "collection" in fail:
            raise RuntimeError("milvus collection boom")
        instance = collection or created.get(name) or FakeCollection(fail=fail)
        created[name] = instance
        return instance

    class _Utility:
        @staticmethod
        def has_collection(name, **kwargs):
            if "has_collection" in fail:
                raise RuntimeError("milvus has_collection boom")
            return False

        @staticmethod
        def list_collections(**kwargs):
            if "list_collections" in fail:
                raise RuntimeError("milvus liveness boom")
            return list(created)

    class _FieldSchema:
        def __init__(self, **kwargs):
            pass

    class _CollectionSchema:
        def __init__(self, **kwargs):
            pass

    class _DataType:
        VARCHAR = "VARCHAR"
        FLOAT_VECTOR = "FLOAT_VECTOR"

    module.connections = _Connections()
    module.Collection = _collection_factory
    module.CollectionSchema = _CollectionSchema
    module.FieldSchema = _FieldSchema
    module.DataType = _DataType
    module.utility = _Utility()
    monkeypatch.setitem(sys.modules, "pymilvus", module)
    return created


def _clear_milvus_state():
    from app.services import vector_index

    vector_index._MILVUS_CONNECTED_KEYS.clear()
    vector_index._MILVUS_CONNECTION_LOCKS.clear()
    vector_index._MILVUS_COLLECTION_CACHE.clear()
    vector_index._MILVUS_COLLECTION_LOCKS.clear()
    vector_index._MILVUS_FAILURE_UNTIL.clear()


def test_reid_only_chain_search_upsert_flush(monkeypatch, tmp_path):
    """No text/visual provider anywhere, and the whole raw-vector chain still works."""

    load_app(monkeypatch, tmp_path, "test-milvus-chain", **REID_ONLY_ENV)
    from app.config.settings import get_settings
    from app.services.vector_index import MilvusVectorIndex

    created = install_fake_pymilvus(monkeypatch)
    _clear_milvus_state()
    index = MilvusVectorIndex(get_settings())
    assert index.is_enabled() is False  # embed-for-you paths stay off

    object_id = uuid.uuid4()
    index.upsert_vector("reid_person_crop", object_id, [0.1] * 4096)
    hits = index.search_vector("reid_person_crop", [0.1] * 4096, 5)
    index.flush("reid_person_crop")

    assert hits and hits[0].score == pytest.approx(0.9)
    (collection,) = created.values()
    assert collection.inserted
    # flush=True on upsert plus the explicit flush call.
    assert collection.flushed == 2


@pytest.mark.parametrize(
    "failing_op", ["connect", "has_collection", "load", "search", "insert", "flush"]
)
def test_milvus_failures_become_domain_errors_with_cooldown(monkeypatch, tmp_path, failing_op):
    load_app(monkeypatch, tmp_path, f"test-milvus-fail-{failing_op}", **REID_ONLY_ENV)
    from app.config.settings import get_settings
    from app.services.vector_index import MilvusVectorIndex, VectorIndexError

    install_fake_pymilvus(monkeypatch, fail={failing_op})
    _clear_milvus_state()
    index = MilvusVectorIndex(get_settings())

    with pytest.raises(VectorIndexError):
        if failing_op in {"search"}:
            index._search_vector("reid_person_crop", [0.1] * 4096, 5)
        elif failing_op in {"insert"}:
            index.upsert_vector("reid_person_crop", uuid.uuid4(), [0.1] * 4096)
        elif failing_op == "flush":
            index.flush("reid_person_crop")
        else:
            index._collection("reid_person_crop")

    # Every failure trips the cooldown, so is_available flips and status stops being ready.
    assert MilvusVectorIndex(get_settings()).is_available() is False


def test_status_not_ready_when_milvus_unreachable(monkeypatch, tmp_path):
    """A fresh process with Milvus down must report ready=false with an actionable error."""

    main = load_app(monkeypatch, tmp_path, "test-status-milvus-down", **REID_ONLY_ENV)
    from fastapi.testclient import TestClient

    from app.services.reid import ReidEmbeddingService

    install_fake_pymilvus(monkeypatch, fail={"connect"})
    _clear_milvus_state()
    # The ReID service itself is healthy; only Milvus is down.
    monkeypatch.setattr(
        ReidEmbeddingService, "probe", lambda self, timeout_seconds=3.0: (True, None)
    )
    with TestClient(main.create_app()) as client:
        payload = client.get("/api/reid/status").json()

    assert payload["enabled"] is True
    assert payload["reid_service_ok"] is True
    assert payload["milvus_ok"] is False
    assert payload["ready"] is False
    assert "connect boom" in payload["last_error"]


def test_probe_performs_rpc_even_when_connection_alias_is_cached(monkeypatch, tmp_path):
    load_app(monkeypatch, tmp_path, "test-probe-live-rpc", **REID_ONLY_ENV)
    from app.config.settings import get_settings
    from app.services.vector_index import MilvusVectorIndex

    install_fake_pymilvus(monkeypatch, fail={"list_collections"})
    _clear_milvus_state()
    first = MilvusVectorIndex(get_settings())
    first._connect()
    assert first._connection_key() in __import__(
        "app.services.vector_index", fromlist=["_MILVUS_CONNECTED_KEYS"]
    )._MILVUS_CONNECTED_KEYS

    ok, error = MilvusVectorIndex(get_settings()).probe()

    assert ok is False
    assert "liveness boom" in error


def test_probe_detects_service_identity_mismatch(monkeypatch, tmp_path):
    """A service running a different checkpoint than the index must fail readiness."""

    load_app(monkeypatch, tmp_path, "test-identity-mismatch", **REID_ONLY_ENV)
    from app.config.settings import get_settings
    from app.services.reid import ReidEmbeddingService, ReidRuntimeError

    service = ReidEmbeddingService(get_settings())
    mismatch = service._identity_mismatch(
        {
            "model": "sapiensid_wb4m",
            "checkpoint_revision": service.settings.reid_checkpoint_revision,
            "embedding_dim": 4096,
            "preprocess_version": "squarepad-v1",
        }
    )
    assert "sapiensid_wb4m" in mismatch

    with pytest.raises(ReidRuntimeError, match="identity mismatch"):
        service._validate_response_identity(
            {
                "model": "sapiensid_wb12m",
                "checkpoint_revision": service.settings.reid_checkpoint_revision,
                "dim": 4096,
                "preprocess_version": "stretch-v0",
            }
        )

    # Matching identity passes.
    assert (
        service._identity_mismatch(
            {"model": "sapiensid_wb12m", "dim": 4096, "preprocess_version": "squarepad-v1"}
        )
        == "service identity is incomplete; missing checkpoint_revision"
    )

    invalid_dim = service._identity_mismatch(
        {
            "model": "sapiensid_wb12m",
            "checkpoint_revision": service.settings.reid_checkpoint_revision,
            "dim": "not-an-int",
            "preprocess_version": "squarepad-v1",
        }
    )
    assert "dimension is invalid" in invalid_dim
    assert (
        service._identity_mismatch(
            {
                "model": service.settings.reid_model,
                "checkpoint_revision": service.settings.reid_checkpoint_revision,
                "dim": service.settings.reid_embedding_dim,
                "preprocess_version": service.settings.reid_preprocess_version,
            }
        )
        is None
    )


def test_space_switches_change_collection_and_requeue(monkeypatch, tmp_path):
    """Every identity/namespace axis must isolate the vector space or SQL coverage markers."""

    load_app(monkeypatch, tmp_path, "test-space-switch", **REID_ONLY_ENV)
    from app.config.settings import get_settings
    from app.services.reid_index import REID_OBJECT_TYPE, ReidIndexService
    from app.services.vector_index import MilvusVectorIndex

    base = get_settings()
    base_name = MilvusVectorIndex(base)._collection_name(REID_OBJECT_TYPE)

    by_model = base.model_copy(update={"reid_model": "sapiensid_wb4m"})
    by_revision = base.model_copy(update={"reid_checkpoint_revision": "sha256:new"})
    by_preprocess = base.model_copy(update={"reid_preprocess_version": "stretch-v0"})
    by_prefix = base.model_copy(update={"milvus_collection_prefix": "othersite"})
    by_namespace = base.model_copy(update={"milvus_namespace_id": "other-cluster"})
    by_metric = base.model_copy(update={"milvus_metric_type": "IP"})

    names = {
        "model": MilvusVectorIndex(by_model)._collection_name(REID_OBJECT_TYPE),
        "revision": MilvusVectorIndex(by_revision)._collection_name(REID_OBJECT_TYPE),
        "preprocess": MilvusVectorIndex(by_preprocess)._collection_name(REID_OBJECT_TYPE),
        "prefix": MilvusVectorIndex(by_prefix)._collection_name(REID_OBJECT_TYPE),
        "namespace": MilvusVectorIndex(by_namespace)._collection_name(REID_OBJECT_TYPE),
        "metric": MilvusVectorIndex(by_metric)._collection_name(REID_OBJECT_TYPE),
    }
    # Every axis lands in a different collection: searches during a partial rebuild can only
    # ever see vectors of their own space. Same-dimension model switches included.
    assert len({base_name, *names.values()}) == 7

    fingerprints = {
        "base": ReidIndexService(None, base).fingerprint,
        "model": ReidIndexService(None, by_model).fingerprint,
        "revision": ReidIndexService(None, by_revision).fingerprint,
        "preprocess": ReidIndexService(None, by_preprocess).fingerprint,
        "prefix": ReidIndexService(None, by_prefix).fingerprint,
        "namespace": ReidIndexService(None, by_namespace).fingerprint,
        "metric": ReidIndexService(None, by_metric).fingerprint,
    }
    assert len(set(fingerprints.values())) == 7


def test_reid_rejects_non_cosine_metric(monkeypatch, tmp_path):
    load_app(
        monkeypatch,
        tmp_path,
        "test-reid-metric",
        **REID_ONLY_ENV,
        MILVUS_METRIC_TYPE="L2",
    )
    from app.config.settings import get_settings
    from app.services.reid_index import REID_OBJECT_TYPE, ReidIndexService
    from app.services.vector_index import MilvusVectorIndex, VectorIndexError

    settings = get_settings()
    assert ReidIndexService(None, settings).is_enabled() is False
    with pytest.raises(VectorIndexError, match="requires MILVUS_METRIC_TYPE=COSINE"):
        MilvusVectorIndex(settings)._collection(REID_OBJECT_TYPE)


def test_connection_alias_and_fallback_namespace_follow_endpoint(monkeypatch, tmp_path):
    load_app(monkeypatch, tmp_path, "test-endpoint-identity", **REID_ONLY_ENV)
    from app.config.settings import get_settings
    from app.services.vector_index import MilvusVectorIndex

    base = get_settings()
    other = base.model_copy(update={"milvus_host": "milvus-b", "milvus_db": "tenant-b"})
    base_index = MilvusVectorIndex(base)
    other_index = MilvusVectorIndex(other)

    assert base_index._alias != other_index._alias
    assert base_index.namespace_identity != other_index.namespace_identity

    moved = other.model_copy(update={"milvus_namespace_id": "production-vectors"})
    moved_again = base.model_copy(update={"milvus_namespace_id": "production-vectors"})
    assert MilvusVectorIndex(moved).namespace_identity == MilvusVectorIndex(
        moved_again
    ).namespace_identity


def test_preprocess_switch_requeues_markers(monkeypatch, tmp_path):
    main = load_app(monkeypatch, tmp_path, "test-preproc-switch", **REID_ONLY_ENV)
    from fastapi.testclient import TestClient

    from app.config.settings import get_settings
    from app.db.session import SessionLocal
    from app.models.media import Image, PersonCrop
    from app.services.reid import ReidEmbeddingService
    from app.services.reid_index import ReidIndexService
    from app.services.vector_index import MilvusVectorIndex

    with TestClient(main.create_app()):
        pass
    crops_dir = tmp_path / "data" / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    (crops_dir / "c.jpg").write_bytes(b"img")

    monkeypatch.setattr(MilvusVectorIndex, "is_enabled", lambda self: True)
    monkeypatch.setattr(
        MilvusVectorIndex,
        "upsert_vector",
        lambda self, ot, oid, vec, content="", flush=True: None,
    )
    monkeypatch.setattr(MilvusVectorIndex, "flush", lambda self, ot: None)
    monkeypatch.setattr(
        ReidEmbeddingService,
        "embed_images",
        lambda self, paths: [[0.1] * self.dim for _ in paths],
    )

    base = get_settings()
    with SessionLocal() as db:
        image = Image(image_url="/data/frames/p.jpg", source_type="stream_frame")
        db.add(image)
        db.flush()
        crop = PersonCrop(image_id=image.id, crop_url="/data/crops/c.jpg", bbox={})
        db.add(crop)
        db.commit()
        db.refresh(crop)

        ReidIndexService(db, base).index_crops_batch([crop])
        db.commit()
        assert ReidIndexService(db, base).pending_count() == 0

        # The preprocessing changed (e.g. the square-pad fix): every old vector is stale.
        switched = base.model_copy(update={"reid_preprocess_version": "squarepad-v2"})
        assert ReidIndexService(db, switched).pending_count() == 1
