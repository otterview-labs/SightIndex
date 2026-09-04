import uuid

import pytest

from app.config.settings import Settings
from app.services.reid_index import COLLAPSE_CANDIDATE_LIMIT
from app.services.vector_index import (
    MilvusVectorIndex,
    VectorIndexError,
    _hnsw_ef,
)


@pytest.mark.parametrize("top_k", [1, 20, 64, 200, COLLAPSE_CANDIDATE_LIMIT, 5000])
def test_hnsw_ef_is_never_below_the_requested_k(top_k):
    """Milvus rejects ef < k outright, and only once the collection is big enough to index."""

    assert _hnsw_ef(top_k) >= top_k


def test_hnsw_ef_stays_inside_the_engine_limit():
    assert _hnsw_ef(1_000_000) <= 32768


class _RecordingCollection:
    def __init__(self) -> None:
        self.params: dict | None = None

    def search(self, *, param, limit, **_kwargs):
        self.params = {**param, "limit": limit}
        return [[]]


def test_search_passes_an_ef_matching_the_candidate_pool(monkeypatch, tmp_path):
    collection = _RecordingCollection()
    settings = Settings(data_dir=tmp_path, milvus_enabled=True)
    index = MilvusVectorIndex(settings)
    monkeypatch.setattr(MilvusVectorIndex, "_collection", lambda self, object_type: collection)

    index.search_vector("reid_person_crop", [0.1] * 8, COLLAPSE_CANDIDATE_LIMIT)

    assert collection.params is not None
    assert collection.params["params"]["ef"] >= collection.params["limit"]


def test_face_collection_uses_the_face_embedding_dimension(tmp_path):
    settings = Settings(
        data_dir=tmp_path,
        milvus_enabled=True,
        face_embedding_dim=512,
        visual_embedding_dim=2048,
    )

    index = MilvusVectorIndex(settings)

    assert index.collection_suffixes["face_embedding"] == "face_embeddings"
    assert index._embedding_dim("face_embedding") == 512


def test_visual_collection_prefix_does_not_move_identity_indexes(tmp_path):
    settings = Settings(
        data_dir=tmp_path,
        milvus_enabled=True,
        milvus_collection_prefix="sightindex",
        milvus_visual_collection_prefix="sightindex_clip",
    )

    index = MilvusVectorIndex(settings)

    assert index._collection_name("image") == "sightindex_clip_vl_images"
    assert index._collection_name("person_crop") == "sightindex_clip_vl_person_crops"
    assert index._collection_name("face_embedding") == "sightindex_face_embeddings"
    assert index._collection_name("reid_person_crop").startswith(
        "sightindex_reid_person_crops_"
    )


def test_a_failed_vector_fetch_does_not_disable_the_whole_index(monkeypatch, tmp_path):
    """The fetch is optional; tripping the shared cooldown would take the search down with it."""

    class _Broken:
        def query(self, **_kwargs):
            raise RuntimeError("boom")

    settings = Settings(data_dir=tmp_path, milvus_enabled=True)
    index = MilvusVectorIndex(settings)
    monkeypatch.setattr(MilvusVectorIndex, "_collection", lambda self, object_type: _Broken())

    with pytest.raises(VectorIndexError):
        index.fetch_vectors("reid_person_crop", [uuid.uuid4()])

    assert index.is_available(), "an optional read put Milvus into failure cooldown"
