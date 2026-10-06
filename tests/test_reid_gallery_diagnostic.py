import json
import math
import sys
import types
import uuid
from collections.abc import Sequence
from typing import Any

import pytest

from scripts import diagnose_reid_gallery as diagnostic
from scripts.diagnose_reid_gallery import Hit, Options, compare_votes


def test_synthetic_censoring_is_reproducible_without_identity_claim() -> None:
    report = diagnostic.synthetic_report(Options())
    row = report["rows"][0]
    assert row["legacy_score"] == pytest.approx(0.80 / 3)
    assert row["complete_score"] == pytest.approx(0.46)
    assert row["floor_censored_votes"] == 2
    assert row["ann_missing_votes"] == 0
    assert row["complete_strong_vote_count"] == 1
    assert not row["legacy_clears_floor"]
    assert row["complete_clears_floor"]
    assert report["identity_accuracy"] is None
    assert row["legacy_rank_in_bounded_union"] is None
    assert row["complete_rank_in_bounded_union"] == 1


def test_complete_votes_recover_ann_truncation_not_just_floor_censoring() -> None:
    queries = {"q1": [1.0, 0.0], "q2": [1.0, 0.0], "q3": [1.0, 0.0]}
    report = compare_votes(queries, {"q1": [Hit("a", 0.8)]}, {"a": [0.8, 0.6]}, Options())
    row = report["rows"][0]
    assert row["ann_missing_votes"] == 2
    assert row["floor_censored_votes"] == 0
    assert row["legacy_score"] == pytest.approx(0.8 / 3)
    assert row["complete_score"] == pytest.approx(0.8)
    assert row["complete_strong_vote_count"] == 3


@pytest.mark.parametrize("missing", [None, [], [0, 0], [math.nan, 0], [1], [True, 0]])
def test_unknown_candidate_vector_is_not_zero_or_identity_evidence(missing: object) -> None:
    report = compare_votes({"q": [1.0, 0.0]}, {"q": [Hit("a", 0.8)]}, {"a": missing}, Options())
    row = report["rows"][0]
    assert row["legacy_score"] == 0.8
    assert row["complete_score"] is None
    assert row["complete_clears_floor"] is None
    assert row["complete_strong_vote_count"] is None
    assert row["complete_rank_in_bounded_union"] is None
    assert report["missing_or_invalid_candidate_vectors"] == 1


@pytest.mark.parametrize(
    "queries",
    [
        {},
        {"a": None},
        {"a": [1.0], "b": [1.0, 0.0]},
        {str(i): [1.0] for i in range(9)},
    ],
)
def test_incomplete_query_cannot_silently_change_denominator(queries: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        compare_votes(queries, {}, {}, Options())


def test_top_votes_remains_fixed_and_penalizes_one_lucky_vote() -> None:
    report = compare_votes(
        {"q1": [1.0, 0.0], "q2": [0.0, 1.0]},
        {"q1": [Hit("lucky", 0.99), Hit("steady", 0.85)], "q2": [Hit("steady", 0.85)]},
        {},
        Options(top_votes=2),
    )
    rows = {row["crop_id"]: row for row in report["rows"]}
    assert rows["steady"]["legacy_score"] == pytest.approx(0.85)
    assert rows["lucky"]["legacy_score"] == pytest.approx(0.495)
    assert rows["steady"]["legacy_rank_in_bounded_union"] == 1
    assert rows["lucky"]["legacy_rank_in_bounded_union"] == 2


def test_union_is_prefloor_bounded_and_ann_duplicate_hits_do_not_create_votes() -> None:
    report = compare_votes(
        {"q1": [1.0], "q2": [1.0]},
        {
            "q1": [Hit("a", 0.8), Hit("a", 0.9), Hit("b", 0.2), Hit("excluded", 1.0)],
            "q2": [Hit("c", 0.1)],
        },
        {"a": [1.0], "b": [1.0], "c": [1.0]},
        Options(ann_top_k=3, candidate_limit=2),
    )
    assert report["raw_union_count"] == 3
    assert report["union_omitted_count"] == 1
    assert [row["crop_id"] for row in report["rows"]] == ["a", "b"]
    assert report["rows"][0]["legacy_retained_votes"] == 1
    assert report["rows"][0]["legacy_score"] == pytest.approx(0.45)
    assert report["rows"][1]["legacy_score"] is None
    assert report["rows"][1]["complete_score"] == 1.0


def test_known_negative_scores_are_not_replaced_with_zero() -> None:
    report = compare_votes(
        {"q": [1.0, 0.0]}, {"q": [Hit("a", -0.8)]}, {"a": [-0.8, 0.6]}, Options(floor=-1.0)
    )
    assert report["rows"][0]["complete_score"] == pytest.approx(-0.8)
    assert report["rows"][0]["legacy_score"] == pytest.approx(-0.8)


def test_empty_ann_union_is_empty_not_missing_vector_failure() -> None:
    report = compare_votes({"q": [1.0]}, {}, {}, Options())
    assert report["rows"] == []
    assert report["raw_union_count"] == 0


@pytest.mark.parametrize(
    "values", [None, "text", [], [False], [math.inf], [0], [10**500], [1.0] * 8193]
)
def test_invalid_vectors_are_rejected(values: object) -> None:
    assert diagnostic.unit_vector(values) is None


def test_stable_normalization_and_numpy_float32_vectors() -> None:
    import numpy as np

    assert diagnostic.unit_vector([1e308, 1e308]) == pytest.approx((2**-0.5, 2**-0.5))
    assert diagnostic.unit_vector([np.float32(3), np.float32(4)]) == pytest.approx((0.6, 0.8))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"top_votes": 0},
        {"top_votes": 9},
        {"ann_top_k": 201},
        {"candidate_limit": 1001},
        {"candidate_limit": True},
        {"floor": math.nan},
        {"floor": 2.0},
        {"strong_score": -2.0},
    ],
)
def test_resource_limits_and_bars_are_validated(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        Options(**kwargs)


@pytest.mark.parametrize("score", [math.nan, math.inf, 2.0])
def test_non_cosine_ann_scores_fail(score: float) -> None:
    with pytest.raises(ValueError):
        compare_votes({"q": [1.0]}, {"q": [Hit("a", score)]}, {}, Options())


@pytest.mark.parametrize("prefix,namespace", [("__site__", None), ("", "logical-v2")])
def test_collection_name_matches_current_production_space(
    prefix: str, namespace: str | None
) -> None:
    from app.config.settings import Settings
    from app.services.vector_index import MilvusVectorIndex

    settings = Settings(
        _env_file=None,
        milvus_collection_prefix=prefix,
        milvus_namespace_id=namespace,
        milvus_host="Example.COM",
    )
    assert diagnostic.configured_collection_name(settings) == (
        MilvusVectorIndex(settings)._collection_name("reid_person_crop")
    )


class ReadOnlyCollection:
    """Expose query/search only: any data/index mutation would fail the tests."""

    def __init__(self, ids: Sequence[str]) -> None:
        self.ids = list(ids)
        self.calls: list[str] = []
        self.schema = types.SimpleNamespace(
            fields=[
                types.SimpleNamespace(name="embedding", params={"dim": 2}),
            ]
        )

    def query(self, *, expr: str, **kwargs: Any) -> list[dict[str, object]]:
        self.calls.append("query")
        requested = json.loads(expr.removeprefix("object_id in "))
        assert len(requested) <= 100
        return [{"object_id": crop_id, "embedding": [1.0, 0.0]} for crop_id in requested]

    def search(self, **kwargs: Any) -> list[list[Any]]:
        self.calls.append("search")
        return [[types.SimpleNamespace(entity={"object_id": self.ids[-1]}, score=0.8)]]


def test_live_collection_reader_uses_only_bounded_query_and_search() -> None:
    ids = [str(uuid.uuid4()) for _ in range(3)]
    collection = ReadOnlyCollection(ids)
    report = diagnostic._read_collection(collection, ids[:2], Options(), 1.0, 2)
    assert collection.calls == ["query", "search", "search", "query"]
    assert report["mode"] == "stored_vectors_read_only"
    assert report["rows"][0]["complete_score"] == 1.0
    assert "embedding" not in json.dumps(report)


def test_collection_reader_checks_dimension_before_search() -> None:
    ids = [str(uuid.uuid4())]
    collection = ReadOnlyCollection(ids)
    with pytest.raises(ValueError, match="dimension"):
        diagnostic._read_collection(collection, ids, Options(), 1.0, 3)
    assert collection.calls == ["query"]


def test_fetch_chunks_ids_and_rejects_expression_injection() -> None:
    ids = [str(uuid.uuid4()) for _ in range(201)]
    collection = ReadOnlyCollection(ids)
    assert len(diagnostic._fetch_vectors(collection, ids, 1.0)) == 201
    assert collection.calls == ["query"] * 3
    with pytest.raises(ValueError):
        diagnostic._fetch_vectors(collection, ['x" or 1==1'], 1.0)


def test_synthetic_cli_emits_scalar_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert diagnostic.main(["--synthetic"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["mode"] == "synthetic"
    assert "embedding" not in json.dumps(output)


@pytest.mark.parametrize(
    "args",
    [
        ["--synthetic", "--floor", "nan"],
        ["--crop-id", str(uuid.UUID(int=1)), "--crop-id", str(uuid.UUID(int=1))],
    ],
)
def test_cli_refuses_invalid_limits_before_connecting(args: list[str]) -> None:
    with pytest.raises(SystemExit) as error:
        diagnostic.main(args)
    assert error.value.code == 2


def test_cli_does_not_leak_sensitive_client_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(*args: Any) -> dict[str, Any]:
        raise RuntimeError("password=secret-credential")

    monkeypatch.setattr(diagnostic, "live_report", fail)
    assert diagnostic.main(["--crop-id", str(uuid.uuid4())]) == 1
    output = capsys.readouterr().out
    assert "secret-credential" not in output
    assert json.loads(output)["status"] == "error"


@pytest.mark.parametrize(
    "exists,dimension,enabled",
    [(True, 2, True), (False, 2, True), (True, 3, True), (True, 2, False)],
)
def test_live_wrapper_never_creates_or_loads_collections(
    monkeypatch: pytest.MonkeyPatch,
    exists: bool,
    dimension: int,
    enabled: bool,
) -> None:
    from app.config import settings as settings_module

    crop_id = str(uuid.uuid4())
    collection = ReadOnlyCollection([crop_id])
    calls: list[str] = []
    fake = types.ModuleType("pymilvus")
    fake.connections = types.SimpleNamespace(  # type: ignore[attr-defined]
        connect=lambda **kwargs: calls.append("connect"),
        disconnect=lambda alias: calls.append("disconnect"),
    )
    fake.utility = types.SimpleNamespace(  # type: ignore[attr-defined]
        has_collection=lambda *args, **kwargs: exists,
    )
    fake.Collection = lambda *args, **kwargs: collection  # type: ignore[attr-defined]
    settings = settings_module.Settings(
        _env_file=None,
        milvus_enabled=enabled,
        reid_embedding_dim=dimension,
        milvus_user="user",
        milvus_password="private",
    )
    monkeypatch.setattr(settings_module, "get_settings", lambda: settings)
    monkeypatch.setitem(sys.modules, "pymilvus", fake)
    if enabled and exists and dimension == 2:
        assert diagnostic.live_report([crop_id], Options())["mode"] == "stored_vectors_read_only"
        assert collection.calls == ["query", "search", "query"]
    else:
        with pytest.raises(ValueError):
            diagnostic.live_report([crop_id], Options())
    assert calls == (["connect", "disconnect"] if enabled else [])
