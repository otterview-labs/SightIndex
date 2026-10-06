"""Read-only, bounded comparison of sparse and complete ReID gallery votes.

Run ``python scripts/diagnose_reid_gallery.py --synthetic`` without app dependencies.
On the server, repeat ``--crop-id UUID`` for up to eight deliberately selected query
frames. Stored vectors, not images, are read from the existing configured Milvus
collection. This does not run automatic tracklet selection or the full API pipeline.
No models, face caches, database migrations, collection loads, or index writes run.
JSON goes only to stdout and never includes images, embeddings, or credentials.

Complete-vote scoring is an experiment, NOT a production threshold recommendation.
In particular, the current sparse-vote rule deliberately penalizes one lucky frame.
An increased score or recovered candidate does not establish the same identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.config.settings import Settings

MAX_QUERY_FRAMES = 8
MAX_ANN_TOP_K = 200
MAX_CANDIDATES = 1000
MAX_DIMENSION = 8192


@dataclass(frozen=True)
class Hit:
    """One scalar ANN hit, before the per-frame similarity floor."""

    crop_id: str
    score: float


@dataclass(frozen=True)
class Options:
    """Fixed resource limits and diagnostic comparison bars."""

    floor: float = 0.30
    top_votes: int = 3
    ann_top_k: int = 200
    candidate_limit: int = 500
    strong_score: float = 0.50

    def __post_init__(self) -> None:
        for name, maximum in (
            ("top_votes", MAX_QUERY_FRAMES),
            ("ann_top_k", MAX_ANN_TOP_K),
            ("candidate_limit", MAX_CANDIDATES),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
                raise ValueError(f"{name} must be an integer in 1..{maximum}")
        for name in ("floor", "strong_score"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or not -1 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [-1, 1]")


def unit_vector(values: object, dimension: int | None = None) -> tuple[float, ...] | None:
    """Validate and normalize without overflowing or fabricating absent vectors."""

    if not isinstance(values, list | tuple) or not 1 <= len(values) <= MAX_DIMENSION:
        return None
    if dimension is not None and len(values) != dimension:
        return None
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in values):
        return None
    try:
        numeric = tuple(float(value) for value in values)
    except OverflowError:
        return None
    if any(not math.isfinite(value) for value in numeric):
        return None
    maximum = max(abs(value) for value in numeric)
    if maximum == 0:
        return None
    scaled = tuple(value / maximum for value in numeric)
    norm = math.sqrt(sum(value * value for value in scaled))
    return tuple(value / norm for value in scaled)


def _query_vectors(vectors: Mapping[str, object]) -> dict[str, tuple[float, ...]]:
    if not 1 <= len(vectors) <= MAX_QUERY_FRAMES:
        raise ValueError(f"Supply 1..{MAX_QUERY_FRAMES} distinct query frames")
    validated: dict[str, tuple[float, ...]] = {}
    dimension: int | None = None
    for crop_id, vector in vectors.items():
        normalized = unit_vector(vector, dimension)
        if normalized is None:
            raise ValueError(f"Query vector unavailable or invalid: {crop_id}")
        dimension = len(normalized)
        validated[crop_id] = normalized
    return validated


def _bounded_hits(
    query_ids: Sequence[str],
    hits_by_query: Mapping[str, Sequence[Hit]],
    options: Options,
) -> tuple[list[str], dict[str, dict[str, float]], int]:
    """Bound the pre-floor union by strongest ANN score; do not scan a collection."""

    by_candidate: dict[str, dict[str, float]] = {}
    for query_id in query_ids:
        for hit in hits_by_query.get(query_id, ())[: options.ann_top_k]:
            if not math.isfinite(hit.score) or not -1.00001 <= hit.score <= 1.00001:
                raise ValueError("ANN returned a non-finite or non-cosine score")
            per_frame = by_candidate.setdefault(hit.crop_id, {})
            per_frame[query_id] = max(per_frame.get(query_id, -1.0), hit.score)
    ordered = sorted(
        by_candidate,
        key=lambda crop_id: (-max(by_candidate[crop_id].values()), crop_id),
    )
    return ordered[: options.candidate_limit], by_candidate, len(ordered)


def compare_votes(
    query_vectors: Mapping[str, object],
    hits_by_query: Mapping[str, Sequence[Hit]],
    candidate_vectors: Mapping[str, object],
    options: Options,
) -> dict[str, Any]:
    """Compare rules on the same bounded ANN union, never infer human identity.

    Missing candidate vectors yield a null complete score. Missing query vectors
    fail the experiment instead of changing the denominator or treating them as zero.
    Legacy null means no retained vote, distinct from a known numeric zero score.
    """

    queries = _query_vectors(query_vectors)
    candidate_ids, by_candidate, union_size = _bounded_hits(list(queries), hits_by_query, options)
    denominator = min(options.top_votes, len(queries))
    dimension = len(next(iter(queries.values())))
    rows: list[dict[str, Any]] = []
    for crop_id in candidate_ids:
        observed = by_candidate[crop_id]
        retained = sorted(
            (score for score in observed.values() if score >= options.floor), reverse=True
        )
        legacy = sum(retained[:denominator]) / denominator if retained else None
        candidate = unit_vector(candidate_vectors.get(crop_id), dimension)
        complete_scores = (
            [
                max(-1.0, min(1.0, sum(a * b for a, b in zip(query, candidate, strict=True))))
                for query in queries.values()
            ]
            if candidate is not None
            else None
        )
        complete = (
            sum(sorted(complete_scores, reverse=True)[:denominator]) / denominator
            if complete_scores is not None
            else None
        )
        rows.append(
            {
                "crop_id": crop_id,
                "ann_recalled_frames": len(observed),
                "ann_missing_votes": len(queries) - len(observed),
                "floor_censored_votes": len(observed) - len(retained),
                "legacy_retained_votes": len(retained),
                "legacy_score": legacy,
                "legacy_clears_floor": legacy is not None and legacy >= options.floor,
                "candidate_vector_status": "valid"
                if candidate is not None
                else "missing_or_invalid",
                "complete_score": complete,
                "complete_clears_floor": (
                    complete >= options.floor if complete is not None else None
                ),
                "complete_scores_by_query": complete_scores,
                "complete_strong_vote_count": (
                    sum(score >= options.strong_score for score in complete_scores)
                    if complete_scores is not None
                    else None
                ),
                "score_delta": complete - legacy
                if complete is not None and legacy is not None
                else None,
            }
        )
    for prefix in ("legacy", "complete"):
        eligible = sorted(
            (row for row in rows if row[f"{prefix}_clears_floor"]),
            key=lambda row: (-row[f"{prefix}_score"], row["crop_id"]),
        )
        ranks = {row["crop_id"]: rank for rank, row in enumerate(eligible, 1)}
        for row in rows:
            row[f"{prefix}_rank_in_bounded_union"] = ranks.get(row["crop_id"])
    return {
        "schema_version": 1,
        "status": "diagnostic_only",
        "query_crop_ids": list(queries),
        "query_frame_count": len(queries),
        "denominator": denominator,
        "floor": options.floor,
        "strong_vote_threshold": options.strong_score,
        "ann_top_k_per_query": options.ann_top_k,
        "raw_union_count": union_size,
        "candidate_limit": options.candidate_limit,
        "compared_union_count": len(rows),
        "union_omitted_count": union_size - len(rows),
        "missing_or_invalid_candidate_vectors": sum(
            row["candidate_vector_status"] != "valid" for row in rows
        ),
        "identity_accuracy": None,
        "limitations": [
            "No identity ground truth: score/rank changes are not accuracy or same-person proof.",
            "Complete-vote thresholds need separate calibration and held-out validation.",
            "Only the bounded ANN union is compared; never a full-collection recall measurement.",
            "No live tracklet selection, camera quotas, face/tag fusion, collapse, or admission.",
            "Stored query vectors may differ from freshly batched inference used by the API.",
            "Strong votes use a diagnostic score bar, not a reliable identity decision.",
        ],
        "rows": rows,
    }


def synthetic_report(options: Options) -> dict[str, Any]:
    """Reproduce floor censoring with genuine unit vectors and no dependencies."""

    scores = (0.80, 0.29, 0.29)
    queries = {
        f"synthetic-query-{i}": [score, math.sqrt(1 - score * score)]
        for i, score in enumerate(scores, 1)
    }
    hits = {
        query_id: [Hit("synthetic-candidate", score)]
        for query_id, score in zip(queries, scores, strict=True)
    }
    report = compare_votes(queries, hits, {"synthetic-candidate": [1.0, 0.0]}, options)
    report["mode"] = "synthetic"
    return report


def configured_collection_name(settings: Settings) -> str:
    """Derive the current space name without importing/initializing model services."""

    namespace = (settings.milvus_namespace_id or "").strip()
    if not namespace:
        endpoint = "\0".join(
            (
                settings.milvus_host.strip().lower(),
                str(settings.milvus_port),
                settings.milvus_db.strip(),
            )
        )
        namespace = "endpoint-" + hashlib.sha256(endpoint.encode()).hexdigest()[:16]
    material = "\0".join(
        (
            settings.reid_model,
            settings.reid_checkpoint_revision,
            str(settings.reid_embedding_dim),
            settings.reid_preprocess_version,
            settings.milvus_metric_type.upper(),
            namespace,
        )
    )
    digest = hashlib.sha256(material.encode()).hexdigest()[:16]
    prefix = settings.milvus_collection_prefix.strip("_")
    return f"{prefix + '_' if prefix else ''}reid_person_crops_{digest}"


def _fetch_vectors(collection: Any, ids: Sequence[str], timeout: float) -> dict[str, object]:
    vectors: dict[str, object] = {}
    # UUID validation prevents interpolated filter expressions from accepting arbitrary text.
    validated_ids = [str(uuid.UUID(crop_id)) for crop_id in ids]
    for start in range(0, len(validated_ids), 100):
        batch = validated_ids[start : start + 100]
        rows = collection.query(
            expr=f"object_id in {json.dumps(batch)}",
            output_fields=["object_id", "embedding"],
            limit=len(batch),
            timeout=timeout,
        )
        requested = set(batch)
        for row in rows:
            crop_id = str(row.get("object_id", ""))
            if crop_id in requested:
                raw = row.get("embedding")
                # pymilvus can return a numpy array; normalize the container, not its contents.
                try:
                    vectors[crop_id] = list(raw) if raw is not None else None
                except TypeError:
                    vectors[crop_id] = None
    return vectors


def _read_collection(
    collection: Any,
    query_ids: Sequence[str],
    options: Options,
    timeout: float,
    dimension: int,
) -> dict[str, Any]:
    """The live reader only calls query/search on an already existing collection."""

    stored = _fetch_vectors(collection, query_ids, timeout)
    query_vectors = {crop_id: stored.get(crop_id) for crop_id in query_ids}
    validated = _query_vectors(query_vectors)
    if any(len(vector) != dimension for vector in validated.values()):
        raise ValueError("Stored query dimension differs from configured ReID dimension")
    hits_by_query: dict[str, list[Hit]] = {}
    for crop_id, vector in validated.items():
        results = collection.search(
            data=[list(vector)],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": max(64, options.ann_top_k * 2)}},
            limit=options.ann_top_k,
            output_fields=["object_id"],
            timeout=timeout,
        )
        hits_by_query[crop_id] = [
            Hit(str(uuid.UUID(str(hit.entity.get("object_id")))), float(hit.score))
            for hit in results[0]
        ]
    candidates, _, _ = _bounded_hits(list(validated), hits_by_query, options)
    candidate_vectors = _fetch_vectors(collection, candidates, timeout)
    report = compare_votes(query_vectors, hits_by_query, candidate_vectors, options)
    report["mode"] = "stored_vectors_read_only"
    return report


def live_report(query_ids: Sequence[str], options: Options) -> dict[str, Any]:
    """Read an existing, already loaded collection; refuse missing/mismatched spaces."""

    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)
    from pymilvus import Collection, connections, utility

    from app.config.settings import get_settings

    settings = get_settings()
    if not settings.milvus_enabled or settings.milvus_metric_type.strip().upper() != "COSINE":
        raise ValueError("Requires existing enabled COSINE Milvus index")
    if not 1 <= settings.reid_embedding_dim <= MAX_DIMENSION:
        raise ValueError("Configured embedding dimension exceeds diagnostic safety limit")
    alias = f"reid_diagnostic_{uuid.uuid4().hex}"
    timeout = min(settings.milvus_timeout_seconds, 10.0)
    connection: dict[str, Any] = {
        "alias": alias,
        "host": settings.milvus_host,
        "port": settings.milvus_port,
        "db_name": settings.milvus_db,
        "timeout": timeout,
    }
    if settings.milvus_user:
        connection["user"] = settings.milvus_user
    if settings.milvus_password:
        connection["password"] = settings.milvus_password
    try:
        connections.connect(**connection)
        name = configured_collection_name(settings)
        if not utility.has_collection(name, using=alias, timeout=timeout):
            raise ValueError("Configured ReID collection is absent; refusing to create it")
        collection = Collection(name, using=alias, timeout=timeout)
        field = next(
            (field for field in collection.schema.fields if field.name == "embedding"), None
        )
        if field is None or int(field.params.get("dim", 0)) != settings.reid_embedding_dim:
            raise ValueError("Existing collection dimension differs from ReID settings")
        report = _read_collection(
            collection, query_ids, options, timeout, settings.reid_embedding_dim
        )
        report["collection"] = name
        return report
    finally:
        connections.disconnect(alias)


def main(argv: Sequence[str] | None = None) -> int:
    """Print scalar-only JSON; no output files or external inference are created."""

    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--synthetic", action="store_true")
    mode.add_argument(
        "--crop-id",
        action="append",
        type=uuid.UUID,
        help="Repeat for 1..8 known query frames; no automatic identity selection",
    )
    parser.add_argument(
        "--floor",
        type=float,
        default=0.30,
        help="Explicit experimental per-frame/final floor (default: 0.30)",
    )
    parser.add_argument("--top-votes", type=int, default=3)
    parser.add_argument("--ann-top-k", type=int, default=200)
    parser.add_argument("--candidate-limit", type=int, default=500)
    parser.add_argument("--strong-score", type=float, default=0.50)
    args = parser.parse_args(argv)
    try:
        options = Options(
            args.floor, args.top_votes, args.ann_top_k, args.candidate_limit, args.strong_score
        )
        query_ids = [str(crop_id) for crop_id in (args.crop_id or [])]
        if not args.synthetic and (
            not 1 <= len(query_ids) <= MAX_QUERY_FRAMES or len(set(query_ids)) != len(query_ids)
        ):
            raise ValueError("Supply 1..8 distinct --crop-id values")
    except ValueError as exc:
        parser.error(str(exc))
    try:
        report = synthetic_report(options) if args.synthetic else live_report(query_ids, options)
    except Exception as exc:
        # Client errors may contain connection material. Do not print raw exceptions/tracebacks.
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "message": "Read-only diagnostic failed; check vector availability, "
                    "configuration and that the existing collection is already loaded.",
                }
            )
        )
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
