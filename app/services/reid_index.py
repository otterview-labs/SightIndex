import math
import uuid
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from time import monotonic

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is a declared dependency
    np = None  # type: ignore[assignment]

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from app.config.settings import Settings
from app.models.media import PersonCrop
from app.models.vectors import VectorIndexCapacityLock, VLEmbedding
from app.schemas.reid import ReidMatchItem
from app.services.observation_index import ObservationIndexService
from app.services.reid import ReidEmbeddingService, ReidRuntimeError
from app.services.vector_index import MilvusVectorIndex, VectorIndexError

REID_OBJECT_TYPE = "reid_person_crop"
BATCH_SIZE = 16
# Collapsing merges an unbounded number of consecutive frames into one visit, so the candidate
# pool cannot be the number of visits the caller asked for: twenty rows from one doorway merge
# into a single result. Ask Milvus for its cap and let the grouping decide what survives.
COLLAPSE_CANDIDATE_LIMIT = 200


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return -1.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    denominator = left_norm * right_norm
    return dot / denominator if denominator > 0 else -1.0


def _bounded_number(value: object, *, default: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return default
    return max(0.0, min(1.0, float(value)))


def _positive_number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0.0
    return max(0.0, float(value))


def collapse_occurrences(
    items: list["ReidMatchItem"],
    window_seconds: float,
    limit: int,
    *,
    vectors: dict[uuid.UUID, list[float]] | None = None,
    identity_threshold: float = 0.0,
) -> list["ReidMatchItem"]:
    """Merges frames of one person at one camera into a single visit.

    Three conditions, all required. Same camera and a gap no larger than the window are the
    obvious ones. The third is that the frames actually look like the same person, and it is not
    optional: time and camera alone chain a busy doorway into one endless visit. Measured on a
    real feed, a 60s window swallowed 127 frames across five minutes, and half of the 9180 pairs
    inside that supposed single visit scored below 0.7 against each other.

    Similarity between two candidates cannot be read off their scores, which only say how close
    each is to the query, so callers pass the stored vectors. Without them the identity test is
    skipped and grouping degrades to the time-only behaviour.
    """

    if window_seconds <= 0:
        return items[:limit]

    similarity = _pairwise_similarity(vectors)
    by_camera: dict[uuid.UUID | None, list[ReidMatchItem]] = {}
    groups: list[list[ReidMatchItem]] = []
    for item in items:
        if item.captured_at is None:
            # Undated crops cannot be neighbours of anything; merging them would be a guess.
            groups.append([item])
        else:
            by_camera.setdefault(item.camera_id, []).append(item)

    for camera_items in by_camera.values():
        camera_items.sort(key=lambda item: item.captured_at)
        open_groups: list[list[ReidMatchItem]] = []
        for item in camera_items:
            joined = _join_open_group(
                open_groups, item, window_seconds, similarity, identity_threshold
            )
            if not joined:
                open_groups.append([item])
        groups.extend(open_groups)

    merged = [_merge_occurrence(group) for group in groups]
    merged.sort(key=lambda item: item.score, reverse=True)
    return merged[:limit]


def _join_open_group(
    open_groups: list[list["ReidMatchItem"]],
    item: "ReidMatchItem",
    window_seconds: float,
    similarity: "_Similarity | None",
    identity_threshold: float,
) -> bool:
    """Adds the frame to the best still-open group at this camera, or reports that none fit.

    Complete linkage: the frame must resemble every member of the group, not just the newest.
    Comparing against the newest only is what single-linkage chaining is, and it is exactly how
    the earlier attempt failed -- consecutive frames are near-identical, so A joins B joins C all
    the way down a five-minute run whose two ends share a similarity of 0.30.
    """

    best_group: list[ReidMatchItem] | None = None
    best_weakest = identity_threshold
    for group in open_groups:
        if (item.captured_at - group[-1].captured_at).total_seconds() > window_seconds:
            continue  # this group has lapsed, and a later frame cannot revive it
        if similarity is None or identity_threshold <= 0:
            best_group = group
            break
        weakest = similarity.weakest_link(item.crop_id, [member.crop_id for member in group])
        if weakest is not None and weakest >= best_weakest:
            best_group, best_weakest = group, weakest
    if best_group is None:
        return False
    best_group.append(item)
    return True


class _Similarity:
    """Cosine between every pair of candidates, computed once as a single matrix product."""

    def __init__(self, vectors: dict[uuid.UUID, list[float]]) -> None:
        self._position = {crop_id: index for index, crop_id in enumerate(vectors)}
        # The ReID service returns L2-normalised vectors, so this product is the cosine.
        matrix = np.asarray(list(vectors.values()), dtype="float32")
        self._scores = matrix @ matrix.T

    def weakest_link(self, crop_id: uuid.UUID, members: list[uuid.UUID]) -> float | None:
        row = self._position.get(crop_id)
        columns = [self._position[member] for member in members if member in self._position]
        if row is None or len(columns) != len(members):
            return None  # an unknown vector makes the group's worst pair unknowable
        return float(self._scores[row, columns].min())


def _pairwise_similarity(
    vectors: dict[uuid.UUID, list[float]] | None,
) -> "_Similarity | None":
    if np is None or not vectors or len(vectors) < 2:
        return None
    dimensions = {len(vector) for vector in vectors.values()}
    if len(dimensions) != 1:
        return None
    return _Similarity(vectors)


def _merge_occurrence(group: list["ReidMatchItem"]) -> "ReidMatchItem":
    best = max(group, key=lambda item: item.score)
    stamps = [item.captured_at for item in group if item.captured_at is not None]
    return best.model_copy(
        update={
            "frame_count": len(group),
            "first_seen": min(stamps) if stamps else None,
            "last_seen": max(stamps) if stamps else None,
        }
    )


@dataclass(frozen=True)
class ReidMatch:
    crop_id: uuid.UUID
    score: float


@dataclass(frozen=True)
class BatchResult:
    indexed: list[uuid.UUID]
    skipped: list[uuid.UUID]
    failures: dict[uuid.UUID, str]


class ReidIndexService:
    """Keeps person-crop ReID vectors in Milvus and answers identity queries against them."""

    def __init__(self, db: Session, settings: Settings) -> None:
        self.db = db
        self.settings = settings
        self.reid = ReidEmbeddingService(settings)
        self.index = MilvusVectorIndex(settings)

    def is_enabled(self) -> bool:
        return (
            self.reid.is_enabled()
            and self.index.is_available()
            and self.index.reid_metric_supported
        )

    @property
    def fingerprint(self) -> str:
        """One string naming the vector space markers and status count against.

        Model, immutable checkpoint revision, dimension, preprocessing, logical Milvus
        namespace and collection all participate. Any switch changes the fingerprint, so old
        markers stop counting as coverage and their crops surface as pending again.
        """

        return "|".join(
            [
                self.settings.reid_model,
                self.settings.reid_checkpoint_revision,
                str(self.settings.reid_embedding_dim),
                self.settings.reid_preprocess_version,
                self.index.namespace_identity,
                self.index._collection_name(REID_OBJECT_TYPE),
            ]
        )

    # -- indexing ---------------------------------------------------------------------------

    def index_crop(self, crop: PersonCrop, *, flush: bool = True) -> bool:
        path = self._crop_path(crop)
        if path is None:
            return False
        vector = self.reid.embed_image(path)
        self.index.upsert_vector(REID_OBJECT_TYPE, crop.id, vector, flush=flush)
        self.record_indexed_crop(crop)
        return True

    def index_crops_batch(
        self,
        crops: list[PersonCrop],
        *,
        flush: bool = True,
        record: bool = True,
    ) -> BatchResult:
        """Embeds a batch over /embed-batch and upserts with a single flush.

        A crop whose file is gone is skipped, not failed. Only a bad-input rejection (400/422)
        splits into per-image retries, so one undecodable file cannot take the rest down with
        it. Everything else raises for the caller to retry the batch later: 401/403/404 mean
        the configuration is wrong and 429/5xx/transport mean re-sending the same batch is
        exactly the right response - splitting would just amplify them into N+1 failures.
        """

        readable: list[tuple[PersonCrop, Path]] = []
        skipped: list[uuid.UUID] = []
        for crop in crops:
            path = self._crop_path(crop)
            if path is None:
                skipped.append(crop.id)
            else:
                readable.append((crop, path))

        indexed: list[uuid.UUID] = []
        failures: dict[uuid.UUID, str] = {}
        if readable:
            try:
                vectors = self.reid.embed_images([path for _, path in readable])
                pairs = list(zip(readable, vectors, strict=True))
            except ReidRuntimeError as exc:
                if exc.status_code not in (400, 422):
                    raise
                pairs = []
                for crop, path in readable:
                    try:
                        pairs.append(((crop, path), self.reid.embed_image(path)))
                    except ReidRuntimeError as single_exc:
                        if single_exc.status_code not in (400, 422):
                            raise
                        failures[crop.id] = str(single_exc)
            for (crop, _), vector in pairs:
                self.index.upsert_vector(REID_OBJECT_TYPE, crop.id, vector, flush=False)
                if record:
                    self.record_indexed_crop(crop)
                indexed.append(crop.id)

        if indexed and flush:
            self.index.flush(REID_OBJECT_TYPE)
        return BatchResult(indexed=indexed, skipped=skipped, failures=failures)

    def record_indexed_crop(self, crop: PersonCrop) -> None:
        """Stage the durable SQL marker and observation for an already-flushed vector."""

        self._lock_marker_writes_in_session()
        self._record(crop.id)
        ObservationIndexService(self.db, self.settings).upsert_crop(crop)

    def _lock_marker_writes_in_session(self) -> None:
        """Serialize marker + observation upserts across API processes.

        PostgreSQL locks the dedicated row; SQLite's UPDATE obtains its database write
        reservation. The lock is acquired only after external vector I/O has completed, so it
        protects a short SQL transaction and never surrounds model or Milvus latency.
        """

        result = self.db.execute(
            update(VectorIndexCapacityLock)
            .where(VectorIndexCapacityLock.target == "reid_marker")
            .values(revision=VectorIndexCapacityLock.revision + 1)
            .execution_options(synchronize_session=False)
        )
        if result.rowcount != 1:
            raise RuntimeError("ReID marker lock is missing; run init_db()")

    def backfill(self, limit: int) -> dict[str, object]:
        """Indexes the most recent crops without a ReID vector for the current model."""

        crops = list(self.db.scalars(self._unindexed_query(limit)))
        indexed = 0
        skipped = 0
        failed = 0
        unprocessed = 0
        errors: list[str] = []
        indexed_crops: list[PersonCrop] = []
        for start in range(0, len(crops), BATCH_SIZE):
            chunk = crops[start : start + BATCH_SIZE]
            try:
                result = self.index_crops_batch(chunk, flush=False, record=False)
            except (ReidRuntimeError, VectorIndexError) as exc:
                # Service or Milvus down mid-backfill: report and stop asking.
                errors.append(f"batch at {start}: {exc}")
                unprocessed += len(crops) - start
                break
            indexed += len(result.indexed)
            indexed_ids = set(result.indexed)
            indexed_crops.extend(crop for crop in chunk if crop.id in indexed_ids)
            skipped += len(result.skipped)
            failed += len(result.failures)
            errors.extend(f"{crop_id}: {message}" for crop_id, message in result.failures.items())
        if indexed:
            self.index.flush(REID_OBJECT_TYPE)
            for crop in indexed_crops:
                self.record_indexed_crop(crop)
            self.db.commit()
        # Invariant: seen == indexed + skipped + failed + unprocessed, whatever failed.
        return {
            "requested": limit,
            "seen": len(crops),
            "indexed": indexed,
            "skipped": skipped,
            "failed": failed,
            "unprocessed": unprocessed,
            "errors": errors[:20],
        }

    def pending_count(self, cap: int | None = 1000) -> int:
        query = self._unindexed_query(cap)
        if cap is None:
            count_query = select(func.count()).select_from(query.order_by(None).subquery())
            return int(self.db.scalar(count_query) or 0)
        return len(list(self.db.scalars(query)))

    def indexed_count(self) -> int:
        return int(
            self.db.scalar(
                select(func.count())
                .select_from(VLEmbedding)
                .where(
                    VLEmbedding.object_type == REID_OBJECT_TYPE,
                    VLEmbedding.embedding_model == self.fingerprint,
                    VLEmbedding.embedding_dim == self.settings.reid_embedding_dim,
                )
            )
            or 0
        )

    def candidate_pool_limit(self) -> int:
        return max(
            COLLAPSE_CANDIDATE_LIMIT,
            min(self.indexed_count(), self.settings.reid_candidate_pool_max),
        )

    def _unindexed_query(self, limit: int | None):
        # A marker only counts for the complete current vector-space identity; any checkpoint
        # or Milvus namespace switch makes previously indexed crops pending again.
        indexed = select(VLEmbedding.object_id).where(
            VLEmbedding.object_type == REID_OBJECT_TYPE,
            VLEmbedding.embedding_model == self.fingerprint,
            VLEmbedding.embedding_dim == self.settings.reid_embedding_dim,
        )
        query = (
            select(PersonCrop)
            .where(PersonCrop.crop_url.is_not(None))
            .where(PersonCrop.id.not_in(indexed))
            .order_by(PersonCrop.created_at.desc())
        )
        return query.limit(limit) if limit is not None else query

    def _record(self, crop_id: uuid.UUID) -> None:
        existing = list(
            self.db.scalars(
                select(VLEmbedding).where(
                    VLEmbedding.object_type == REID_OBJECT_TYPE,
                    VLEmbedding.object_id == crop_id,
                ).order_by(VLEmbedding.created_at.desc(), VLEmbedding.id.desc())
            )
        )
        if existing:
            marker = existing[0]
            marker.embedding = None
            marker.embedding_model = self.fingerprint
            marker.embedding_dim = self.settings.reid_embedding_dim
            self.db.add(marker)
            for duplicate in existing[1:]:
                self.db.delete(duplicate)
            return
        self.db.add(
            VLEmbedding(
                object_type=REID_OBJECT_TYPE,
                object_id=crop_id,
                embedding=None,
                embedding_model=self.fingerprint,
                embedding_dim=self.settings.reid_embedding_dim,
            )
        )

    # -- querying ---------------------------------------------------------------------------

    def search_by_image(
        self,
        image_path: Path,
        top_k: int | None = None,
        *,
        min_score: float | None = None,
    ) -> list[ReidMatch]:
        limit = top_k or self.settings.reid_search_top_k
        vector = self.reid.embed_image(image_path)
        return self._search(vector, limit, min_score=min_score)

    def search_by_crop(
        self,
        crop: PersonCrop,
        top_k: int | None = None,
        *,
        min_score: float | None = None,
    ) -> list[ReidMatch]:
        path = self._crop_path(crop)
        if path is None:
            return []
        return self.search_by_image(path, top_k, min_score=min_score)

    def query_tracklet(self, crop: PersonCrop) -> list[PersonCrop]:
        """Build a small, temporally diverse gallery for an observation-table query.

        Camera and time only produce a crowd, not an identity. Every neighbour therefore has to
        clear a cosine threshold against the selected crop's stored ReID vector. Any missing
        metadata, vector-index outage or absent local file returns the source crop alone.
        """

        maximum = min(
            self.settings.reid_query_tracklet_frames,
            self.settings.reid_gallery_size,
        )
        window = self.settings.reid_query_tracklet_window_seconds
        if maximum <= 1 or window <= 0 or crop.camera_id is None or crop.captured_at is None:
            return [crop]
        start = crop.captured_at - timedelta(seconds=window)
        end = crop.captured_at + timedelta(seconds=window)
        nearby = list(
            self.db.scalars(
                select(PersonCrop)
                .where(
                    PersonCrop.camera_id == crop.camera_id,
                    PersonCrop.captured_at >= start,
                    PersonCrop.captured_at <= end,
                    PersonCrop.crop_url.is_not(None),
                )
                .order_by(PersonCrop.captured_at.asc())
                .limit(self.settings.reid_query_tracklet_candidate_limit)
            )
        )
        if not any(item.id == crop.id for item in nearby):
            nearby.append(crop)
        try:
            vectors = self.index.fetch_vectors(REID_OBJECT_TYPE, [item.id for item in nearby])
        except VectorIndexError:
            return [crop]
        query_vector = vectors.get(crop.id)
        if not query_vector:
            return [crop]

        eligible: list[tuple[PersonCrop, float]] = []
        for item in nearby:
            path = self._crop_path(item)
            vector = vectors.get(item.id)
            if path is None or vector is None:
                continue
            similarity = _cosine(query_vector, vector)
            if (
                item.id == crop.id
                or similarity >= self.settings.reid_query_tracklet_identity_threshold
            ):
                eligible.append((item, similarity))
        if len(eligible) <= 1:
            return [crop]

        # Keep the selected observation, then take the best-quality crop from temporal buckets.
        # This prevents six nearly identical consecutive frames from pretending to be six votes.
        selected: dict[uuid.UUID, PersonCrop] = {crop.id: crop}
        others = [(item, score) for item, score in eligible if item.id != crop.id]
        span = max(1.0, (end - start).total_seconds())
        buckets: dict[int, tuple[float, PersonCrop]] = {}
        bucket_count = max(1, maximum - 1)
        for item, similarity in others:
            assert item.captured_at is not None
            position = (item.captured_at - start).total_seconds() / span
            bucket = min(bucket_count - 1, max(0, int(position * bucket_count)))
            bbox = item.bbox if isinstance(item.bbox, dict) else {}
            confidence = _bounded_number(bbox.get("confidence"), default=0.5)
            area = _positive_number(bbox.get("width")) * _positive_number(
                bbox.get("height")
            )
            # Detection confidence is the reliable quality signal; log area only breaks close
            # ties without allowing a huge but blurry crop to beat a much cleaner detection.
            quality = 0.70 * similarity + 0.25 * confidence + 0.05 * min(
                1.0, math.log1p(area) / 12.0
            )
            current = buckets.get(bucket)
            if current is None or quality > current[0]:
                buckets[bucket] = (quality, item)
        for _quality, item in sorted(buckets.values(), key=lambda pair: pair[0], reverse=True):
            selected[item.id] = item
            if len(selected) >= maximum:
                break
        return sorted(
            selected.values(),
            key=lambda item: (item.id != crop.id, item.captured_at or crop.captured_at),
        )

    def search_by_crop_gallery(
        self,
        crops: list[PersonCrop],
        top_k: int | None = None,
        *,
        min_score: float | None = None,
    ) -> list[ReidMatch]:
        paths = [path for crop in crops if (path := self._crop_path(crop)) is not None]
        if not paths:
            return []
        if len(paths) == 1:
            return self.search_by_image(paths[0], top_k, min_score=min_score)
        scores = self.gallery_matches(paths, top_k, min_score=min_score)
        floor = min_score
        if floor is None:
            floor = min(self.settings.reid_min_score, self.settings.reid_min_score_cross_camera)
        return [
            ReidMatch(crop_id=crop_id, score=score)
            for crop_id, score in sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
            if score >= floor
        ]

    def gallery_matches(
        self,
        gallery_paths: list[Path],
        top_k: int | None = None,
        *,
        min_score: float | None = None,
        deadline: float | None = None,
        warnings: list[str] | None = None,
    ) -> dict[uuid.UUID, float]:
        """Scores candidates against a person's whole gallery, not one lucky frame.

        Each gallery image votes, and a candidate's score is the sum of its best few votes over a
        fixed denominator, so a missing vote counts as zero. Averaging over however many votes a
        candidate happened to get does not work: one frame matching at 0.99 would outrank an
        identity matched three times around 0.85, which is the opposite of what a gallery is for.
        """

        if not gallery_paths:
            return {}
        gallery = gallery_paths[: self.settings.reid_gallery_size]
        limit = top_k or self.settings.reid_search_top_k
        votes: dict[uuid.UUID, list[float]] = {}
        try:
            vectors = (
                self.reid.embed_images(gallery, deadline=deadline)
                if deadline is not None
                else self.reid.embed_images(gallery)
            )
        except ReidRuntimeError as exc:
            if exc.status_code not in (400, 422):
                raise
            vectors = []
            rejected = 0
            for path in gallery:
                try:
                    vectors.append(self.reid.embed_image(path, deadline=deadline))
                except ReidRuntimeError as single_exc:
                    if single_exc.status_code not in (400, 422):
                        raise
                    rejected += 1
            if rejected and warnings is not None:
                warnings.append(f"已跳过 {rejected} 个无法解码的 ReID gallery crop")
            if not vectors:
                raise ReidRuntimeError(
                    "No decodable ReID gallery crops remain",
                    status_code=422,
                ) from exc
        for vector in vectors:
            index = self.index
            if deadline is not None:
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise ReidRuntimeError("ReID gallery search deadline exceeded")
                bounded_settings = self.settings.model_copy(
                    update={
                        "milvus_timeout_seconds": max(
                            0.1,
                            min(self.settings.milvus_timeout_seconds, remaining),
                        )
                    }
                )
                index = MilvusVectorIndex(bounded_settings)
            for match in self._search(vector, limit, index=index, min_score=min_score):
                votes.setdefault(match.crop_id, []).append(match.score)

        # Never divide by more votes than the gallery could possibly have cast.
        denominator = min(self.settings.reid_gallery_top_k, len(vectors))
        aggregated: dict[uuid.UUID, float] = {}
        for crop_id, scores in votes.items():
            scores.sort(reverse=True)
            aggregated[crop_id] = sum(scores[:denominator]) / denominator
        return aggregated

    def _search(
        self,
        vector: list[float],
        top_k: int,
        *,
        index: MilvusVectorIndex | None = None,
        min_score: float | None = None,
    ) -> list[ReidMatch]:
        hits = (index or self.index).search_vector(REID_OBJECT_TYPE, vector, top_k) or []
        # The looser of the two bars: which one a hit is actually held to depends on its camera,
        # and this layer has only crop ids. Dropping at the strict bar here would discard every
        # cross-camera match before anything could tell it apart from a same-camera one. Callers
        # that rank rather than decide -- the per-camera links -- pass their own floor, because
        # any bar at all would hide the very cameras they exist to report on.
        floor = (
            min(self.settings.reid_min_score, self.settings.reid_min_score_cross_camera)
            if min_score is None
            else min_score
        )
        return [
            ReidMatch(crop_id=hit.object_id, score=float(hit.score))
            for hit in hits
            if hit.score >= floor
        ]

    def _crop_path(self, crop: PersonCrop) -> Path | None:
        prefix = "/data/"
        if not crop.crop_url or not crop.crop_url.startswith(prefix):
            return None
        path = self.settings.data_dir / Path(crop.crop_url.removeprefix(prefix))
        return path if path.exists() else None
