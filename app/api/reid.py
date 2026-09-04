import logging
import tempfile
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from sqlalchemy import select

from app.api.deps import AppSettings, DBSession
from app.models.media import PersonCrop, VideoStream
from app.schemas.media import SearchFilters
from app.schemas.reid import (
    ReidCameraLink,
    ReidLinkResponse,
    ReidMatchItem,
    ReidRebuildResponse,
    ReidSearchResponse,
    ReidStatusResponse,
)
from app.services.faces import face_runtime_status
from app.services.observation_index import ObservationIndexService
from app.services.reid import ReidRuntimeError
from app.services.reid_attributes import aggregate_reid_attributes, compare_reid_attributes
from app.services.reid_fusion import (
    annotate_fusion_decision,
    enrich_face_evidence,
    fusion_rank,
    reject_reliable_face_mismatches,
)
from app.services.reid_index import (
    REID_OBJECT_TYPE,
    ReidIndexService,
    ReidMatch,
    collapse_occurrences,
)
from app.services.structured_attributes import StructuredAttributeService
from app.services.vector_index import VectorIndexError
from app.services.vlm import VLMRuntimeError, VLMStructuredAnalysisService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/reid", tags=["reid"])

UploadImage = Annotated[UploadFile, File(...)]


def _service(db: DBSession, settings: AppSettings) -> ReidIndexService:
    service = ReidIndexService(db, settings)
    if not service.is_enabled():
        if not service.index.reid_metric_supported:
            raise HTTPException(
                status_code=503,
                detail="ReID requires MILVUS_METRIC_TYPE=COSINE",
            )
        raise HTTPException(
            status_code=503,
            detail="ReID is not configured; set REID_ENABLED, REID_SERVICE_URL and MILVUS_ENABLED",
        )
    return service


def _to_items(
    db: DBSession,
    settings: AppSettings,
    matches: list[ReidMatch],
) -> list[ReidMatchItem]:
    """Decorates bare crop ids with the observation row, in one query rather than per match."""

    if not matches:
        return []
    rows = ObservationIndexService(db, settings).rows_by_crop_ids(
        [match.crop_id for match in matches],
        SearchFilters(),
    )
    # The observation row is a cache written just behind ingest, so the freshest crops -- the ones
    # a live search is actually asking about -- routinely have no row yet. Reporting them with no
    # camera and no timestamp is worse than a stale name: they cannot be placed on a camera, and
    # collapsing has nothing to group them by, so each frame comes back as its own visit. The crop
    # already carries both facts.
    pending = [match.crop_id for match in matches if match.crop_id not in rows]
    crops = (
        {crop.id: crop for crop in db.scalars(select(PersonCrop).where(PersonCrop.id.in_(pending)))}
        if pending
        else {}
    )
    # One extra read for the whole page rather than one per match. The observation row does not
    # carry attributes, and stature lives there.
    stature = {
        crop_id: _stature_percentile(attributes)
        for crop_id, attributes in db.execute(
            select(PersonCrop.id, PersonCrop.attributes).where(
                PersonCrop.id.in_([match.crop_id for match in matches])
            )
        )
    }
    camera_ids = {crop.camera_id for crop in crops.values() if crop.camera_id}
    streams = (
        {
            stream.camera_id: stream
            for stream in db.scalars(
                select(VideoStream).where(VideoStream.camera_id.in_(camera_ids))
            )
        }
        if camera_ids
        else {}
    )

    items: list[ReidMatchItem] = []
    for match in matches:
        row = rows.get(match.crop_id)
        if row is not None:
            items.append(
                ReidMatchItem(
                    crop_id=match.crop_id,
                    score=match.score,
                    image_id=row.image_id,
                    crop_url=row.crop_url,
                    image_url=row.image_url,
                    captured_at=row.captured_at,
                    camera_id=row.camera_id,
                    camera_name=row.camera_name,
                    location_id=row.location_id,
                    location_name=row.location_name,
                    person_id=row.person_id,
                    person_name=row.person_name,
                    stature_percentile=stature.get(match.crop_id),
                )
            )
            continue
        crop = crops.get(match.crop_id)
        stream = streams.get(crop.camera_id) if crop else None
        items.append(
            ReidMatchItem(
                crop_id=match.crop_id,
                score=match.score,
                image_id=crop.image_id if crop else None,
                crop_url=crop.crop_url if crop else None,
                captured_at=crop.captured_at if crop else None,
                camera_id=crop.camera_id if crop else None,
                camera_name=stream.name if stream else None,
                location_id=crop.location_id if crop else None,
                location_name=stream.location_name if stream else None,
                person_id=crop.person_id if crop else None,
                stature_percentile=stature.get(match.crop_id),
            )
        )
    return items



def _stature_percentile(attributes: dict | None) -> int | None:
    if not isinstance(attributes, dict):
        return None
    stature = attributes.get("stature")
    if not isinstance(stature, dict):
        return None
    value = stature.get("percentile")
    return int(value) if isinstance(value, (int, float)) else None


def _stature_bonus(
    item: ReidMatchItem,
    query_percentile: int | None,
    query_camera: uuid.UUID | None,
    weight: float,
) -> float:
    """A bounded nudge from height agreement, and zero whenever height cannot speak.

    Applied only across cameras. Within one camera the embedding already separates people at 0.92
    and above, so height adds nothing there and its own error would only blur a decided ranking.

    Height is compared as a rank inside each camera's own crowd, never as the raw ratio: the two
    doorways here spread the same population over 0.935-1.062 and 0.882-1.150, so a raw 1.06 is
    the 90th percentile at one and around the 65th at the other.
    """

    if weight <= 0 or query_percentile is None or item.stature_percentile is None:
        return 0.0
    if query_camera is not None and item.camera_id == query_camera:
        return 0.0
    gap = abs(item.stature_percentile - query_percentile)
    # Agreement runs 1 at identical rank down to 0 at fifty points apart, which is where the
    # measured same-person gap (23) and the random gap (34) have both been left behind.
    agreement = max(0.0, 1.0 - gap / 50.0)
    item.stature_agreement = round(agreement, 3)
    return weight * (agreement * 2.0 - 1.0)


def _admit(
    items: list[ReidMatchItem],
    settings: AppSettings,
    query_camera: uuid.UUID | None,
) -> list[ReidMatchItem]:
    """Holds same-camera matches to the strict bar and everything else to the cross-camera one.

    One bar cannot serve both. Measured on this deployment, a same-camera best match sits near
    0.89 while the strongest of 24378 temporally plausible cross-camera pairs reached 0.481 --
    people who look like the same person crossing between two doors thirty seconds apart score
    in the 0.43 to 0.48 band. A single 0.5 cut answers "who else was at this door" perfectly and
    "where else did they go" never.

    An uploaded photo has no camera, so nothing is same-camera and every match is held to the
    cross-camera bar.
    """

    return [
        item
        for item in items
        if item.score
        >= (
            settings.reid_min_score
            if query_camera is not None and item.camera_id == query_camera
            else settings.reid_min_score_cross_camera
        )
    ]


def _reserve_camera_slots(
    items: list[ReidMatchItem],
    limit: int,
    quota: int,
    key=None,
) -> list[ReidMatchItem]:
    """Guarantees each camera a few slots without reordering what survives.

    Sorting purely by score hands every slot to the camera the query came from, because a
    same-camera visit outscores a genuine crossing two to one. The seats come out of whichever
    camera has the most, so the strongest matches are never the ones displaced.
    """

    key = key or (lambda item: item.score)
    ranked = sorted(items, key=key, reverse=True)
    if quota <= 0 or len(ranked) <= limit:
        return ranked[:limit]

    chosen = ranked[:limit]
    for candidate in ranked[limit:]:
        seats = Counter(item.camera_id for item in chosen)
        if seats[candidate.camera_id] >= quota:
            continue
        crowded, count = seats.most_common(1)[0]
        if count <= quota:
            break  # nobody is over-represented, so nothing can be given up
        victim = min(
            (item for item in chosen if item.camera_id == crowded),
            key=key,
        )
        chosen.remove(victim)
        chosen.append(candidate)
    return sorted(chosen, key=key, reverse=True)


def _collapse(
    db: DBSession,
    settings: AppSettings,
    service: ReidIndexService,
    matches: list[ReidMatch],
    visits: int,
    query_camera: uuid.UUID | None = None,
    query_stature: int | None = None,
    query_attributes: dict | None = None,
    query_crop: PersonCrop | None = None,
    query_crops: list[PersonCrop] | None = None,
    query_image_path: Path | None = None,
) -> list[ReidMatchItem]:
    """Turns raw hits into visits, fetching the vectors the identity test needs."""

    # Camera metadata is not stored in the current Milvus schema, so the raw pool is intentionally
    # deep. Drop scores that cannot pass either calibrated threshold before loading SQL metadata.
    score_floor = min(settings.reid_min_score, settings.reid_min_score_cross_camera)
    matches = [match for match in matches if match.score >= score_floor]
    items = _admit(
        _to_items(db, settings, matches),
        settings,
        query_camera,
    )
    items, attribute_bonus = _filter_by_attributes(
        db, settings, items, query_attributes
    )
    # Scored after admission, never before: the two thresholds were measured against the
    # embedding's own scores, and moving those scores would quietly move the thresholds too.
    ranking = {
        item.crop_id: item.score
        + _stature_bonus(item, query_stature, query_camera, settings.reid_stature_weight)
        + attribute_bonus.get(item.crop_id, 0.0)
        for item in items
    }
    vectors: dict[uuid.UUID, list[float]] = {}
    if settings.reid_collapse_window_seconds > 0 and settings.reid_collapse_identity_threshold > 0:
        try:
            vectors = service.index.fetch_vectors(
                REID_OBJECT_TYPE, [item.crop_id for item in items]
            )
        except VectorIndexError:
            # Grouping without the identity test is worse, not broken; a search that still
            # answers beats one that 503s because a secondary read failed.
            logger.warning("ReID identity grouping degraded: vector fetch failed", exc_info=True)
    # Do not globally truncate before the camera quota runs. A camera-domain shift can otherwise
    # fill the first N visits with one doorway and erase every other camera a second time.
    grouped = collapse_occurrences(
        items,
        settings.reid_collapse_window_seconds,
        len(items),
        vectors=vectors,
        identity_threshold=settings.reid_collapse_identity_threshold,
    )
    enrich_face_evidence(
        db,
        settings,
        query_crop,
        grouped,
        query_image_path=query_image_path,
        query_crops=query_crops,
    )
    grouped = _reject_attribute_conflicts(
        grouped,
        settings,
        query_captured_at=query_crop.captured_at if query_crop is not None else None,
    )
    grouped = reject_reliable_face_mismatches(
        grouped,
        settings.reid_face_hard_reject_threshold,
    )
    fused_ranking = {
        item.crop_id: fusion_rank(
            item,
            ranking.get(item.crop_id, item.score),
        )
        for item in grouped
    }
    for item in grouped:
        annotate_fusion_decision(
            item,
            ranking.get(item.crop_id, item.score),
            query_camera=query_camera,
            chance_ceiling=settings.reid_chance_ceiling,
        )
    return _reserve_camera_slots(
        grouped,
        visits,
        settings.reid_camera_quota,
        key=lambda item: fused_ranking.get(
            item.crop_id,
            fusion_rank(item, item.score),
        ),
    )


def _filter_by_attributes(
    db: DBSession,
    settings: AppSettings,
    items: list[ReidMatchItem],
    query_attributes: dict | None,
) -> tuple[list[ReidMatchItem], dict[uuid.UUID, float]]:
    if not settings.reid_attribute_filter_enabled or not query_attributes or not items:
        return items, {}
    attributes = {
        crop_id: value
        for crop_id, value in db.execute(
            select(PersonCrop.id, PersonCrop.attributes).where(
                PersonCrop.id.in_([item.crop_id for item in items])
            )
        )
    }
    kept: list[ReidMatchItem] = []
    bonuses: dict[uuid.UUID, float] = {}
    for item in items:
        compatibility = compare_reid_attributes(
            query_attributes,
            attributes.get(item.crop_id),
            min_confidence=settings.reid_attribute_min_confidence,
        )
        item.attribute_agreement = compatibility.agreement
        item.attribute_matches = list(compatibility.matched)
        item.attribute_conflicts = list(compatibility.conflicted)
        item.attribute_comparable_count = compatibility.compared_count
        item.attribute_match_count = compatibility.match_count
        item.attribute_conflict_count = compatibility.conflict_count
        item.attribute_evidence_weight = round(compatibility.evidence_weight, 4)
        item.attribute_conflict_weight = round(compatibility.conflict_weight, 4)
        kept.append(item)
        if compatibility.agreement is not None:
            coverage = min(
                compatibility.evidence_weight / settings.reid_attribute_full_weight,
                1.0,
            )
            bonuses[item.crop_id] = (
                settings.reid_attribute_weight
                * (compatibility.agreement * 2.0 - 1.0)
                * coverage
            )
    return kept, bonuses


def _reject_attribute_conflicts(
    items: list[ReidMatchItem],
    settings: AppSettings,
    *,
    query_captured_at: datetime | None = None,
) -> list[ReidMatchItem]:
    """Apply the hard label gate only after face enrichment.

    A reliable matching face is identity evidence and must be allowed to overrule changed clothes
    or a pair of VLM mistakes. Clothing is allowed to reject only within a short configurable time
    window and only when enough tags are present; across days it is expected to change and remains a
    continuous ranking signal.
    """

    def inside_hard_filter_window(item: ReidMatchItem) -> bool:
        if (
            settings.reid_attribute_hard_filter_window_hours <= 0
            or query_captured_at is None
            or item.captured_at is None
        ):
            return False
        delta_hours = abs((item.captured_at - query_captured_at).total_seconds()) / 3600.0
        return delta_hours <= settings.reid_attribute_hard_filter_window_hours

    return [
        item
        for item in items
        if item.face_match is True
        or not inside_hard_filter_window(item)
        or item.attribute_comparable_count < settings.reid_attribute_hard_conflicts
        or len(item.attribute_conflicts) < settings.reid_attribute_hard_conflicts
        or (item.attribute_conflict_weight or 0.0)
        < settings.reid_attribute_hard_confidence
    ]


def _query_image_attributes(path: Path, settings: AppSettings) -> dict | None:
    if not settings.reid_attribute_filter_enabled or not settings.reid_attribute_analyze_upload:
        return None
    service = VLMStructuredAnalysisService(settings)
    if not service.is_enabled():
        return None
    try:
        return service.analyze_image(path, object_type="person")
    except VLMRuntimeError:
        logger.warning("ReID upload attribute analysis degraded to vector-only", exc_info=True)
        return None


def _query_crop_attributes(
    db: DBSession,
    settings: AppSettings,
    crop: PersonCrop,
) -> dict | None:
    attributes = crop.attributes
    if not settings.reid_attribute_filter_enabled or not settings.reid_attribute_enrich_query_crop:
        return attributes
    if attributes and attributes.get("source") != "cv_tone":
        return attributes
    if not VLMStructuredAnalysisService(settings).is_enabled():
        return attributes
    try:
        return StructuredAttributeService(db, settings).analyze_person_crop(crop, persist=True)
    except VLMRuntimeError:
        logger.warning("ReID query-crop attribute enrichment degraded", exc_info=True)
        return attributes


def _query_tracklet_attributes(
    db: DBSession,
    settings: AppSettings,
    source: PersonCrop,
    gallery: list[PersonCrop],
) -> dict | None:
    source_attributes = _query_crop_attributes(db, settings, source)
    samples = [
        source_attributes if crop.id == source.id else crop.attributes
        for crop in gallery
    ]
    return aggregate_reid_attributes(
        samples,
        min_confidence=settings.reid_attribute_min_confidence,
    )


def _visit_limits(
    settings: AppSettings,
    service: ReidIndexService,
    top_k: int,
) -> tuple[int, int]:
    """How many candidates to pull, and how many visits to return after collapsing."""

    visits = top_k or settings.reid_search_top_k
    if settings.reid_collapse_window_seconds <= 0:
        return visits, visits
    return service.candidate_pool_limit(), visits


@router.get("/status", response_model=ReidStatusResponse)
def reid_status(db: DBSession, settings: AppSettings) -> ReidStatusResponse:
    service = ReidIndexService(db, settings)
    configured = service.reid.is_enabled() and bool(settings.milvus_enabled)
    service_ok, probe_error = service.reid.probe() if configured else (False, None)
    milvus_configured = bool(settings.milvus_enabled)
    milvus_ok, milvus_error = (
        service.index.probe() if milvus_configured else (False, "MILVUS_ENABLED is not set")
    )
    milvus_in_cooldown = milvus_configured and not service.index.is_available()
    metric_error = (
        None
        if service.index.reid_metric_supported
        else "ReID requires MILVUS_METRIC_TYPE=COSINE"
    )
    indexed = service.indexed_count()
    face_status = face_runtime_status(settings)
    last_error = metric_error or probe_error or (milvus_error if milvus_configured else None)
    return ReidStatusResponse(
        enabled=configured,
        ready=configured and service_ok and milvus_ok and metric_error is None,
        reid_service_ok=service_ok,
        milvus_configured=milvus_configured,
        milvus_ok=milvus_ok,
        milvus_in_cooldown=milvus_in_cooldown,
        last_error=last_error,
        service_url=settings.reid_service_url,
        model=settings.reid_model,
        checkpoint_revision=settings.reid_checkpoint_revision,
        embedding_dim=settings.reid_embedding_dim,
        preprocess_version=settings.reid_preprocess_version,
        milvus_namespace=service.index.namespace_identity,
        index_fingerprint=service.fingerprint,
        indexed_crops=int(indexed or 0),
        pending_crops=service.pending_count(cap=None),
        min_score=settings.reid_min_score,
        attribute_filter_enabled=settings.reid_attribute_filter_enabled,
        attribute_min_confidence=settings.reid_attribute_min_confidence,
        attribute_hard_conflicts=settings.reid_attribute_hard_conflicts,
        attribute_hard_confidence=settings.reid_attribute_hard_confidence,
        attribute_full_weight=settings.reid_attribute_full_weight,
        attribute_hard_filter_window_hours=settings.reid_attribute_hard_filter_window_hours,
        face_priority_enabled=settings.reid_face_priority_enabled,
        face_priority_ready=(
            settings.reid_face_priority_enabled and face_status.ready
        ),
        face_priority_error=face_status.error,
        face_provider=face_status.provider,
        face_model=face_status.model,
        face_device=face_status.device,
        face_candidate_limit=settings.reid_face_candidate_limit,
        face_min_quality=settings.reid_face_min_quality,
        face_strong_reliability=settings.reid_face_strong_reliability,
    )


@router.post("/search", response_model=ReidSearchResponse)
def reid_search(
    file: UploadImage,
    db: DBSession,
    settings: AppSettings,
    top_k: int = Query(default=0, ge=0, le=200),
) -> ReidSearchResponse:
    """Finds the same person across cameras from an uploaded body crop."""

    service = _service(db, settings)
    candidates, visits = _visit_limits(settings, service, top_k)
    suffix = Path(file.filename or "query.jpg").suffix or ".jpg"
    with tempfile.TemporaryDirectory(prefix="sightindex-reid-") as temp_dir:
        query_path = Path(temp_dir) / f"query{suffix}"
        query_path.write_bytes(file.file.read())
        try:
            matches = service.search_by_image(query_path, candidates)
            query_attributes = _query_image_attributes(query_path, settings)
        except ReidRuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except VectorIndexError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return ReidSearchResponse(
            items=_collapse(
                db,
                settings,
                service,
                matches,
                visits,
                query_attributes=query_attributes,
                query_image_path=query_path,
            ),
            model=settings.reid_model,
            min_score=settings.reid_min_score,
            collapse_window_seconds=settings.reid_collapse_window_seconds,
        )


@router.post("/crops/{crop_id}/similar", response_model=ReidSearchResponse)
def reid_similar_to_crop(
    crop_id: uuid.UUID,
    db: DBSession,
    settings: AppSettings,
    top_k: int = Query(default=0, ge=0, le=200),
) -> ReidSearchResponse:
    service = _service(db, settings)
    candidates, visits = _visit_limits(settings, service, top_k)
    crop = db.get(PersonCrop, crop_id)
    if crop is None:
        raise HTTPException(status_code=404, detail="Crop not found")
    try:
        query_crops = service.query_tracklet(crop)
        matches = service.search_by_crop_gallery(query_crops, candidates)
        query_attributes = _query_tracklet_attributes(db, settings, crop, query_crops)
    except (ReidRuntimeError, VectorIndexError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    # Drop the query itself before collapsing, not after: it scores 1.0 against its own vector
    # and would win the visit it belongs to, reporting the query back as its own best match.
    query_crop_ids = {query_crop.id for query_crop in query_crops}
    matches = [match for match in matches if match.crop_id not in query_crop_ids]
    return ReidSearchResponse(
        items=_collapse(
            db,
            settings,
            service,
            matches,
            visits,
            query_camera=crop.camera_id,
            query_stature=_stature_percentile(crop.attributes),
            query_attributes=query_attributes,
            query_crop=crop,
            query_crops=query_crops,
        ),
        model=settings.reid_model,
        min_score=settings.reid_min_score,
        collapse_window_seconds=settings.reid_collapse_window_seconds,
        query_mode="tracklet" if len(query_crops) > 1 else "single_frame",
        query_frame_count=len(query_crops),
    )


@router.post("/crops/{crop_id}/links", response_model=ReidLinkResponse)
def reid_camera_links(
    crop_id: uuid.UUID,
    db: DBSession,
    settings: AppSettings,
) -> ReidLinkResponse:
    """The likeliest appearance of this person at every other camera, threshold or not.

    Deciding "match / no match" cannot be done well here: a genuine crossing scores 0.43 to 0.48
    on this footage and coincidence reaches 0.44, so the two overlap and any bar either invents
    crossings or hides them. Ranking is a different question and the model answers it fine -- of
    all the crops at that other door, which is most like this one. That is what a person tracing
    a route actually needs, and the score comes with it so they can weigh it.
    """

    service = _service(db, settings)
    crop = db.get(PersonCrop, crop_id)
    if crop is None:
        raise HTTPException(status_code=404, detail="Crop not found")
    try:
        # No floor: a camera's best candidate is the answer even when it is a weak one, and the
        # flag says so. Filtering here would silently drop whole cameras from the trace.
        query_crops = service.query_tracklet(crop)
        matches = service.search_by_crop_gallery(
            query_crops,
            service.candidate_pool_limit(),
            min_score=0.0,
        )
    except (ReidRuntimeError, VectorIndexError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    # This is the one place height earns its keep. Nothing here is a threshold decision, and
    # picking one crop out of a whole camera's worth is exactly the close call the embedding
    # cannot settle on its own: a real crossing and a coincidence both score in the 0.43-0.48
    # band. In the search endpoint the same term does nothing, because the 0.45 admission bar
    # and the per-camera quota leave at most one cross-camera row with nothing to reorder.
    query_attributes = _query_tracklet_attributes(db, settings, crop, query_crops)
    query_stature = _stature_percentile(crop.attributes)
    best: dict[
        uuid.UUID | None,
        tuple[tuple[int, float, float], ReidMatchItem],
    ] = {}
    linked_items, attribute_bonus = _filter_by_attributes(
        db, settings, _to_items(db, settings, matches), query_attributes
    )
    linked_items = [
        item
        for item in linked_items
        if item.crop_id not in {query_crop.id for query_crop in query_crops}
        and item.camera_id is not None
        and item.camera_id != crop.camera_id
    ]
    enrich_face_evidence(db, settings, crop, linked_items, query_crops=query_crops)
    linked_items = _reject_attribute_conflicts(
        linked_items,
        settings,
        query_captured_at=crop.captured_at,
    )
    for item in linked_items:
        appearance_score = item.score + _stature_bonus(
            item, query_stature, crop.camera_id, settings.reid_stature_weight
        ) + attribute_bonus.get(item.crop_id, 0.0)
        ranking = fusion_rank(item, appearance_score)
        annotate_fusion_decision(
            item,
            appearance_score,
            query_camera=crop.camera_id,
            chance_ceiling=settings.reid_chance_ceiling,
            is_camera_link=True,
        )
        current = best.get(item.camera_id)
        if current is None or ranking > current[0]:
            best[item.camera_id] = (ranking, item)

    links = [
        ReidCameraLink(
            camera_id=item.camera_id,
            camera_name=item.camera_name,
            location_name=item.location_name,
            crop_id=item.crop_id,
            crop_url=item.crop_url,
            score=item.score,
            stature_agreement=item.stature_agreement,
            attribute_agreement=item.attribute_agreement,
            attribute_matches=item.attribute_matches,
            attribute_conflicts=item.attribute_conflicts,
            attribute_evidence_weight=item.attribute_evidence_weight,
            attribute_conflict_weight=item.attribute_conflict_weight,
            face_similarity=item.face_similarity,
            face_match=item.face_match,
            face_query_quality=item.face_query_quality,
            face_candidate_quality=item.face_candidate_quality,
            face_reliability=item.face_reliability,
            fusion_score=item.fusion_score,
            evidence_level=item.evidence_level,
            decision_reason=item.decision_reason,
            captured_at=item.captured_at,
            # Still judged on the embedding's own score: height reorders candidates, it does not
            # get to promote one past the ceiling that says whether to believe it at all.
            beats_chance=item.score > settings.reid_chance_ceiling,
        )
        for _ranking, item in best.values()
    ]
    links.sort(key=lambda link: best[link.camera_id][0], reverse=True)
    source = _to_items(db, settings, [ReidMatch(crop_id=crop_id, score=1.0)])
    return ReidLinkResponse(
        crop_id=crop_id,
        camera_id=crop.camera_id,
        camera_name=source[0].camera_name if source else None,
        captured_at=crop.captured_at,
        links=links,
        chance_ceiling=settings.reid_chance_ceiling,
        query_mode="tracklet" if len(query_crops) > 1 else "single_frame",
        query_frame_count=len(query_crops),
    )


@router.post("/index/rebuild", response_model=ReidRebuildResponse)
def reid_rebuild(
    db: DBSession,
    settings: AppSettings,
    limit: int = Query(default=200, ge=1, le=5000),
) -> ReidRebuildResponse:
    service = _service(db, settings)
    try:
        result = service.backfill(limit)
    except VectorIndexError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return ReidRebuildResponse(**result)
