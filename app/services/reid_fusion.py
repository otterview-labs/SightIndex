import logging
import uuid
from collections import defaultdict
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config.settings import Settings
from app.models.media import PersonCrop
from app.schemas.reid import ReidFaceCoverage, ReidMatchItem
from app.services.faces import FaceRecognitionService, face_extraction_errors, face_runtime_status

logger = logging.getLogger(__name__)


def enrich_face_evidence(
    db: Session,
    settings: Settings,
    query_crop: PersonCrop | None,
    items: list[ReidMatchItem],
    *,
    query_image_path: Path | None = None,
    query_crops: list[PersonCrop] | None = None,
    coverage: ReidFaceCoverage | None = None,
) -> ReidFaceCoverage:
    """Annotate a camera-balanced shortlist with optional face evidence in place."""

    coverage = coverage if coverage is not None else ReidFaceCoverage()
    if not settings.reid_face_priority_enabled:
        coverage.status = "disabled"
        return coverage
    if query_crop is None and query_image_path is None:
        coverage.status = "query_unavailable"
        return coverage
    if not items:
        coverage.status = "no_candidates"
        return coverage

    shortlist = _camera_balanced_shortlist(
        items,
        settings.reid_face_candidate_limit,
    )
    coverage.shortlist_count = len(shortlist)
    if not shortlist:
        return coverage
    if not face_runtime_status(settings).ready:
        coverage.status = "unavailable"
        return coverage
    member_ids: dict[uuid.UUID, list[uuid.UUID]] = {}
    if settings.reid_collapse_identity_threshold > 0 and settings.reid_collapse_window_seconds > 0:
        member_ids = {
            item.crop_id: list(
                dict.fromkeys(
                    member for member in item.occurrence_crop_ids if member != item.crop_id
                )
            )[:2]
            for item in shortlist
            if item.camera_id is not None
        }
    crop_ids = {item.crop_id for item in shortlist}
    crop_ids.update(member for members in member_ids.values() for member in members)
    crops = {
        crop.id: crop for crop in db.scalars(select(PersonCrop).where(PersonCrop.id.in_(crop_ids)))
    }
    ordered_crops = [crops[item.crop_id] for item in shortlist if item.crop_id in crops]
    if not ordered_crops:
        coverage.status = "candidate_unavailable"
        coverage.candidate_absence_reasons["candidate_row_missing"] = len(shortlist)
        return coverage
    try:
        face_service = FaceRecognitionService(db, settings)
        for item in shortlist:
            representative = crops.get(item.crop_id)
            if representative is None or representative.camera_id != item.camera_id:
                continue
            face_service.candidate_occurrences[item.crop_id] = [
                crops[member]
                for member in member_ids.get(item.crop_id, [])
                if member in crops
                and crops[member].camera_id == representative.camera_id
                and crops[member].captured_at is not None
                and representative.captured_at is not None
                and abs((crops[member].captured_at - representative.captured_at).total_seconds())
                <= settings.reid_collapse_window_seconds
            ]
        if query_crops and len(query_crops) > 1:
            evidence = face_service.compare_person_crop_gallery(
                query_crops,
                ordered_crops,
                min_quality=settings.reid_face_min_quality,
                anchor=query_crop,
            )
        elif query_crop is not None:
            evidence = face_service.compare_person_crops(
                query_crop,
                ordered_crops,
                min_quality=settings.reid_face_min_quality,
            )
        else:
            assert query_image_path is not None
            evidence = face_service.compare_image_to_crops(
                query_image_path,
                ordered_crops,
                min_quality=settings.reid_face_min_quality,
            )
        for name in type(coverage).model_fields:
            setattr(coverage, name, getattr(face_service.coverage, name))
        coverage.shortlist_count = len(shortlist)
        missing = len(shortlist) - len(ordered_crops)
        if missing:
            coverage.candidate_absence_reasons["candidate_row_missing"] = (
                coverage.candidate_absence_reasons.get("candidate_row_missing", 0) + missing
            )
    except face_extraction_errors():
        logger.warning("ReID face priority degraded to body and attributes", exc_info=True)
        coverage.status = "error"
        return coverage

    for item in shortlist:
        comparison = evidence.get(item.crop_id)
        if comparison is None:
            continue
        item.face_similarity = round(comparison.similarity, 4)
        item.face_query_quality = round(comparison.query_quality, 4)
        item.face_candidate_quality = round(comparison.candidate_quality, 4)
        item.face_query_identity_verified = comparison.query_identity_verified
        item.face_candidate_identity_verified = comparison.candidate_identity_verified
        item.face_candidate_source_crop_id = comparison.candidate_source_crop_id
        reliability = min(comparison.query_quality, comparison.candidate_quality)
        item.face_reliability = round(reliability, 4)
        if (
            not comparison.query_identity_verified
            or not comparison.candidate_identity_verified
            or reliability < settings.reid_face_strong_reliability
        ):
            item.face_match = None
        elif comparison.similarity >= settings.face_match_threshold:
            item.face_match = True
        elif comparison.similarity < settings.reid_face_hard_reject_threshold:
            item.face_match = False
        else:
            item.face_match = None
    coverage.hard_match_count = sum(item.face_match is True for item in shortlist)
    coverage.hard_conflict_count = sum(item.face_match is False for item in shortlist)
    return coverage


def _face_evidence(item: ReidMatchItem) -> float:
    """Bounded, reliability-scaled nudge from soft face similarity.

    The calibrated 0.30..0.45 uncertainty band is centred at zero. Face quality controls how
    far soft evidence can move a candidate; it cannot create a hard decision by itself. Shared
    by ``fusion_rank`` and ``annotate_fusion_decision`` so a calibration tweak never desyncs the
    ranking score from the displayed fusion score.
    """

    if item.face_similarity is None:
        return 0.0
    reliability = item.face_reliability or 0.0
    return reliability * max(-1.0, min(1.0, (item.face_similarity - 0.375) / 0.075))


def fusion_rank(
    item: ReidMatchItem,
    appearance_score: float,
) -> tuple[int, float, float]:
    """Reliable face decisions first; every other signal remains continuous.

    ``appearance_score`` already contains the bounded label and stature nudges. Making label
    agreement another categorical tier used to let a mediocre body match with two common colour
    labels jump ahead of a much stronger unlabelled body match.
    """

    face_tier = 2 if item.face_match is True else 0 if item.face_match is False else 1
    # Soft face evidence is a bounded nudge, not another tier. Only a reliable face decision gets
    # categorical priority; otherwise a tiny low-quality face difference must not beat a much
    # stronger body vector.
    fused_score = appearance_score + 0.05 * _face_evidence(item)
    return (
        face_tier,
        fused_score,
        item.attribute_agreement
        if item.attribute_comparable_count >= 2 and item.attribute_agreement is not None
        else 0.5,
    )


def annotate_fusion_decision(
    item: ReidMatchItem,
    appearance_score: float,
    *,
    query_camera: uuid.UUID | None,
    chance_ceiling: float,
    is_camera_link: bool = False,
) -> None:
    """Expose one backend-owned score and explanation for display and later evaluation."""

    item.fusion_score = round(appearance_score + 0.05 * _face_evidence(item), 4)

    if item.face_match is True:
        item.evidence_level = "reliable"
        item.decision_reason = "可靠人脸吻合，优先级高于人体和衣着标签"
    elif item.face_match is False:
        item.evidence_level = "rejected"
        item.decision_reason = "可靠人脸明确冲突"
    elif query_camera is not None and item.camera_id == query_camera and item.score >= 0.85:
        item.evidence_level = "reliable"
        item.decision_reason = "同摄像头人体特征高度相似"
    elif is_camera_link and item.score <= chance_ceiling:
        item.evidence_level = "clue"
        item.decision_reason = "该摄像头最佳候选，但人体分数仍可由巧合解释"
    elif item.face_similarity is not None:
        item.evidence_level = "similar"
        item.decision_reason = "人体达到候选范围；人脸未形成可靠身份结论，仅作软证据"
    elif item.attribute_conflicts and item.attribute_comparable_count >= 2:
        item.evidence_level = "similar"
        item.decision_reason = "人体达到候选范围；衣着存在冲突，已降权但未硬排除"
    else:
        item.evidence_level = "similar"
        item.decision_reason = "人体达到候选范围；标签和身高只参与小幅加权"


def reject_reliable_face_mismatches(
    items: list[ReidMatchItem],
    threshold: float,
) -> list[ReidMatchItem]:
    """Remove only measured, decisive face contradictions.

    Decisive conflicts are omitted from ordinary candidates. Missing, low-quality, uncertain or
    unanchored tracklet comparisons remain body/attribute candidates; they are not hard conflicts.
    """

    if threshold <= 0:
        return items
    return [
        item
        for item in items
        if not (
            item.face_match is False
            and item.face_similarity is not None
            and item.face_similarity < threshold
        )
    ]


def _camera_balanced_shortlist(
    items: list[ReidMatchItem],
    limit: int,
) -> list[ReidMatchItem]:
    groups: dict[uuid.UUID | None, list[ReidMatchItem]] = defaultdict(list)
    for item in items:
        groups[item.camera_id].append(item)
    queues = sorted(
        (sorted(group, key=lambda item: item.score, reverse=True) for group in groups.values()),
        key=lambda group: group[0].score,
        reverse=True,
    )
    chosen: list[ReidMatchItem] = []
    depth = 0
    while len(chosen) < limit:
        added = False
        for queue in queues:
            if depth < len(queue):
                chosen.append(queue[depth])
                added = True
                if len(chosen) >= limit:
                    break
        if not added:
            break
        depth += 1
    return chosen
