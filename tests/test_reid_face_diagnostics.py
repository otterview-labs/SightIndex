"""Synthetic-only regressions for strict ReID face evidence and recoverable caching."""

import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session

from app.config.settings import Settings
from app.db.session import Base, _ensure_compatible_schema
from app.models.media import Image, PersonCrop
from app.models.vectors import CropFaceExtraction
from app.schemas.reid import ReidMatchItem
from app.services.faces import FaceCandidate, FaceCropComparison, FaceRecognitionService
from app.services.reid_fusion import enrich_face_evidence


@pytest.fixture
def face_db() -> Iterator[Session]:
    """Keep cache writes and schema creation isolated from project databases."""
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine, autoflush=False) as db:
        yield db
    engine.dispose()


def _face(
    quality: float = 0.9,
    *,
    model: str = "insightface-buffalo_l",
    embedding: list[float] | None = None,
) -> FaceCandidate:
    return FaceCandidate(
        embedding=[1.0, 0.0] if embedding is None else embedding,
        bbox={"x": 10.0, "y": 5.0, "width": 40.0, "height": 40.0},
        quality_score=quality,
        model=model,
    )


def _crop(name: str) -> PersonCrop:
    return PersonCrop(
        id=uuid.uuid4(),
        image_id=uuid.uuid4(),
        crop_url=f"/data/crops/{name}.png",
        bbox={"label": "person"},
    )


@pytest.fixture
def original_crop(face_db: Session, tmp_path: Path) -> tuple[PersonCrop, Path]:
    """Generate noise, not a real person's image, for original-ROI and stat tests."""
    frame_path = tmp_path / "original.png"
    pixels = np.random.default_rng(42).integers(0, 255, (300, 200, 3), dtype=np.uint8)
    assert cv2.imwrite(str(frame_path), pixels)
    image = Image(id=uuid.uuid4(), image_url="/data/original.png", source_type="upload")
    crop = _crop("synthetic")
    crop.image_id = image.id
    crop.bbox = {"x": 30, "y": 10, "width": 100, "height": 240}
    face_db.add_all([image, crop])
    face_db.commit()
    return crop, frame_path


def _service(face_db: Session, tmp_path: Path) -> FaceRecognitionService:
    return FaceRecognitionService(face_db, Settings(data_dir=tmp_path))


def test_face_cache_schema_upgrade_preserves_legacy_rows_and_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An additive migration must not erase either positive or unexplained negative rows."""
    engine = create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE crop_face_extractions (crop_id VARCHAR PRIMARY KEY, "
                "signature VARCHAR NOT NULL, embedding JSON, face_bbox JSON, "
                "quality_score NUMERIC, face_model VARCHAR, created_at TIMESTAMP, "
                "updated_at TIMESTAMP)"
            )
        )
        connection.execute(
            text(
                "INSERT INTO crop_face_extractions (crop_id, signature, embedding, quality_score) "
                "VALUES ('positive', 'legacy', '[1.0, 0.0]', 0.9), "
                "('negative', 'legacy', NULL, NULL)"
            )
        )
    monkeypatch.setitem(_ensure_compatible_schema.__globals__, "engine", engine)
    try:
        _ensure_compatible_schema()
        _ensure_compatible_schema()
        columns = {
            column["name"] for column in inspect(engine).get_columns("crop_face_extractions")
        }
        assert {"absence_reason", "input_fingerprint"} <= columns
        with engine.connect() as connection:
            rows = connection.execute(
                text(
                    "SELECT crop_id, embedding, absence_reason, input_fingerprint "
                    "FROM crop_face_extractions ORDER BY crop_id"
                )
            ).all()
        assert rows == [
            ("negative", None, None, None),
            ("positive", "[1.0, 0.0]", None, None),
        ]
    finally:
        engine.dispose()


@pytest.mark.parametrize(
    "reason",
    [
        "source_missing",
        "source_unreadable",
        "temp_write_failed",
        "unsupported_url",
        "inference_error",
        "extraction_unexplained",
    ],
)
def test_transient_absence_does_not_poison_cache_and_next_request_recovers(
    monkeypatch: pytest.MonkeyPatch,
    face_db: Session,
    tmp_path: Path,
    original_crop: tuple[PersonCrop, Path],
    reason: str,
) -> None:
    crop, _ = original_crop
    service = _service(face_db, tmp_path)
    attempts = 0

    def extract(_crop: PersonCrop) -> FaceCandidate | None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            service._extraction_absence_reason = reason
            return None
        assert service._extraction_absence_reason is None
        return _face()

    monkeypatch.setattr(service, "_identity_candidate_for_crop", extract)
    assert service._cached_strict_candidate(crop) is None
    face_db.commit()
    assert face_db.get(CropFaceExtraction, crop.id) is None
    assert service._cached_strict_candidate(crop) == _face()
    face_db.commit()
    assert attempts == 2
    assert face_db.get(CropFaceExtraction, crop.id).embedding == [1.0, 0.0]


def test_legacy_unexplained_negative_retries_but_legacy_positive_is_reusable(
    monkeypatch: pytest.MonkeyPatch,
    face_db: Session,
    tmp_path: Path,
    original_crop: tuple[PersonCrop, Path],
) -> None:
    crop, _ = original_crop
    service = _service(face_db, tmp_path)
    positive = _crop("legacy-positive-without-file")
    face_db.add_all(
        [
            CropFaceExtraction(crop_id=crop.id, signature=service._face_extraction_signature()),
            CropFaceExtraction(
                crop_id=positive.id,
                signature=service._face_extraction_signature(),
                embedding=[1.0, 0.0],
                face_model="insightface-buffalo_l-upscaled-3.00x",
                quality_score=0.9,
            ),
        ]
    )
    face_db.commit()
    extractor = MagicMock(return_value=_face())
    monkeypatch.setattr(service, "_identity_candidate_for_crop", extractor)
    assert service._cached_strict_candidate(crop) == _face()
    cached_positive = service._cached_strict_candidate(positive)
    assert cached_positive is not None and cached_positive.embedding == [1.0, 0.0]
    extractor.assert_called_once_with(crop)


def test_negative_cache_reuses_unchanged_input_and_expires_in_utc(
    monkeypatch: pytest.MonkeyPatch,
    face_db: Session,
    tmp_path: Path,
    original_crop: tuple[PersonCrop, Path],
) -> None:
    crop, _ = original_crop
    service = _service(face_db, tmp_path)
    calls: list[uuid.UUID] = []

    def no_face(current: PersonCrop) -> None:
        calls.append(current.id)
        service._extraction_absence_reason = "no_face"

    monkeypatch.setattr(service, "_identity_candidate_for_crop", no_face)
    assert service._cached_strict_candidate(crop) is None
    face_db.commit()
    cached = face_db.get(CropFaceExtraction, crop.id)
    assert cached is not None
    assert cached.absence_reason == "no_face" and cached.input_fingerprint
    assert service._cached_strict_candidate(crop) is None
    assert calls == [crop.id]
    assert service._extraction_absence_reason == "no_face"
    cached.updated_at = datetime.now(UTC) - timedelta(seconds=301)
    face_db.commit()
    assert service._cached_strict_candidate(crop) is None
    face_db.commit()
    assert calls == [crop.id, crop.id]


def test_source_file_change_invalidates_fresh_negative_cache(
    monkeypatch: pytest.MonkeyPatch,
    face_db: Session,
    tmp_path: Path,
    original_crop: tuple[PersonCrop, Path],
) -> None:
    crop, frame_path = original_crop
    service = _service(face_db, tmp_path)
    calls = 0

    def extract(_crop: PersonCrop) -> FaceCandidate | None:
        nonlocal calls
        calls += 1
        service._extraction_absence_reason = "no_face" if calls == 1 else None
        return None if calls == 1 else _face()

    monkeypatch.setattr(service, "_identity_candidate_for_crop", extract)
    assert service._cached_strict_candidate(crop) is None
    face_db.commit()
    old_fingerprint = face_db.get(CropFaceExtraction, crop.id).input_fingerprint
    # A synthetic rewrite changes size as well as mtime, avoiding timestamp-resolution races.
    assert cv2.imwrite(str(frame_path), np.zeros((320, 220, 3), dtype=np.uint8))
    assert service._cached_strict_candidate(crop) == _face()
    face_db.commit()
    assert calls == 2
    assert face_db.get(CropFaceExtraction, crop.id).input_fingerprint != old_fingerprint


@pytest.mark.parametrize("count,reason", [(0, "no_face"), (2, "multiple_faces")])
def test_original_roi_abstention_records_detection_stage_reason(
    monkeypatch: pytest.MonkeyPatch,
    face_db: Session,
    tmp_path: Path,
    original_crop: tuple[PersonCrop, Path],
    count: int,
    reason: str,
) -> None:
    crop, _ = original_crop
    service = _service(face_db, tmp_path)
    calls: list[Path] = []

    def extract(path: Path, allow_fallback: bool) -> list[FaceCandidate]:
        assert allow_fallback is False
        assert cv2.imread(str(path)).shape[:2] == (240, 100)
        calls.append(path)
        return [_face() for _ in range(count)]

    monkeypatch.setattr(service, "_extract_candidates", extract)
    monkeypatch.setattr(
        service,
        "_identity_quality",
        MagicMock(
            side_effect=AssertionError(
                "No or multiple detections must abstain before quality evaluation"
            )
        ),
    )
    assert service._identity_candidate_for_crop(crop) is None
    assert service._extraction_absence_reason == reason
    assert len(calls) == 1 and not calls[0].exists()


@pytest.mark.parametrize(
    "invalid_box",
    [
        {"x": float("nan"), "y": 0, "width": 20, "height": 20},
        {"x": 0, "y": 0, "width": float("inf"), "height": 20},
        {"x": 0, "y": 0, "width": -10, "height": 20},
        {"x": 0, "y": 0, "width": 20},
        {"x": "0", "y": 0, "width": 20, "height": 20},
    ],
)
def test_invalid_body_geometry_abstains_without_using_padded_display_crop(
    monkeypatch: pytest.MonkeyPatch,
    face_db: Session,
    tmp_path: Path,
    original_crop: tuple[PersonCrop, Path],
    invalid_box: dict[str, object],
) -> None:
    crop, _ = original_crop
    crop.bbox = invalid_box
    service = _service(face_db, tmp_path)
    fallback = MagicMock(side_effect=AssertionError("Invalid ROI cannot use a padded crop"))
    monkeypatch.setattr(service, "_best_candidate", fallback)
    monkeypatch.setattr(service, "_best_candidate_path", fallback)
    assert service._identity_candidate_for_crop(crop) is None
    assert service._extraction_absence_reason == "invalid_bbox"
    fallback.assert_not_called()


def test_missing_original_is_retryable_and_does_not_use_padded_crop(
    monkeypatch: pytest.MonkeyPatch,
    face_db: Session,
    tmp_path: Path,
    original_crop: tuple[PersonCrop, Path],
) -> None:
    crop, _ = original_crop
    image = face_db.get(Image, crop.image_id)
    image.image_url = "/data/not-yet-available.png"
    service = _service(face_db, tmp_path)
    fallback = MagicMock(side_effect=AssertionError("Missing original cannot use display crop"))
    monkeypatch.setattr(service, "_best_candidate", fallback)
    assert service._cached_strict_candidate(crop) is None
    assert service._extraction_absence_reason == "source_missing"
    face_db.commit()
    assert face_db.get(CropFaceExtraction, crop.id) is None


def test_original_read_failure_is_not_cached_as_no_face(
    monkeypatch: pytest.MonkeyPatch,
    face_db: Session,
    tmp_path: Path,
    original_crop: tuple[PersonCrop, Path],
) -> None:
    crop, _ = original_crop
    service = _service(face_db, tmp_path)
    monkeypatch.setattr(cv2, "imread", lambda _path: None)
    assert service._cached_strict_candidate(crop) is None
    assert service._extraction_absence_reason == "source_unreadable"
    face_db.commit()
    assert face_db.get(CropFaceExtraction, crop.id) is None


def test_face_below_head_region_is_a_separate_abstention(
    monkeypatch: pytest.MonkeyPatch,
    face_db: Session,
    tmp_path: Path,
    original_crop: tuple[PersonCrop, Path],
) -> None:
    crop, _ = original_crop
    service = _service(face_db, tmp_path)
    lower_face = FaceCandidate(
        [1.0, 0.0], {"x": 5, "y": 180, "width": 40, "height": 40}, 0.9, "test"
    )
    monkeypatch.setattr(service, "_best_candidate_path", lambda *args, **kwargs: lower_face)
    assert service._identity_candidate_for_crop(crop) is None
    assert service._extraction_absence_reason == "face_outside_head"


@pytest.mark.parametrize(
    "neighbour_model,expected_quality",
    [
        ("insightface-buffalo_l-upscaled-3.00x", 0.95),
        ("insightface-buffalo_l-upscaled-1.25x", 0.95),
        ("insightface-other_model", 0.8),
        ("insightface-buffalo_l-custom-upscaled-3.00x", 0.8),
    ],
)
def test_query_gallery_normalizes_only_known_resize_suffixes(
    monkeypatch: pytest.MonkeyPatch,
    face_db: Session,
    tmp_path: Path,
    neighbour_model: str,
    expected_quality: float,
) -> None:
    source, neighbour, candidate = _crop("source"), _crop("neighbour"), _crop("candidate")
    service = _service(face_db, tmp_path)
    faces = {
        source.id: _face(0.8),
        neighbour.id: _face(0.95, model=neighbour_model),
        candidate.id: _face(),
    }
    monkeypatch.setattr(service, "_cached_strict_candidate", lambda crop: faces[crop.id])
    compared = service.compare_person_crop_gallery(
        [source, neighbour],
        [candidate],
        min_quality=0.55,
        anchor=source,
    )
    assert compared[candidate.id].query_quality == expected_quality
    assert compared[candidate.id].query_identity_verified is True


def test_resized_detection_keeps_embedding_model_and_maps_pixels_back(
    face_db: Session,
    tmp_path: Path,
) -> None:
    service = _service(face_db, tmp_path)
    algorithm_face = SimpleNamespace(
        embedding=[1.0, 0.0],
        model="insightface-buffalo_l",
        quality_score=0.9,
        bbox={"x": 30, "y": 15, "width": 120, "height": 120},
        landmarks=[[30, 30], [90, 30], [60, 60]],
    )
    candidates = service._service_candidates(
        [algorithm_face],
        bbox_scale=3.0,
        model_suffix="-upscaled-3.00x",
    )
    assert candidates[0].model == "insightface-buffalo_l"
    assert candidates[0].bbox == {"x": 10, "y": 5, "width": 40, "height": 40}
    assert candidates[0].landmarks == [[10, 10], [30, 10], [20, 20]]


def test_missing_query_short_circuits_candidate_work_and_reports_coverage(
    monkeypatch: pytest.MonkeyPatch,
    face_db: Session,
    tmp_path: Path,
) -> None:
    source, neighbour = _crop("source"), _crop("neighbour")
    candidates = [_crop(f"candidate-{index}") for index in range(100)]
    service = _service(face_db, tmp_path)
    attempted: list[uuid.UUID] = []

    def no_face(crop: PersonCrop) -> None:
        attempted.append(crop.id)
        service._extraction_absence_reason = "no_face"

    monkeypatch.setattr(service, "_cached_strict_candidate", no_face)
    assert (
        service.compare_person_crop_gallery(
            [source, neighbour],
            candidates,
            min_quality=0.55,
            anchor=source,
        )
        == {}
    )
    assert attempted == [source.id, neighbour.id]
    assert service.coverage.status == "query_unavailable"
    assert service.coverage.query_attempted_count == 2
    assert service.coverage.query_face_found is False
    assert service.coverage.candidate_attempted_count == 0
    assert service.coverage.compared_count == 0
    assert service.coverage.query_absence_reasons == {"no_face": 2}
    assert service.coverage.candidate_absence_reasons == {}


@pytest.mark.parametrize("representative_quality", [None, 0.3])
def test_candidate_borrowing_is_bounded_best_quality_and_never_identity_verified(
    monkeypatch: pytest.MonkeyPatch,
    face_db: Session,
    tmp_path: Path,
    representative_quality: float | None,
) -> None:
    source, candidate = _crop("source"), _crop("candidate")
    peers = [_crop(f"peer-{index}") for index in range(3)]
    service = _service(face_db, tmp_path)
    service.candidate_occurrences = {candidate.id: peers}
    faces = {
        source.id: _face(),
        candidate.id: None if representative_quality is None else _face(representative_quality),
        peers[0].id: _face(0.8),
        peers[1].id: _face(0.95, embedding=[0.0, 1.0]),
        peers[2].id: _face(0.99),
    }
    attempted: list[uuid.UUID] = []

    def extract(crop: PersonCrop) -> FaceCandidate | None:
        attempted.append(crop.id)
        service._extraction_absence_reason = "no_face" if faces[crop.id] is None else None
        return faces[crop.id]

    monkeypatch.setattr(service, "_cached_strict_candidate", extract)
    compared = service.compare_person_crops(source, [candidate], min_quality=0.55)
    assert set(attempted) == {source.id, candidate.id, peers[0].id, peers[1].id}
    assert peers[2].id not in attempted
    assert compared[candidate.id].candidate_quality == 0.95
    assert compared[candidate.id].similarity == 0.0
    assert compared[candidate.id].candidate_identity_verified is False
    assert compared[candidate.id].candidate_source_crop_id == peers[1].id
    assert service.coverage.borrowed_candidate_count == 1


def test_usable_representative_is_not_replaced_by_clearer_peer(
    monkeypatch: pytest.MonkeyPatch,
    face_db: Session,
    tmp_path: Path,
) -> None:
    source, candidate, peer = _crop("source"), _crop("candidate"), _crop("peer")
    service = _service(face_db, tmp_path)
    service.candidate_occurrences = {candidate.id: [peer]}
    faces = {source.id: _face(), candidate.id: _face(0.8), peer.id: _face(0.99)}
    extractor = MagicMock(side_effect=lambda crop: faces[crop.id])
    monkeypatch.setattr(service, "_cached_strict_candidate", extractor)
    compared = service.compare_person_crops(source, [candidate], min_quality=0.55)
    assert extractor.call_count == 2
    assert compared[candidate.id].candidate_quality == 0.8
    assert compared[candidate.id].candidate_identity_verified is True
    assert service.coverage.borrowed_candidate_count == 0


@pytest.mark.parametrize("similarity", [0.99, 0.0])
def test_borrowed_candidate_face_cannot_make_hard_match_or_hard_conflict(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    similarity: float,
) -> None:
    source, candidate, peer = _crop("source"), _crop("candidate"), _crop("peer")
    db = MagicMock(spec=Session)
    db.scalars.return_value = [candidate]
    monkeypatch.setitem(
        enrich_face_evidence.__globals__,
        "face_runtime_status",
        lambda _settings: SimpleNamespace(ready=True),
    )
    monkeypatch.setattr(
        FaceRecognitionService,
        "compare_person_crops",
        lambda *args, **kwargs: {
            candidate.id: FaceCropComparison(
                similarity=similarity,
                query_quality=0.95,
                candidate_quality=0.95,
                query_identity_verified=True,
                candidate_identity_verified=False,
                candidate_source_crop_id=peer.id,
            )
        },
    )
    item = ReidMatchItem(crop_id=candidate.id, score=0.6)
    enrich_face_evidence(db, Settings(data_dir=tmp_path), source, [item])
    assert item.face_similarity == similarity
    assert item.face_match is None
    assert item.face_candidate_identity_verified is False
    assert item.face_candidate_source_crop_id == peer.id


def test_low_quality_query_is_explained_without_attempting_candidates(
    monkeypatch: pytest.MonkeyPatch,
    face_db: Session,
    tmp_path: Path,
) -> None:
    source, candidate = _crop("source"), _crop("candidate")
    service = _service(face_db, tmp_path)
    extractor = MagicMock(return_value=_face(0.3))
    monkeypatch.setattr(service, "_cached_strict_candidate", extractor)
    assert service.compare_person_crops(source, [candidate], min_quality=0.55) == {}
    extractor.assert_called_once_with(source)
    assert service.coverage.query_face_quality == 0.3
    assert service.coverage.query_identity_verified is False
    assert service.coverage.query_absence_reasons == {"low_quality": 1}
    assert service.coverage.candidate_attempted_count == 0


@pytest.mark.parametrize(
    "candidate_face",
    [
        _face(model="insightface-other_model"),
        _face(embedding=[1.0, 0.0, 0.0]),
    ],
)
def test_different_face_spaces_cannot_generate_a_comparison(
    monkeypatch: pytest.MonkeyPatch,
    face_db: Session,
    tmp_path: Path,
    candidate_face: FaceCandidate,
) -> None:
    source, candidate = _crop("source"), _crop("candidate")
    service = _service(face_db, tmp_path)
    faces = {source.id: _face(), candidate.id: candidate_face}
    monkeypatch.setattr(service, "_cached_strict_candidate", lambda crop: faces[crop.id])
    assert service.compare_person_crops(source, [candidate], min_quality=0.55) == {}
    assert service.coverage.compared_count == 0
    assert service.coverage.candidate_absence_reasons == {"model_incompatible": 1}


def test_candidate_borrowing_skips_self_and_duplicates_but_preserves_two_peer_budget(
    monkeypatch: pytest.MonkeyPatch,
    face_db: Session,
    tmp_path: Path,
) -> None:
    source, candidate, first, second = [
        _crop(name)
        for name in (
            "source",
            "candidate",
            "first",
            "second",
        )
    ]
    service = _service(face_db, tmp_path)
    service.candidate_occurrences = {candidate.id: [candidate, first, first, second]}
    faces = {
        source.id: _face(),
        candidate.id: None,
        first.id: _face(0.99, model="insightface-other_model"),
        second.id: _face(0.9),
    }
    extractor = MagicMock(side_effect=lambda crop: faces[crop.id])
    monkeypatch.setattr(service, "_cached_strict_candidate", extractor)
    compared = service.compare_person_crops(source, [candidate], min_quality=0.55)
    assert extractor.call_count == 4
    assert compared[candidate.id].candidate_source_crop_id == second.id
    assert compared[candidate.id].candidate_identity_verified is False
    assert service.coverage.candidate_absence_reasons["model_incompatible"] == 1


def test_inference_failure_is_reported_as_retryable_not_as_no_face(
    monkeypatch: pytest.MonkeyPatch,
    face_db: Session,
    tmp_path: Path,
    original_crop: tuple[PersonCrop, Path],
) -> None:
    source, _ = original_crop
    candidate = _crop("candidate")
    service = _service(face_db, tmp_path)
    extractor = MagicMock(side_effect=ValueError("synthetic inference failure"))
    monkeypatch.setattr(service, "_identity_candidate_for_crop", extractor)
    assert service.compare_person_crops(source, [candidate], min_quality=0.55) == {}
    assert service.coverage.status == "error"
    assert service.coverage.query_absence_reasons == {"inference_error": 1}
    assert service.coverage.candidate_attempted_count == 0
    assert face_db.get(CropFaceExtraction, source.id) is None


def test_legacy_existing_unreadable_file_is_not_cached_as_no_face(
    monkeypatch: pytest.MonkeyPatch,
    face_db: Session,
    tmp_path: Path,
    original_crop: tuple[PersonCrop, Path],
) -> None:
    crop, _ = original_crop
    crop.bbox = {"label": "person"}
    crop.crop_url = "/data/original.png"
    service = _service(face_db, tmp_path)
    monkeypatch.setattr(cv2, "imread", lambda _path: None)
    monkeypatch.setattr(service, "_extract_candidates", lambda *args, **kwargs: [])
    assert service._cached_strict_candidate(crop) is None
    assert service._extraction_absence_reason == "source_unreadable"
    face_db.commit()
    assert face_db.get(CropFaceExtraction, crop.id) is None


def test_face_coverage_is_reset_between_requests_on_reused_service(
    monkeypatch: pytest.MonkeyPatch,
    face_db: Session,
    tmp_path: Path,
) -> None:
    source, candidate = _crop("source"), _crop("candidate")
    service = _service(face_db, tmp_path)
    monkeypatch.setattr(service, "_cached_strict_candidate", lambda _crop: _face())
    assert service.compare_person_crops(source, [candidate], min_quality=0.55)
    assert service.coverage.compared_count == 1

    def no_face(_crop: PersonCrop) -> None:
        service._extraction_absence_reason = "no_face"

    monkeypatch.setattr(service, "_cached_strict_candidate", no_face)
    assert service.compare_person_crops(source, [candidate], min_quality=0.55) == {}
    assert service.coverage.query_absence_reasons == {"no_face": 1}
    assert service.coverage.query_attempted_count == 1
    assert service.coverage.query_face_quality is None
    assert service.coverage.query_identity_verified is False
    assert service.coverage.candidate_attempted_count == 0
    assert service.coverage.compared_count == 0


@pytest.mark.parametrize("existing_shared_row", [False, True])
def test_overlapping_face_cache_commits_preserve_both_request_batches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    existing_shared_row: bool,
) -> None:
    """Cold concurrent snapshots must not roll back unrelated rows on a shared crop race."""
    engine = create_engine(f"sqlite:///{tmp_path / 'concurrent-faces.db'}")
    Base.metadata.create_all(engine)
    shared, first_only, second_only = [
        _crop(name)
        for name in (
            "shared",
            "first-only",
            "second-only",
        )
    ]
    with Session(engine) as setup:
        for crop in (shared, first_only, second_only):
            setup.add(Image(id=crop.image_id, image_url=crop.crop_url, source_type="upload"))
            setup.add(crop)
        if existing_shared_row:
            setup.add(
                CropFaceExtraction(
                    crop_id=shared.id,
                    signature="obsolete",
                    embedding=[0.0, 1.0],
                    quality_score=0.7,
                    face_model="old-model",
                )
            )
        setup.commit()
        shared_id, first_id, second_id = shared.id, first_only.id, second_only.id

    try:
        with (
            Session(engine, autoflush=False) as first_db,
            Session(
                engine,
                autoflush=False,
            ) as second_db,
        ):
            first_service = _service(first_db, tmp_path)
            second_service = _service(second_db, tmp_path)
            monkeypatch.setattr(
                first_service,
                "_identity_candidate_for_crop",
                lambda _crop: _face(0.8),
            )
            monkeypatch.setattr(
                second_service,
                "_identity_candidate_for_crop",
                lambda _crop: _face(0.95),
            )
            # Both requests finish extraction/staging before either commits the shared key.
            assert first_service._cached_strict_candidate(first_db.get(PersonCrop, shared_id))
            assert first_service._cached_strict_candidate(first_db.get(PersonCrop, first_id))
            assert second_service._cached_strict_candidate(second_db.get(PersonCrop, shared_id))
            assert second_service._cached_strict_candidate(second_db.get(PersonCrop, second_id))
            first_service._commit_crop_face_cache()
            second_service._commit_crop_face_cache()

        with Session(engine) as check:
            rows = {row.crop_id: row for row in check.query(CropFaceExtraction).all()}
            assert set(rows) == {shared_id, first_id, second_id}
            assert float(rows[shared_id].quality_score) == pytest.approx(0.95)
            assert float(rows[first_id].quality_score) == pytest.approx(0.8)
            assert float(rows[second_id].quality_score) == pytest.approx(0.95)
            assert all(row.embedding == [1.0, 0.0] for row in rows.values())
            assert all(
                row.absence_reason is None and row.input_fingerprint for row in rows.values()
            )
        assert not any(
            "Failed to persist crop face extraction cache" in record.getMessage()
            for record in caplog.records
        )
    finally:
        engine.dispose()


@pytest.mark.parametrize("operation", ["imread", "imwrite"])
def test_original_roi_opencv_errors_degrade_without_negative_cache(
    monkeypatch: pytest.MonkeyPatch,
    face_db: Session,
    tmp_path: Path,
    original_crop: tuple[PersonCrop, Path],
    operation: str,
) -> None:
    """cv2.error is not a ValueError, including failures before the recognizer wrapper."""
    source, _ = original_crop
    candidate = _crop("candidate")
    service = _service(face_db, tmp_path)
    monkeypatch.setattr(cv2, operation, MagicMock(side_effect=cv2.error("synthetic codec error")))
    assert service.compare_person_crops(source, [candidate], min_quality=0.55) == {}
    assert service.coverage.status == "error"
    assert service.coverage.query_absence_reasons == {"inference_error": 1}
    assert service.coverage.candidate_attempted_count == 0
    assert face_db.get(CropFaceExtraction, source.id) is None


def test_upload_original_read_opencv_error_degrades_without_attempting_candidates(
    monkeypatch: pytest.MonkeyPatch,
    face_db: Session,
    tmp_path: Path,
    original_crop: tuple[PersonCrop, Path],
) -> None:
    _, path = original_crop
    service = _service(face_db, tmp_path)
    monkeypatch.setattr(cv2, "imread", MagicMock(side_effect=cv2.error("synthetic codec error")))
    assert service.compare_image_to_crops(path, [_crop("candidate")], min_quality=0.55) == {}
    assert service.coverage.status == "error"
    assert service.coverage.query_absence_reasons == {"inference_error": 1}
    assert service.coverage.candidate_attempted_count == 0
    assert face_db.query(CropFaceExtraction).count() == 0


@pytest.mark.parametrize("uploaded", [False, True])
def test_missing_extraction_dependency_degrades_for_crop_and_upload(
    monkeypatch: pytest.MonkeyPatch,
    face_db: Session,
    tmp_path: Path,
    original_crop: tuple[PersonCrop, Path],
    uploaded: bool,
) -> None:
    source, path = original_crop
    service = _service(face_db, tmp_path)
    extractor = "_best_candidate_path" if uploaded else "_identity_candidate_for_crop"
    monkeypatch.setattr(
        service, extractor, MagicMock(side_effect=ImportError("synthetic missing lib"))
    )
    if uploaded:
        result = service.compare_image_to_crops(path, [_crop("candidate")], min_quality=0.55)
    else:
        result = service.compare_person_crops(source, [_crop("candidate")], min_quality=0.55)
    assert result == {}
    assert service.coverage.status == "error"
    assert service.coverage.query_absence_reasons == {"inference_error": 1}
    assert service.coverage.candidate_attempted_count == 0
    assert face_db.query(CropFaceExtraction).count() == 0


@pytest.mark.parametrize("error_type", [cv2.error, ImportError])
@pytest.mark.parametrize("uploaded", [False, True])
def test_fusion_boundary_contains_native_and_dependency_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    error_type: type[Exception],
    uploaded: bool,
) -> None:
    source, candidate = _crop("source"), _crop("candidate")
    db = MagicMock(spec=Session)
    db.scalars.return_value = [candidate]
    monkeypatch.setitem(
        enrich_face_evidence.__globals__,
        "face_runtime_status",
        lambda _settings: SimpleNamespace(ready=True),
    )
    comparison = "compare_image_to_crops" if uploaded else "compare_person_crops"
    monkeypatch.setattr(
        FaceRecognitionService,
        comparison,
        MagicMock(side_effect=error_type("synthetic extraction failure")),
    )
    item = ReidMatchItem(crop_id=candidate.id, score=0.6)
    coverage = enrich_face_evidence(
        db,
        Settings(data_dir=tmp_path),
        None if uploaded else source,
        [item],
        query_image_path=tmp_path / "upload.png" if uploaded else None,
    )
    assert coverage.status == "error"
    assert item.face_similarity is None and item.face_match is None
    db.add.assert_not_called()


@pytest.mark.parametrize(
    "malformed_embedding",
    [
        ["invalid", 0.0],
        [True, False],
        [float("nan"), 0.0],
        [0.0, 0.0],
        [],
    ],
)
@pytest.mark.parametrize("uploaded", [False, True])
def test_malformed_face_vectors_are_neutral_and_never_cached(
    monkeypatch: pytest.MonkeyPatch,
    face_db: Session,
    tmp_path: Path,
    original_crop: tuple[PersonCrop, Path],
    malformed_embedding: list[object],
    uploaded: bool,
) -> None:
    source, path = original_crop
    service = _service(face_db, tmp_path)
    invalid_face = _face(embedding=cast(list[float], malformed_embedding))
    extractor = "_best_candidate_path" if uploaded else "_identity_candidate_for_crop"
    monkeypatch.setattr(service, extractor, MagicMock(return_value=invalid_face))
    if uploaded:
        result = service.compare_image_to_crops(path, [_crop("candidate")], min_quality=0.55)
    else:
        result = service.compare_person_crops(source, [_crop("candidate")], min_quality=0.55)
    assert result == {}
    assert service.coverage.query_absence_reasons == {"invalid_embedding": 1}
    assert service.coverage.candidate_attempted_count == 0
    assert face_db.query(CropFaceExtraction).count() == 0
