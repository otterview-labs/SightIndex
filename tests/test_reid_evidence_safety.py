"""Adversarial fusion cases; only synthetic vectors and mocked storage are used."""

import uuid
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from sqlalchemy.orm import Session

from app.api.reid import _filter_by_attributes
from app.config.settings import Settings
from app.models.media import PersonCrop
from app.schemas.reid import ReidMatchItem
from app.services.faces import FaceCandidate, FaceRecognitionService, FaceRuntimeStatus
from app.services.reid_fusion import enrich_face_evidence, fusion_rank


@pytest.mark.parametrize("color", ["black", "white"])
def test_one_comparable_tag_is_neutral_in_bonus_and_tiebreak(color: str) -> None:
    """A lone matching or conflicting tag is not sufficient ranking evidence."""
    db = MagicMock(spec=Session)
    tagged = ReidMatchItem(crop_id=uuid.uuid4(), score=0.6)
    unknown = ReidMatchItem(crop_id=uuid.uuid4(), score=0.6)
    query = {"clothing": {"upper_color": "black", "upper_color_confidence": 0.95}}
    db.execute.return_value = [
        (tagged.crop_id, {"clothing": {"upper_color": color, "upper_color_confidence": 0.95}}),
        (unknown.crop_id, None),
    ]
    _, bonuses = _filter_by_attributes(db, Settings(), [tagged, unknown], query)
    assert tagged.attribute_comparable_count == 1
    assert bonuses.get(tagged.crop_id, 0.0) == 0.0
    assert fusion_rank(tagged, tagged.score) == fusion_rank(unknown, unknown.score)


@pytest.mark.parametrize(
    (
        "source_quality",
        "neighbour_vector",
        "candidate_vector",
        "expected_match",
        "expected_quality",
    ),
    [
        (0.8, [0.0, 1.0], [1.0, 0.0], True, 0.8),
        (0.8, [1.0, 0.0], [1.0, 0.0], True, 0.95),
        (None, [0.0, 1.0], [1.0, 0.0], None, 0.95),
        (None, [0.0, 1.0], [0.0, 1.0], None, 0.95),
        (0.6, [1.0, 0.0], [1.0, 0.0], None, 0.95),
    ],
)
def test_gallery_face_cannot_hijack_the_selected_person(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    source_quality: float | None,
    neighbour_vector: list[float],
    candidate_vector: list[float],
    expected_match: bool | None,
    expected_quality: float,
) -> None:
    """Clearer neighbouring faces need a reliable source identity before hard decisions."""
    source = PersonCrop(id=uuid.uuid4(), image_id=uuid.uuid4(), crop_url="/source.jpg")
    neighbour = PersonCrop(id=uuid.uuid4(), image_id=uuid.uuid4(), crop_url="/neighbour.jpg")
    candidate = PersonCrop(id=uuid.uuid4(), image_id=uuid.uuid4(), crop_url="/candidate.jpg")
    faces = {
        source.id: (
            FaceCandidate([1.0, 0.0], {}, source_quality, "test")
            if source_quality is not None else None
        ),
        neighbour.id: FaceCandidate(neighbour_vector, {}, 0.95, "test"),
        candidate.id: FaceCandidate(candidate_vector, {}, 0.9, "test"),
    }
    monkeypatch.setattr(
        FaceRecognitionService, "_cached_strict_candidate", lambda self, crop: faces[crop.id]
    )
    # Patching the function's owning module also works after API tests reload app modules.
    monkeypatch.setitem(
        enrich_face_evidence.__globals__,
        "face_runtime_status",
        lambda settings: FaceRuntimeStatus(True, "test", "test", "cpu", ""),
    )
    db = MagicMock(spec=Session)
    db.scalars.return_value = [candidate]
    item = ReidMatchItem(crop_id=candidate.id, score=0.6)
    enrich_face_evidence(
        db,
        Settings(data_dir=tmp_path),
        source,
        [item],
        # Explicitly selected crop, not list order, anchors identity.
        query_crops=[neighbour, source],
    )
    assert item.face_match is expected_match
    assert item.face_query_quality == expected_quality
