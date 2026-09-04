import uuid

import pytest

from app.config.settings import Settings
from app.models.media import PersonCrop
from app.schemas.reid import ReidMatchItem
from app.services.faces import FaceCandidate, FaceRecognitionService
from app.services.reid_fusion import (
    enrich_face_evidence,
    fusion_rank,
    reject_reliable_face_mismatches,
)


class _EmptyDb:
    def scalars(self, _statement):
        return []


def _crop(name: str, person_id: uuid.UUID | None = None) -> PersonCrop:
    return PersonCrop(
        image_id=uuid.uuid4(),
        crop_url=f"/data/crops/{name}.jpg",
        bbox={"label": "person"},
        person_id=person_id,
    )


def test_person_labels_do_not_masquerade_as_face_evidence(tmp_path, monkeypatch):
    person = uuid.uuid4()
    query = _crop("query", person)
    same = ReidMatchItem(crop_id=uuid.uuid4(), score=0.46, person_id=person)
    different = ReidMatchItem(
        crop_id=uuid.uuid4(), score=0.92, person_id=uuid.uuid4()
    )
    monkeypatch.setattr(
        "app.services.reid_fusion.face_runtime_status",
        lambda _settings: type("Status", (), {"ready": False})(),
    )

    enrich_face_evidence(
        _EmptyDb(),
        Settings(data_dir=tmp_path),
        query,
        [same, different],
    )

    assert same.face_match is None
    assert same.face_similarity is None
    assert different.face_match is None
    assert fusion_rank(different, different.score) > fusion_rank(same, same.score)


def test_reliable_face_mismatch_is_rejected_but_uncertain_and_missing_faces_remain():
    decisive = ReidMatchItem(
        crop_id=uuid.uuid4(),
        score=0.82,
        face_match=False,
        face_similarity=0.18,
    )
    uncertain = ReidMatchItem(
        crop_id=uuid.uuid4(),
        score=0.76,
        face_match=False,
        face_similarity=0.34,
    )
    body_only = ReidMatchItem(crop_id=uuid.uuid4(), score=0.62)

    kept = reject_reliable_face_mismatches(
        [decisive, uncertain, body_only],
        threshold=0.30,
    )

    assert kept == [uncertain, body_only]


def test_face_hard_reject_can_be_disabled():
    mismatch = ReidMatchItem(
        crop_id=uuid.uuid4(),
        score=0.82,
        face_match=False,
        face_similarity=0.0,
    )

    assert reject_reliable_face_mismatches([mismatch], threshold=0) == [mismatch]


def test_attributes_are_continuous_and_do_not_create_an_absolute_tier():
    disagreement = ReidMatchItem(
        crop_id=uuid.uuid4(),
        score=0.90,
        attribute_agreement=0.5,
    )
    unknown = ReidMatchItem(crop_id=uuid.uuid4(), score=0.60)
    strong_agreement = ReidMatchItem(
        crop_id=uuid.uuid4(),
        score=0.45,
        attribute_agreement=0.75,
    )

    assert fusion_rank(disagreement, disagreement.score) > fusion_rank(unknown, unknown.score)
    assert fusion_rank(unknown, unknown.score) > fusion_rank(
        strong_agreement, strong_agreement.score
    )


def test_face_quality_scales_soft_evidence_around_unknown():
    weak_positive = ReidMatchItem(
        crop_id=uuid.uuid4(),
        score=0.60,
        face_similarity=0.70,
        face_reliability=0.60,
    )
    weak_negative = ReidMatchItem(
        crop_id=uuid.uuid4(),
        score=0.90,
        face_similarity=0.20,
        face_reliability=0.60,
    )
    unknown = ReidMatchItem(crop_id=uuid.uuid4(), score=0.50)

    assert fusion_rank(weak_positive, weak_positive.score) > fusion_rank(unknown, unknown.score)
    assert fusion_rank(weak_negative, weak_negative.score) > fusion_rank(unknown, unknown.score)


def test_soft_face_evidence_cannot_overrule_a_large_body_gap():
    weak_face = ReidMatchItem(
        crop_id=uuid.uuid4(),
        score=0.50,
        face_similarity=0.70,
        face_reliability=0.60,
    )
    body = ReidMatchItem(crop_id=uuid.uuid4(), score=0.60)

    assert fusion_rank(body, body.score) > fusion_rank(weak_face, weak_face.score)


def test_small_attribute_difference_does_not_overrule_body_within_the_same_tier():
    slightly_better_labels = ReidMatchItem(
        crop_id=uuid.uuid4(), score=0.52, attribute_agreement=0.53
    )
    much_better_body = ReidMatchItem(
        crop_id=uuid.uuid4(), score=0.66, attribute_agreement=0.50
    )

    assert fusion_rank(much_better_body, much_better_body.score) > fusion_rank(
        slightly_better_labels, slightly_better_labels.score
    )


def test_crop_face_comparison_ignores_missing_or_low_quality_faces(
    monkeypatch, tmp_path
):
    query_path = tmp_path / "query.jpg"
    strong = _crop("strong")
    low_quality = _crop("low")
    missing = _crop("missing")
    vectors = {
        strong.crop_url: FaceCandidate([0.8, 0.6], {}, 0.8, "test"),
        low_quality.crop_url: FaceCandidate([1.0, 0.0], {}, 0.4, "test"),
        missing.crop_url: None,
    }
    service = FaceRecognitionService(None, Settings(data_dir=tmp_path))
    monkeypatch.setattr(
        service,
        "_best_candidate_path",
        lambda image_path, allow_fallback: FaceCandidate(
            [1.0, 0.0], {}, 0.9, "test"
        ),
    )
    monkeypatch.setattr(
        service,
        "_best_candidate",
        lambda crop_url, allow_fallback: vectors[crop_url],
    )

    compared = service.compare_image_to_crops(
        query_path,
        [strong, low_quality, missing],
        min_quality=0.55,
    )

    assert list(compared) == [strong.id]
    assert compared[strong.id].similarity == pytest.approx(0.8)
    assert compared[strong.id].query_quality == 0.9
    assert compared[strong.id].candidate_quality == 0.8


def test_face_gallery_uses_the_best_query_face(monkeypatch, tmp_path):
    weak_query = _crop("weak-query")
    strong_query = _crop("strong-query")
    match = _crop("match")
    faces = {
        weak_query.crop_url: FaceCandidate([0.0, 1.0], {}, 0.6, "test"),
        strong_query.crop_url: FaceCandidate([1.0, 0.0], {}, 0.9, "test"),
        match.crop_url: FaceCandidate([0.8, 0.6], {}, 0.8, "test"),
    }
    service = FaceRecognitionService(None, Settings(data_dir=tmp_path))
    monkeypatch.setattr(
        service,
        "_best_candidate",
        lambda crop_url, allow_fallback: faces[crop_url],
    )

    compared = service.compare_person_crop_gallery(
        [weak_query, strong_query],
        [match],
        min_quality=0.55,
    )

    assert compared[match.id].similarity == pytest.approx(0.8)
    assert compared[match.id].query_quality == 0.9
