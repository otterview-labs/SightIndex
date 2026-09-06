import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config.settings import Settings
from app.db.session import Base
from app.models import events, media, persons  # noqa: F401 - register tables
from app.models import vectors as vectors_models  # noqa: F401 - register tables
from app.models.media import PersonCrop
from app.models.vectors import CropFaceExtraction
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
        id=uuid.uuid4(),
        image_id=uuid.uuid4(),
        crop_url=f"/data/crops/{name}.jpg",
        bbox={"label": "person"},
        person_id=person_id,
    )


def _session(tmp_path, name: str = "faces.db"):
    """Real sqlite session: compare_* now caches through CropFaceExtraction."""

    engine = create_engine(f"sqlite:///{tmp_path / name}")
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine, autoflush=False)()


def test_negative_query_gallery_cache_survives_a_fresh_session(monkeypatch, tmp_path):
    queries = [_crop("back"), _crop("blur")]
    calls = []
    settings = Settings(data_dir=tmp_path)
    for _ in range(2):
        with _session(tmp_path) as db:
            service = FaceRecognitionService(db, settings)

            def no_face(url, allow_fallback, service=service):
                calls.append(url)
                return service._abstain("no_face")

            monkeypatch.setattr(service, "_best_candidate", no_face)
            assert service.compare_person_crop_gallery(queries, [], min_quality=0.55) == {}
    assert calls == [crop.crop_url for crop in queries]
    with _session(tmp_path) as db:
        assert db.query(CropFaceExtraction).count() == 2


def test_strict_face_selection_abstains_on_multiple_people(monkeypatch, tmp_path):
    import cv2
    import numpy as np

    service = FaceRecognitionService(_session(tmp_path), Settings(data_dir=tmp_path))
    faces = [FaceCandidate([1.0], {}, 0.99, "test"), FaceCandidate([0.1], {}, 0.7, "test")]
    monkeypatch.setattr(service, "_extract_candidates", lambda *args, **kwargs: faces)
    cv2.imwrite(str(tmp_path / "crowd.png"), np.zeros((80, 80, 3), dtype=np.uint8))
    assert service._best_candidate_path(tmp_path / "crowd.png", allow_fallback=False) is None


def test_identity_quality_uses_original_pixels_sharpness_and_pose(tmp_path):
    from dataclasses import replace

    import cv2
    import numpy as np

    path = tmp_path / "raw.png"
    service = FaceRecognitionService(_session(tmp_path), Settings(data_dir=tmp_path))
    face = FaceCandidate(
        [1.0],
        {"x": 0, "y": 0, "width": 80, "height": 80},
        0.99,
        "test",
        landmarks=[[20, 20], [60, 20], [40, 40]],
    )
    pixels = np.random.default_rng(4).integers(0, 255, (80, 80, 3), dtype=np.uint8)
    cv2.imwrite(str(path), pixels)
    assert service._identity_quality(path, face).quality_score > 0.9
    tiny = replace(face, bbox={"x": 0, "y": 0, "width": 20, "height": 20})
    assert service._identity_quality(path, tiny).quality_score < 0.55
    no_landmarks = replace(face, landmarks=None)
    assert service._identity_quality(path, no_landmarks).quality_score < 0.55
    cv2.imwrite(str(path), np.full((80, 80, 3), 128, dtype=np.uint8))
    assert service._identity_quality(path, face).quality_score == 0


def test_face_association_uses_original_body_rectangle_not_padded_crop(monkeypatch, tmp_path):
    import cv2
    import numpy as np

    from app.models.media import Image

    # The padded crop may contain bystanders; its URL must never be passed to this extractor.
    raw = np.random.default_rng(5).integers(0, 255, (300, 200, 3), dtype=np.uint8)
    (tmp_path / "frames").mkdir()
    cv2.imwrite(str(tmp_path / "frames" / "raw.png"), raw)
    with _session(tmp_path) as db:
        image = Image(image_url="/data/frames/raw.png", source_type="stream_frame")
        db.add(image)
        db.flush()
        crop = PersonCrop(
            image_id=image.id,
            crop_url="/data/crops/with-bystander.jpg",
            bbox={"x": 50, "y": 20, "width": 80, "height": 240},
        )
        service = FaceRecognitionService(db, Settings(data_dir=tmp_path))

        def exact_rectangle(path, allow_fallback):
            assert not allow_fallback
            assert np.array_equal(cv2.imread(str(path)), raw[20:260, 50:130])
            return FaceCandidate([1.0], {"x": 10, "y": 5, "width": 60, "height": 60}, 0.9, "test")

        monkeypatch.setattr(service, "_best_candidate_path", exact_rectangle)
        assert service._identity_candidate_for_crop(crop) is not None
        monkeypatch.setattr(
            service,
            "_best_candidate_path",
            lambda *args, **kwargs: FaceCandidate(
                [1.0],
                {"x": 0, "y": 180, "width": 60, "height": 60},
                0.9,
                "test",
            ),
        )
        assert service._identity_candidate_for_crop(crop) is None


def test_person_labels_do_not_masquerade_as_face_evidence(tmp_path, monkeypatch):
    person = uuid.uuid4()
    query = _crop("query", person)
    same = ReidMatchItem(crop_id=uuid.uuid4(), score=0.46, person_id=person)
    different = ReidMatchItem(crop_id=uuid.uuid4(), score=0.92, person_id=uuid.uuid4())
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
    much_better_body = ReidMatchItem(crop_id=uuid.uuid4(), score=0.66, attribute_agreement=0.50)

    assert fusion_rank(much_better_body, much_better_body.score) > fusion_rank(
        slightly_better_labels, slightly_better_labels.score
    )


def test_crop_face_comparison_ignores_missing_or_low_quality_faces(monkeypatch, tmp_path):
    query_path = tmp_path / "query.jpg"
    strong = _crop("strong")
    low_quality = _crop("low")
    missing = _crop("missing")
    vectors = {
        strong.crop_url: FaceCandidate([0.8, 0.6], {}, 0.8, "test"),
        low_quality.crop_url: FaceCandidate([1.0, 0.0], {}, 0.4, "test"),
        missing.crop_url: None,
    }
    service = FaceRecognitionService(_session(tmp_path), Settings(data_dir=tmp_path))
    monkeypatch.setattr(
        service,
        "_best_candidate_path",
        lambda image_path, allow_fallback: FaceCandidate([1.0, 0.0], {}, 0.9, "test"),
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
        weak_query.crop_url: FaceCandidate([0.99, 0.1], {}, 0.8, "test"),
        strong_query.crop_url: FaceCandidate([1.0, 0.0], {}, 0.9, "test"),
        match.crop_url: FaceCandidate([0.8, 0.6], {}, 0.8, "test"),
    }
    service = FaceRecognitionService(_session(tmp_path), Settings(data_dir=tmp_path))
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
    assert compared[match.id].query_identity_verified is True


def test_candidate_extraction_is_cached_across_requests(monkeypatch, tmp_path):
    """A candidate crop already extracted for a prior search is not re-run."""

    query_path = tmp_path / "query.jpg"
    candidate = _crop("candidate")
    calls: list[str] = []

    def counting_extract(crop_url, allow_fallback):
        calls.append(crop_url)
        return FaceCandidate([0.8, 0.6], {}, 0.8, "test")

    session = _session(tmp_path)
    settings = Settings(data_dir=tmp_path)
    first_service = FaceRecognitionService(session, settings)
    monkeypatch.setattr(
        first_service,
        "_best_candidate_path",
        lambda p, allow_fallback: FaceCandidate([1.0, 0.0], {}, 0.9, "test"),
    )
    monkeypatch.setattr(first_service, "_best_candidate", counting_extract)

    first = first_service.compare_image_to_crops(query_path, [candidate], min_quality=0.55)
    assert calls == [candidate.crop_url]

    # A brand new service instance (a fresh request) reusing the same database session must
    # hit the persisted cache row instead of extracting again.
    second_service = FaceRecognitionService(session, settings)
    monkeypatch.setattr(
        second_service,
        "_best_candidate_path",
        lambda p, allow_fallback: FaceCandidate([1.0, 0.0], {}, 0.9, "test"),
    )
    monkeypatch.setattr(second_service, "_best_candidate", counting_extract)

    second = second_service.compare_image_to_crops(query_path, [candidate], min_quality=0.55)

    assert calls == [candidate.crop_url]  # still just the one extraction
    assert second[candidate.id].similarity == pytest.approx(first[candidate.id].similarity)

    cached_row = session.get(CropFaceExtraction, candidate.id)
    assert cached_row is not None
    assert cached_row.embedding == [0.8, 0.6]


def test_candidate_extraction_cache_is_invalidated_by_model_signature(monkeypatch, tmp_path):
    """Changing the extractor config (e.g. det_size) must not reuse a stale embedding."""

    query_path = tmp_path / "query.jpg"
    candidate = _crop("candidate")
    calls: list[str] = []

    def counting_extract(crop_url, allow_fallback):
        calls.append(crop_url)
        return FaceCandidate([0.8, 0.6], {}, 0.8, "test")

    session = _session(tmp_path)
    service = FaceRecognitionService(
        session, Settings(data_dir=tmp_path, face_insightface_det_size=640)
    )
    monkeypatch.setattr(
        service,
        "_best_candidate_path",
        lambda p, allow_fallback: FaceCandidate([1.0, 0.0], {}, 0.9, "test"),
    )
    monkeypatch.setattr(service, "_best_candidate", counting_extract)
    service.compare_image_to_crops(query_path, [candidate], min_quality=0.55)
    assert len(calls) == 1

    reconfigured = FaceRecognitionService(
        session, Settings(data_dir=tmp_path, face_insightface_det_size=320)
    )
    monkeypatch.setattr(
        reconfigured,
        "_best_candidate_path",
        lambda p, allow_fallback: FaceCandidate([1.0, 0.0], {}, 0.9, "test"),
    )
    monkeypatch.setattr(reconfigured, "_best_candidate", counting_extract)
    reconfigured.compare_image_to_crops(query_path, [candidate], min_quality=0.55)

    assert len(calls) == 2  # signature changed, so the stale row was not reused
