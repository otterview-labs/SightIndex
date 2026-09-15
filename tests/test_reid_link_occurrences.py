"""Camera links spend their existing face budget on independent, verified occurrences."""

import math
import uuid
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest


@pytest.fixture
def link_case(monkeypatch, tmp_path):
    from app.api import reid as api
    from app.config.settings import Settings
    from app.models.media import PersonCrop
    from app.schemas.reid import ReidFaceCoverage, ReidMatchItem
    from app.services import reid_fusion
    from app.services.faces import FaceCandidate, FaceCropComparison
    from app.services.reid_index import ReidMatch
    from app.services.vector_index import VectorIndexError

    settings = Settings(
        _env_file=None,
        data_dir=tmp_path,
        reid_face_priority_enabled=True,
        reid_face_candidate_limit=12,
        reid_attribute_filter_enabled=False,
        reid_collapse_window_seconds=60,
        reid_collapse_identity_threshold=0.7,
        reid_stature_weight=0,
    )
    source_camera, target_camera = uuid.uuid4(), uuid.uuid4()
    stamp = datetime(2026, 9, 8, 12)
    query = PersonCrop(
        id=uuid.uuid4(),
        image_id=uuid.uuid4(),
        camera_id=source_camera,
        captured_at=stamp,
        bbox={},
    )
    crops = {query.id: query}
    items = {}
    vectors = {}
    face_scores = {}
    checked = []
    extracted = []
    borrowed = {}
    fetched = []
    state = SimpleNamespace(vector_error=False, runtime_ready=True, query_status=None)

    def add(score, vector, face, *, camera=target_camera, second=0):
        crop = PersonCrop(
            id=uuid.uuid4(),
            image_id=query.image_id,
            camera_id=camera,
            captured_at=stamp + timedelta(seconds=second),
            crop_url="/data/synthetic.jpg",
            bbox={},
        )
        crops[crop.id] = crop
        items[crop.id] = ReidMatchItem(
            crop_id=crop.id,
            camera_id=camera,
            captured_at=crop.captured_at,
            score=score,
        )
        if vector is not None:
            vectors[crop.id] = vector
        face_scores[crop.id] = face
        return crop.id

    def fetch_vectors(_object_type, crop_ids):
        fetched.extend(crop_ids)
        if state.vector_error:
            raise VectorIndexError("synthetic outage")
        return {crop_id: vectors[crop_id] for crop_id in crop_ids if crop_id in vectors}

    service = SimpleNamespace(
        query_tracklet=lambda crop: [crop],
        search_by_crop_gallery=lambda *args, **kwargs: [
            ReidMatch(crop_id=item.crop_id, score=item.score) for item in items.values()
        ],
        candidate_pool_limit=lambda: len(items),
        index=SimpleNamespace(fetch_vectors=fetch_vectors),
    )
    db = SimpleNamespace(
        get=lambda _model, crop_id: crops.get(crop_id),
        scalars=lambda _statement: crops.values(),
    )

    def to_items(_db, _settings, matches):
        return [
            items[match.crop_id].model_copy(deep=True)
            if match.crop_id in items
            else ReidMatchItem(crop_id=query.id, score=1, camera_id=source_camera)
            for match in matches
        ]

    class FaceService:
        def __init__(self, _db, _settings):
            self.candidate_occurrences = {}
            self.coverage = ReidFaceCoverage()

        def prepare_person_crop_query(self, queries, **kwargs):
            if state.query_status is not None:
                self.coverage = ReidFaceCoverage(
                    status=state.query_status,
                    query_attempted_count=1,
                    query_absence_reasons={
                        "inference_error" if state.query_status == "error" else "no_face": 1
                    },
                )
                return None
            self.coverage = ReidFaceCoverage(
                query_attempted_count=1,
                query_face_found=True,
                query_face_quality=0.95,
                query_identity_verified=True,
            )
            return FaceCandidate([1, 0], {}, 0.95, "synthetic")

        def compare_prepared_person_crops(self, query_face, candidates, **kwargs):
            self.coverage = ReidFaceCoverage(
                query_face_found=True,
                query_face_quality=query_face.quality_score,
                query_identity_verified=kwargs["query_identity_verified"],
            )
            evidence = {}
            for candidate in candidates:
                checked.append(candidate.id)
                extracted.append(candidate.id)
                self.coverage.candidate_attempted_count += 1
                borrowed[candidate.id] = [
                    peer.id for peer in self.candidate_occurrences.get(candidate.id, [])
                ]
                score = face_scores[candidate.id]
                source_id = candidate.id
                if score is None:
                    reasons = self.coverage.candidate_absence_reasons
                    reasons["no_face"] = reasons.get("no_face", 0) + 1
                    for peer_id in borrowed[candidate.id]:
                        extracted.append(peer_id)
                        self.coverage.candidate_attempted_count += 1
                        peer_score = face_scores[peer_id]
                        if peer_score is None:
                            reasons["no_face"] = reasons.get("no_face", 0) + 1
                        elif score is None:
                            score, source_id = peer_score, peer_id
                    if score is None:
                        continue
                evidence[candidate.id] = FaceCropComparison(
                    similarity=score,
                    query_quality=0.95,
                    candidate_quality=0.9,
                    candidate_identity_verified=source_id == candidate.id,
                    candidate_source_crop_id=source_id,
                )
            self.coverage.compared_count = len(evidence)
            self.coverage.borrowed_candidate_count = sum(
                not comparison.candidate_identity_verified for comparison in evidence.values()
            )
            self.coverage.status = "compared" if evidence else "candidate_unavailable"
            return evidence

    monkeypatch.setattr(api, "_service", lambda *args: service)
    monkeypatch.setattr(api, "_to_items", to_items)
    monkeypatch.setattr(api, "_query_tracklet_attributes", lambda *args: None)
    monkeypatch.setattr(reid_fusion, "FaceRecognitionService", FaceService)
    monkeypatch.setattr(
        reid_fusion,
        "face_runtime_status",
        lambda _settings: SimpleNamespace(ready=state.runtime_ready),
    )
    return SimpleNamespace(
        add=add,
        run=lambda: api.reid_camera_links(query.id, db, settings),
        settings=settings,
        vectors=vectors,
        checked=checked,
        extracted=extracted,
        borrowed=borrowed,
        fetched=fetched,
        state=state,
        target_camera=target_camera,
        source_camera=source_camera,
        query=query,
        service=service,
    )


def test_thirteen_repeated_frames_cannot_hide_a_matching_face(link_case):
    case = link_case
    wrong = [case.add(0.6 - i * 0.001, [0.6, 0.8, 0], 0.1, second=i) for i in range(13)]
    correct = case.add(0.48, [0.48, 0, math.sqrt(1 - 0.48**2)], 0.9, second=13)

    response = case.run()

    assert [link.crop_id for link in response.links] == [correct]
    assert response.links[0].face_match is True
    assert response.links[0].score == 0.48
    assert case.checked == [wrong[0], correct]
    assert case.borrowed[wrong[0]] == wrong[1:3]
    assert case.borrowed[correct] == []  # nearby strangers never become borrowed face sources
    coverage = response.face_coverage
    assert coverage.status == "compared"
    assert coverage.shortlist_count == coverage.candidate_attempted_count == 2
    assert coverage.compared_count == 2
    assert coverage.hard_conflict_count == coverage.hard_match_count == 1
    assert coverage.query_attempted_count == 1  # prepared once and reused by every round
    assert "occurrence_crop_ids" not in response.model_dump_json()


@pytest.mark.parametrize("mode", ["missing", "partial", "outage", "gate_disabled", "window_zero"])
def test_unverified_neighbours_remain_separate_and_do_not_inherit_conflicts(link_case, mode):
    case = link_case
    wrong = case.add(0.6, [1, 0], 0.1)
    correct = case.add(0.59, [1, 0], 0.9, second=1)
    if mode == "missing":
        case.vectors.clear()
    elif mode == "partial":
        case.vectors.pop(correct)
    elif mode == "outage":
        case.state.vector_error = True
    elif mode == "gate_disabled":
        case.settings.reid_collapse_identity_threshold = 0
    else:
        case.settings.reid_collapse_window_seconds = 0

    response = case.run()

    assert [link.crop_id for link in response.links] == [correct]
    assert case.checked == [wrong, correct]
    assert case.borrowed == {wrong: [], correct: []}
    assert response.face_coverage.hard_conflict_count == 1
    assert response.face_coverage.hard_match_count == 1


def test_unknown_and_source_cameras_do_not_consume_link_face_budget(link_case):
    case = link_case
    case.add(0.99, [1, 0], 0.9, camera=None)
    case.add(0.98, [1, 0], 0.9, camera=case.source_camera)
    target = case.add(0.48, [1, 0], 0.9)

    response = case.run()

    assert case.checked == case.fetched == [target]
    assert [link.crop_id for link in response.links] == [target]


def test_refill_never_exceeds_the_original_total_budget(link_case):
    case = link_case
    case.settings.reid_face_candidate_limit = 3
    conflicts = [case.add(0.6 - i * 0.01, [1, 0], 0.1, second=i * 120) for i in range(3)]
    unexamined = case.add(0.48, [1, 0], 0.9, second=360)

    response = case.run()

    assert case.checked == conflicts
    assert response.face_coverage.shortlist_count == 3
    assert response.face_coverage.candidate_attempted_count == 3
    assert response.face_coverage.hard_conflict_count == 3
    # Exhausting the budget does not turn an unexamined independent visit into a face verdict.
    assert [link.crop_id for link in response.links] == [unexamined]
    assert response.links[0].face_match is None
    assert response.links[0].evidence_level == "similar"


def test_matched_camera_releases_remaining_budget_to_other_cameras(link_case):
    case = link_case
    case.settings.reid_face_candidate_limit = 3
    first = case.add(0.7, [1, 0], 0.9)
    case.add(0.69, [1, 0], 0.9, second=120)
    other_camera = uuid.uuid4()
    conflict = case.add(0.6, [1, 0], 0.1, camera=other_camera)
    recovered = case.add(0.48, [1, 0], 0.9, camera=other_camera, second=120)

    response = case.run()

    assert case.checked == [first, conflict, recovered]
    assert {link.crop_id for link in response.links} == {first, recovered}
    assert response.face_coverage.shortlist_count == 3
    assert response.face_coverage.hard_match_count == 2
    assert response.face_coverage.hard_conflict_count == 1


def test_refill_accumulates_absence_reasons_and_stops_after_a_match(link_case):
    case = link_case
    missing = [case.add(0.6, [1, 0], None, second=i * 120) for i in range(2)]
    match = case.add(0.48, [1, 0], 0.9, second=240)
    case.add(0.47, [1, 0], 0.9, second=360)

    response = case.run()

    assert case.checked == [*missing, match]
    assert response.face_coverage.status == "compared"
    assert response.face_coverage.candidate_absence_reasons == {"no_face": 2}
    assert response.face_coverage.shortlist_count == 3
    assert response.face_coverage.candidate_attempted_count == 3
    assert response.face_coverage.query_attempted_count == 1
    assert response.face_coverage.compared_count == 1
    assert [link.crop_id for link in response.links] == [match]


def test_unavailable_runtime_stops_refill_without_face_claims(link_case):
    case = link_case
    case.state.runtime_ready = False
    best = case.add(0.6, [1, 0], 0.9)
    case.add(0.48, [1, 0], 0.9, second=120)

    response = case.run()

    assert case.checked == []
    assert response.face_coverage.status == "unavailable"
    assert response.face_coverage.shortlist_count == 0
    assert response.face_coverage.candidate_attempted_count == 0
    assert case.fetched == []
    assert [link.crop_id for link in response.links] == [best]
    assert response.links[0].face_match is None


@pytest.mark.parametrize("status", ["query_unavailable", "error"])
def test_missing_or_failed_query_face_stops_all_refill(link_case, status):
    case = link_case
    case.state.query_status = status
    best = case.add(0.6, [1, 0], 0.9)
    case.add(0.48, [1, 0], 0.9, second=120)

    response = case.run()

    assert case.checked == []
    assert response.face_coverage.status == status
    assert response.face_coverage.shortlist_count == 0
    assert response.face_coverage.query_attempted_count == 1
    assert response.face_coverage.candidate_attempted_count == 0
    assert case.fetched == []
    assert sum(response.face_coverage.query_absence_reasons.values()) == 1
    assert [link.crop_id for link in response.links] == [best]
    assert response.links[0].face_match is None


def test_later_absent_faces_do_not_erase_an_earlier_comparison(link_case):
    case = link_case
    matched = case.add(0.6, [1, 0], 0.9)
    other_camera = uuid.uuid4()
    missing = case.add(0.5, [1, 0], None, camera=other_camera)
    case.add(0.48, [1, 0], None, camera=other_camera, second=120)

    response = case.run()

    assert response.face_coverage.status == "compared"
    assert response.face_coverage.query_face_found is True
    assert response.face_coverage.query_identity_verified is True
    assert response.face_coverage.query_face_quality == 0.95
    assert response.face_coverage.compared_count == 1
    assert response.face_coverage.hard_match_count == 1
    assert response.face_coverage.candidate_absence_reasons == {"no_face": 2}
    assert response.face_coverage.shortlist_count == 3
    assert {link.crop_id for link in response.links} == {matched, missing}


@pytest.mark.parametrize("budget", [1, 2, 3])
def test_borrowed_member_frames_share_the_total_candidate_attempt_budget(link_case, budget):
    case = link_case
    case.settings.reid_face_candidate_limit = budget
    representative = case.add(0.6, [1, 0], None)
    peer = case.add(0.59, [1, 0], 0.9, second=1)
    last_peer = case.add(0.58, [1, 0], None, second=2)
    case.add(0.48, [0, 1], 0.9)

    response = case.run()

    assert case.checked == [representative]
    assert case.extracted == [representative, peer, last_peer][:budget]
    assert response.face_coverage.candidate_attempted_count == budget
    assert response.face_coverage.shortlist_count == 1
    assert response.links[0].face_match is None  # a borrowed face is still soft evidence only
    assert response.face_coverage.borrowed_candidate_count == (1 if budget > 1 else 0)


def test_real_face_extraction_errors_stop_refill_and_preserve_comparisons(link_case, monkeypatch):
    from app.services import reid_fusion
    from app.services.faces import FaceCandidate, FaceRecognitionService

    case = link_case
    matched = case.add(0.7, [1, 0], 0.9)
    failing_camera = uuid.uuid4()
    failed = case.add(0.6, [1, 0], None, camera=failing_camera)
    case.add(0.5, [1, 0], None, camera=failing_camera, second=120)
    extracted = []

    def cached_candidate(_service, crop):
        extracted.append(crop.id)
        if crop.id in {case.query.id, matched}:
            return FaceCandidate([1, 0], {}, 0.95, "synthetic")
        raise ValueError("synthetic inference failure")

    monkeypatch.setattr(reid_fusion, "FaceRecognitionService", FaceRecognitionService)
    monkeypatch.setattr(FaceRecognitionService, "_cached_strict_candidate", cached_candidate)
    monkeypatch.setattr(FaceRecognitionService, "_commit_crop_face_cache", lambda self: None)

    response = case.run()

    assert extracted == [case.query.id, matched, failed]
    assert response.face_coverage.status == "error"
    assert response.face_coverage.shortlist_count == 2
    assert response.face_coverage.query_attempted_count == 1
    assert response.face_coverage.candidate_attempted_count == 2
    assert response.face_coverage.compared_count == 1
    assert response.face_coverage.hard_match_count == 1
    assert response.face_coverage.candidate_absence_reasons == {"inference_error": 1}
    link = next(link for link in response.links if link.crop_id == matched)
    assert link.face_match is True


def test_real_query_without_face_skips_candidate_vectors_and_collapse(link_case, monkeypatch):
    from app.api import reid as api
    from app.services import reid_fusion
    from app.services.faces import FaceRecognitionService

    case = link_case
    best = case.add(0.6, [1, 0], None)
    for i in range(13):
        case.add(0.59 - i * 0.001, [1, 0], None, second=i)
    extracted = []
    cache_commits = []

    def no_face(service, crop):
        extracted.append(crop.id)
        return service._abstain("no_face")

    def unexpected_collapse(*args, **kwargs):
        pytest.fail("an unavailable query face must not trigger candidate occurrence collapse")

    monkeypatch.setattr(reid_fusion, "FaceRecognitionService", FaceRecognitionService)
    monkeypatch.setattr(FaceRecognitionService, "_cached_strict_candidate", no_face)
    monkeypatch.setattr(
        FaceRecognitionService, "_commit_crop_face_cache", lambda self: cache_commits.append(True)
    )
    monkeypatch.setattr(api, "collapse_occurrences", unexpected_collapse)

    response = case.run()

    assert extracted == [case.query.id]
    assert cache_commits == [True]
    assert case.fetched == []
    assert response.face_coverage.status == "query_unavailable"
    assert response.face_coverage.query_attempted_count == 1
    assert response.face_coverage.query_absence_reasons == {"no_face": 1}
    assert response.face_coverage.shortlist_count == 0
    assert response.face_coverage.candidate_attempted_count == 0
    assert [link.crop_id for link in response.links] == [best]
    assert response.links[0].score == 0.6
    assert response.links[0].face_match is None


def test_real_face_query_is_prepared_once_for_multiple_candidate_rounds(link_case, monkeypatch):
    from app.services import reid_fusion
    from app.services.faces import FaceCandidate, FaceRecognitionService

    case = link_case
    wrong = case.add(0.6, [1, 0], 0.1)
    correct = case.add(0.48, [0, 1], 0.9)
    extracted = []

    def cached_candidate(_service, crop):
        extracted.append(crop.id)
        vector = [0, 1] if crop.id == wrong else [1, 0]
        return FaceCandidate(vector, {}, 0.95, "synthetic")

    monkeypatch.setattr(reid_fusion, "FaceRecognitionService", FaceRecognitionService)
    monkeypatch.setattr(FaceRecognitionService, "_cached_strict_candidate", cached_candidate)
    monkeypatch.setattr(FaceRecognitionService, "_commit_crop_face_cache", lambda self: None)

    response = case.run()

    assert extracted == [case.query.id, wrong, correct]
    assert case.fetched == [wrong, correct]
    assert response.face_coverage.query_attempted_count == 1
    assert response.face_coverage.candidate_attempted_count == 2
    assert response.face_coverage.compared_count == 2
    assert response.face_coverage.hard_match_count == 1
    assert response.face_coverage.hard_conflict_count == 1
    assert [link.crop_id for link in response.links] == [correct]


@pytest.mark.parametrize("candidate_vector", [[1, 0], [0, 1]])
def test_prepared_borrowed_query_face_cannot_make_hard_decisions(
    link_case, monkeypatch, candidate_vector
):
    from app.models.media import PersonCrop
    from app.services import reid_fusion
    from app.services.faces import FaceCandidate, FaceRecognitionService

    case = link_case
    candidate = case.add(0.6, [1, 0], 0.9)
    neighbour = PersonCrop(id=uuid.uuid4(), image_id=case.query.image_id, bbox={})
    case.service.query_tracklet = lambda crop: [crop, neighbour]
    extracted = []

    def cached_candidate(service, crop):
        extracted.append(crop.id)
        if crop.id == case.query.id:
            return service._abstain("no_face")
        vector = [1, 0] if crop.id == neighbour.id else candidate_vector
        return FaceCandidate(vector, {}, 0.95, "synthetic")

    monkeypatch.setattr(reid_fusion, "FaceRecognitionService", FaceRecognitionService)
    monkeypatch.setattr(FaceRecognitionService, "_cached_strict_candidate", cached_candidate)
    monkeypatch.setattr(FaceRecognitionService, "_commit_crop_face_cache", lambda self: None)

    response = case.run()

    assert extracted == [case.query.id, neighbour.id, candidate]
    assert response.face_coverage.query_attempted_count == 2
    assert response.face_coverage.query_face_found is True
    assert response.face_coverage.query_identity_verified is False
    assert response.face_coverage.query_absence_reasons == {"no_face": 1}
    assert response.face_coverage.hard_match_count == 0
    assert response.face_coverage.hard_conflict_count == 0
    assert [link.crop_id for link in response.links] == [candidate]
    assert response.links[0].face_match is None
