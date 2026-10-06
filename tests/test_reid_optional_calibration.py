"""Camera-specific calibration remains optional, display-only and numerically safe."""

import uuid

import pytest
from pydantic import ValidationError

from app.config.settings import Settings
from app.schemas.reid import ReidMatchItem
from app.services.reid_fusion import (
    annotate_fusion_decision,
    cross_camera_match_probability,
    fusion_rank,
)


def configured_settings(**overrides: float | None) -> Settings:
    values = {
        "reid_cross_camera_calibration_coef": 2.0,
        "reid_cross_camera_calibration_intercept": -1.0,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


@pytest.mark.parametrize("missing", ["coef", "intercept", "both"])
def test_missing_calibration_is_neutral(missing: str) -> None:
    overrides = {}
    if missing in {"coef", "both"}:
        overrides["reid_cross_camera_calibration_coef"] = None
    if missing in {"intercept", "both"}:
        overrides["reid_cross_camera_calibration_intercept"] = None
    assert cross_camera_match_probability(0.5, configured_settings(**overrides)) is None


@pytest.mark.parametrize("coefficient, expected", [(1e308, 1.0), (-1e308, 0.0), (0.0, 0.5)])
def test_stable_sigmoid(coefficient: float, expected: float) -> None:
    settings = configured_settings(
        reid_cross_camera_calibration_coef=coefficient,
        reid_cross_camera_calibration_intercept=0.0,
    )
    assert cross_camera_match_probability(2.0, settings) == expected


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_score_has_no_estimate(score: float) -> None:
    assert cross_camera_match_probability(score, configured_settings()) is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("name", ["coef", "intercept"])
def test_configuration_rejects_nonfinite_coefficients(name: str, value: float) -> None:
    with pytest.raises(ValidationError):
        configured_settings(**{f"reid_cross_camera_calibration_{name}": value})


def test_only_known_cross_camera_pairs_receive_estimate_and_ranking_is_unchanged() -> None:
    query_camera = uuid.uuid4()
    other_camera = uuid.uuid4()
    item = ReidMatchItem(crop_id=uuid.uuid4(), camera_id=other_camera, score=0.5)
    previous_rank = fusion_rank(item, 0.5)
    annotate_fusion_decision(
        item,
        0.5,
        query_camera=query_camera,
        chance_ceiling=0.44,
        settings=configured_settings(),
    )
    assert item.calibrated_match_probability == 0.5
    assert fusion_rank(item, 0.5) == previous_rank
    assert item.evidence_level == "similar"
    for next_camera in (other_camera, None):
        annotate_fusion_decision(
            item,
            0.5,
            query_camera=next_camera,
            chance_ceiling=0.44,
            settings=configured_settings(),
        )
        assert item.calibrated_match_probability is None


def test_unknown_candidate_camera_receives_no_probability() -> None:
    item = ReidMatchItem(crop_id=uuid.uuid4(), camera_id=None, score=0.5)
    annotate_fusion_decision(
        item,
        0.5,
        query_camera=uuid.uuid4(),
        chance_ceiling=0.44,
        settings=configured_settings(),
    )
    assert item.calibrated_match_probability is None


@pytest.mark.parametrize("face_match, level", [(True, "reliable"), (False, "rejected")])
def test_face_decision_keeps_priority(face_match: bool, level: str) -> None:
    item = ReidMatchItem(
        crop_id=uuid.uuid4(), camera_id=uuid.uuid4(), score=0.1, face_match=face_match
    )
    annotate_fusion_decision(
        item,
        0.1,
        query_camera=uuid.uuid4(),
        chance_ceiling=0.44,
        settings=configured_settings(),
        is_camera_link=True,
    )
    assert item.evidence_level == level
