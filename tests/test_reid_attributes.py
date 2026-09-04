from app.config.settings import Settings
from app.services.reid_attributes import aggregate_reid_attributes, compare_reid_attributes
from app.services.structured_attributes import StructuredAttributeService


def _attributes(
    *,
    upper: str = "black",
    lower: str = "black",
    backpack: bool = True,
    confidence: float = 0.9,
    falling: bool = False,
) -> dict:
    return {
        "clothing": {
            "upper_color": upper,
            "lower_color": lower,
            "upper_color_confidence": confidence,
            "lower_color_confidence": confidence,
        },
        "objects": {"backpack": backpack, "backpack_confidence": confidence},
        "behavior": {"falling": falling, "falling_confidence": confidence},
    }


def test_reid_attributes_report_stable_matches_and_conflicts():
    result = compare_reid_attributes(
        _attributes(),
        _attributes(upper="white", lower="blue", backpack=True),
        min_confidence=0.75,
    )

    assert result.matched == ("backpack",)
    assert result.conflicted == ("upper_color", "lower_color")
    assert result.agreement == 1 / 3


def test_reid_attributes_ignore_low_confidence_and_behavior():
    result = compare_reid_attributes(
        _attributes(falling=True),
        _attributes(upper="white", lower="white", backpack=False, confidence=0.4),
        min_confidence=0.75,
    )

    assert result.compared == ()
    assert result.conflicted == ()
    assert result.agreement is None


def test_reid_attributes_treat_broad_tone_as_compatible():
    query = _attributes(upper="black", lower="white")
    candidate = _attributes(upper="dark", lower="light")

    result = compare_reid_attributes(query, candidate, min_confidence=0.75)

    assert result.matched == ("upper_color", "lower_color", "backpack")
    assert result.conflicted == ()
    assert result.agreement == 1.0


def test_reid_attributes_weight_agreement_by_the_weaker_confidence():
    query = _attributes(confidence=0.99)
    candidate = _attributes(confidence=0.95, upper="white")
    candidate["clothing"]["upper_color_confidence"] = 0.76

    result = compare_reid_attributes(query, candidate, min_confidence=0.75)

    assert result.evidence_weight == 0.76 + 0.95 + 0.95
    assert result.conflict_weight == 0.76
    assert result.agreement == (0.95 + 0.95) / (0.76 + 0.95 + 0.95)


def test_tracklet_attributes_vote_across_frames():
    first = _attributes(upper="black", lower="white", backpack=True, confidence=0.9)
    second = _attributes(upper="black", lower="blue", backpack=True, confidence=0.8)
    third = _attributes(upper="white", lower="blue", backpack=False, confidence=0.8)

    result = aggregate_reid_attributes(
        [first, second, third],
        min_confidence=0.75,
    )

    assert result is not None
    assert result["source"] == "tracklet"
    assert result["clothing"]["upper_color"] == "black"
    assert result["clothing"]["lower_color"] == "blue"
    assert result["objects"]["backpack"] is True
    assert result["clothing"]["upper_color_confidence"] >= 0.75


def test_vlm_attributes_preserve_pose_geometry_not_returned_by_model():
    merged = StructuredAttributeService._merge_existing_geometry(
        {
            "source": "cv_tone",
            "clothing": {"upper_color": "dark", "lower_length": "long"},
            "facing": "front",
            "stature": {"percentile": 72, "band": "tall"},
        },
        {
            "object_type": "person",
            "clothing": {"upper_color": "black", "lower_color": "blue"},
            "objects": {"backpack": True},
        },
    )

    assert merged["source"] == "vlm"
    assert merged["clothing"] == {
        "upper_color": "black",
        "lower_color": "blue",
        "lower_length": "long",
    }
    assert merged["facing"] == "front"
    assert merged["stature"] == {"percentile": 72, "band": "tall"}


def test_structured_backfill_upgrades_tone_rows_and_skips_existing_vlm(
    monkeypatch, tmp_path
):
    class Query:
        def order_by(self, _column):
            return self

        def all(self):
            return [already_vlm, tone_only, missing]

    class Db:
        def query(self, _model):
            return Query()

    already_vlm = type("Crop", (), {"attributes": {"source": "vlm"}})()
    tone_only = type("Crop", (), {"attributes": {"source": "cv_tone"}})()
    missing = type("Crop", (), {"attributes": None})()
    service = StructuredAttributeService(Db(), Settings(data_dir=tmp_path))
    analyzed = []
    monkeypatch.setattr(
        service,
        "analyze_person_crop",
        lambda crop, persist: analyzed.append(crop) or {"source": "vlm"},
    )

    seen, updated, errors = service.analyze_unparsed_person_crops(10)

    assert (seen, updated, errors) == (2, 2, [])
    assert analyzed == [tone_only, missing]
