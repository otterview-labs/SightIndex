from app.config.settings import Settings
from app.services.vlm import VLMStructuredAnalysisService


def test_low_confidence_person_attributes_are_treated_as_unknown():
    service = VLMStructuredAnalysisService(Settings(vlm_structured_min_confidence=0.6))

    attributes = service._normalize_person_attributes(
        {
            "appearance": {
                "hat": True,
                "hat_confidence": 0.4,
                "glasses": False,
                "glasses_confidence": 0.9,
            },
            "clothing": {
                "upper_color": "blue",
                "upper_color_confidence": 0.3,
                "lower_color": "black",
                "lower_color_confidence": 0.8,
            },
            "objects": {"backpack": True, "backpack_confidence": 0.5},
            "behavior": {"smoking": True, "smoking_confidence": 0.2},
        }
    )

    assert attributes["has_hat"] is None
    assert attributes["appearance"]["hat"] is None
    assert attributes["has_glasses"] is False
    assert attributes["top_color"] is None
    assert attributes["clothing"]["upper_color"] == "unknown"
    assert attributes["bottom_color"] == "black"
    assert attributes["objects"]["backpack"] is None
    assert attributes["behavior"]["smoking"] is None


def test_missing_confidence_keeps_backward_compatible_values():
    service = VLMStructuredAnalysisService(Settings(vlm_structured_min_confidence=0.9))

    attributes = service._normalize_person_attributes(
        {
            "appearance": {"hat": True},
            "clothing": {"upper_color": "red"},
            "objects": {"backpack": False},
        }
    )

    assert attributes["has_hat"] is True
    assert attributes["top_color"] == "red"
    assert attributes["has_backpack"] is False
