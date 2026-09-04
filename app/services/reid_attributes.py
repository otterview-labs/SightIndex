from copy import deepcopy
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AttributeEvidence:
    value: object
    confidence: float


@dataclass(frozen=True)
class AttributeCompatibility:
    compared: tuple[str, ...]
    matched: tuple[str, ...]
    conflicted: tuple[str, ...]
    weights: tuple[tuple[str, float], ...]

    @property
    def agreement(self) -> float | None:
        total = self.evidence_weight
        if total <= 0:
            return None
        matched = set(self.matched)
        return sum(weight for field, weight in self.weights if field in matched) / total

    @property
    def evidence_weight(self) -> float:
        return sum(weight for _field, weight in self.weights)

    @property
    def conflict_weight(self) -> float:
        conflicted = set(self.conflicted)
        return sum(weight for field, weight in self.weights if field in conflicted)

    @property
    def compared_count(self) -> int:
        return len(self.compared)

    @property
    def match_count(self) -> int:
        return len(self.matched)

    @property
    def conflict_count(self) -> int:
        return len(self.conflicted)


# These fields describe appearance that normally survives a short cross-camera journey.
# Behaviours such as phone use, smoking and falling deliberately do not participate in identity.
_FIELDS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "upper_color": (("clothing", "upper_color"), ("clothing", "upper_color_confidence")),
    "lower_color": (("clothing", "lower_color"), ("clothing", "lower_color_confidence")),
    "upper_length": (("clothing", "upper_length"), ("clothing", "upper_length_confidence")),
    "lower_length": (("clothing", "lower_length"), ("clothing", "lower_length_confidence")),
    "hat": (("appearance", "hat"), ("appearance", "hat_confidence")),
    "glasses": (("appearance", "glasses"), ("appearance", "glasses_confidence")),
    "backpack": (("objects", "backpack"), ("objects", "backpack_confidence")),
}


def compare_reid_attributes(
    query: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
    *,
    min_confidence: float,
) -> AttributeCompatibility:
    query_profile = _profile(query, min_confidence)
    candidate_profile = _profile(candidate, min_confidence)
    compared: list[str] = []
    matched: list[str] = []
    conflicted: list[str] = []
    weights: list[tuple[str, float]] = []
    for field in _FIELDS:
        left = query_profile.get(field)
        right = candidate_profile.get(field)
        if left is None or right is None:
            continue
        compared.append(field)
        weights.append((field, min(left.confidence, right.confidence)))
        if _compatible(field, left.value, right.value):
            matched.append(field)
        else:
            conflicted.append(field)
    return AttributeCompatibility(
        tuple(compared), tuple(matched), tuple(conflicted), tuple(weights)
    )


def aggregate_reid_attributes(
    samples: list[dict[str, Any] | None],
    *,
    min_confidence: float,
) -> dict[str, Any] | None:
    """Combine stable query labels across a tracklet without inventing missing evidence.

    Values vote with their own confidence. A tied field is omitted, while a majority keeps the
    mean confidence of the supporting frames with a small disagreement penalty. Geometry and
    stature stay attached to the selected source crop because they describe that exact frame.
    """

    usable = [sample for sample in samples if isinstance(sample, dict)]
    if not usable:
        return None
    result = deepcopy(usable[0])
    result["source"] = "tracklet"
    profiles = [_profile(sample, min_confidence) for sample in usable]
    for field, (value_path, confidence_path) in _FIELDS.items():
        _delete_nested(result, value_path)
        _delete_nested(result, confidence_path)
        votes: dict[object, list[float]] = {}
        for profile in profiles:
            evidence = profile.get(field)
            if evidence is not None:
                votes.setdefault(evidence.value, []).append(evidence.confidence)
        if not votes:
            continue
        ranked = sorted(
            votes.items(),
            key=lambda pair: (sum(pair[1]), len(pair[1])),
            reverse=True,
        )
        if len(ranked) > 1 and sum(ranked[0][1]) == sum(ranked[1][1]):
            continue
        value, confidences = ranked[0]
        total_weight = sum(sum(items) for items in votes.values())
        dominance = sum(confidences) / total_weight if total_weight else 0.0
        confidence = (sum(confidences) / len(confidences)) * (0.8 + 0.2 * dominance)
        _assign_nested(result, value_path, value)
        _assign_nested(result, confidence_path, round(confidence, 4))
    return result


def _profile(
    attributes: dict[str, Any] | None,
    min_confidence: float,
) -> dict[str, AttributeEvidence]:
    if not isinstance(attributes, dict):
        return {}
    source = str(attributes.get("source") or "").lower()
    default_confidence = 0.55 if source == "cv_tone" else 0.0
    profile: dict[str, AttributeEvidence] = {}
    for field, (value_path, confidence_path) in _FIELDS.items():
        value = _nested(attributes, value_path)
        value = _normalize(value)
        if value is None:
            continue
        confidence = _number(_nested(attributes, confidence_path))
        if confidence is None:
            confidence = default_confidence
        if confidence < min_confidence:
            continue
        profile[field] = AttributeEvidence(value=value, confidence=confidence)
    return profile


def _compatible(field: str, left: object, right: object) -> bool:
    if left == right:
        return True
    if field not in {"upper_color", "lower_color"}:
        return False
    # Tone-only labels are broad evidence, not contradictions to a specific hue.
    dark = {"black", "brown", "gray", "blue", "green", "purple"}
    light = {"white", "yellow", "pink", "orange"}
    return (
        (left == "dark" and right in dark)
        or (right == "dark" and left in dark)
        or (left == "light" and right in light)
        or (right == "light" and left in light)
    )


def _nested(data: dict[str, Any], path: tuple[str, ...]) -> object | None:
    current: object = data
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _assign_nested(data: dict[str, Any], path: tuple[str, ...], value: object) -> None:
    current = data
    for key in path[:-1]:
        child = current.get(key)
        if not isinstance(child, dict):
            child = {}
            current[key] = child
        current = child
    current[path[-1]] = value


def _delete_nested(data: dict[str, Any], path: tuple[str, ...]) -> None:
    current: object = data
    for key in path[:-1]:
        if not isinstance(current, dict):
            return
        current = current.get(key)
    if isinstance(current, dict):
        current.pop(path[-1], None)


def _normalize(value: object) -> object | None:
    if isinstance(value, str):
        value = value.strip().lower()
        return None if value in {"", "unknown", "null", "none"} else value
    if isinstance(value, bool):
        return value
    return value if isinstance(value, int | float) else None


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return max(0.0, min(1.0, float(value)))
