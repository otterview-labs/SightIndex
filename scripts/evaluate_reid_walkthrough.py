"""Evaluate exported human feedback offline, never inference or production state.

Only explicit CSV same_person verdicts are ground truth. Scores are evidence
snapshots, never labels. This module imports no Settings, database, image, network
or model code. Suggestions never change config. Pair metrics are not Rank-1/mAP
or an independently verified identity-disjoint test.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import stat
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MAX_CSV_BYTES = 32 * 1024 * 1024
MAX_PAIRS = 100_000
MIN_POSITIVE_PAIRS = 30
MIN_NEGATIVE_PAIRS = 60
MANUAL_SOURCES = frozenset({"search", "camera_link", "manual", "walkthrough"})


class EvaluationError(ValueError):
    """A bounded reason code, never pair IDs or untrusted CSV contents."""


@dataclass(frozen=True)
class FeedbackPair:
    """One explicit directed human verdict with optional snapshot evidence."""

    query_id: uuid.UUID
    candidate_id: uuid.UUID
    same_person: bool
    body_score: float | None
    face_score: float | None
    face_reliability: float | None
    conflict_count: int | None
    comparable_count: int | None
    group: str | None


@dataclass(frozen=True)
class FeedbackData:
    """Validated feedback and content-only dataset identity."""

    pairs: tuple[FeedbackPair, ...]
    sha256: str
    deduplicated_rows: int


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise EvaluationError(code)


def _boolean(value: str, field: str) -> bool:
    """Uncertainty must never silently become a negative label."""
    value = value.strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise EvaluationError(f"invalid_{field}")


def _number(row: dict[str, str], field: str, lower: float, upper: float) -> float | None:
    raw = row.get(field, "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        raise EvaluationError(f"invalid_{field}") from None
    _require(math.isfinite(value) and lower <= value <= upper, f"invalid_{field}")
    return value


def _integer(row: dict[str, str], field: str) -> int | None:
    raw = row.get(field, "").strip()
    if not raw:
        return None
    _require(re.fullmatch(r"[0-9]{1,2}", raw) is not None, f"invalid_{field}")
    value = int(raw)
    _require(value <= 20, f"invalid_{field}")
    return value


def read_feedback(path: Path, group_column: str | None = None) -> FeedbackData:
    """Read a bounded regular CSV, rejecting ambiguous or contradictory verdicts."""
    _require(not path.is_symlink(), "input_link")
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as handle:
        _require(stat.S_ISREG(os.fstat(handle.fileno()).st_mode), "input_not_regular")
        raw = handle.read(MAX_CSV_BYTES + 1)
    _require(len(raw) <= MAX_CSV_BYTES, "input_too_large")
    try:
        contents = raw.decode("utf-8-sig")
    except UnicodeError:
        raise EvaluationError("invalid_csv_encoding") from None
    reader = csv.DictReader(io.StringIO(contents), strict=True)
    try:
        fields = reader.fieldnames
        _require(fields is not None, "missing_csv_header")
        assert fields is not None
        _require(len(fields) == len(set(fields)), "duplicate_csv_header")
        _require(
            {"query_crop_id", "candidate_crop_id", "same_person"} <= set(fields),
            "missing_required_columns",
        )
        if group_column is not None:
            _require(group_column in fields, "missing_group_column")
        pairs: dict[tuple[uuid.UUID, uuid.UUID], FeedbackPair] = {}
        truths: dict[frozenset[uuid.UUID], bool] = {}
        duplicates = 0
        for row_number, original in enumerate(reader, 1):
            _require(row_number <= MAX_PAIRS, "too_many_pairs")
            _require(
                None not in original and all(v is not None for v in original.values()),
                "malformed_csv_row",
            )
            row = {key: str(value) for key, value in original.items()}
            label = _boolean(row["same_person"], "same_person")
            try:
                query = uuid.UUID(row["query_crop_id"].strip())
                candidate = uuid.UUID(row["candidate_crop_id"].strip())
            except ValueError:
                raise EvaluationError("invalid_crop_id") from None
            _require(query != candidate, "self_pair")
            source = row.get("source", "").strip().lower()
            _require(not source or source in MANUAL_SOURCES, "non_manual_label_source")
            for field, lower, upper in (
                ("attribute_agreement", 0.0, 1.0),
                ("fusion_score", -2.0, 2.0),
            ):
                _number(row, field, lower, upper)
            for field in (
                "face_match",
                "face_query_identity_verified",
                "face_candidate_identity_verified",
            ):
                if row.get(field, "").strip():
                    _boolean(row[field], field)
            _integer(row, "attribute_match_count")
            conflicts = _integer(row, "attribute_conflict_count")
            comparable = _integer(row, "attribute_comparable_count")
            _require(
                conflicts is None or comparable is None or conflicts <= comparable,
                "invalid_attribute_counts",
            )
            group = row.get(group_column, "").strip() if group_column is not None else None
            _require(group is None or bool(group), "missing_group_value")
            pair = FeedbackPair(
                query,
                candidate,
                label,
                _number(row, "body_score", -1.0, 1.0),
                _number(row, "face_similarity", -1.0, 1.0),
                _number(row, "face_reliability", 0.0, 1.0),
                conflicts,
                comparable,
                group,
            )
            identity_pair = frozenset({query, candidate})
            if identity_pair in truths:
                _require(truths[identity_pair] == label, "contradictory_pair_truth")
            truths[identity_pair] = label
            key = (query, candidate)
            if key in pairs:
                _require(pairs[key] == pair, "conflicting_duplicate_pair")
                duplicates += 1
            else:
                pairs[key] = pair
        _require(bool(pairs), "no_labelled_pairs")
    except csv.Error:
        raise EvaluationError("invalid_csv") from None
    return FeedbackData(tuple(pairs.values()), hashlib.sha256(raw).hexdigest(), duplicates)


def _counts_metrics(tp: int, tn: int, fp: int, fn: int) -> dict[str, Any]:
    """Undefined rates stay null instead of posing as measured zero accuracy."""
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    balanced = (
        (recall + specificity) / 2 if recall is not None and specificity is not None else None
    )
    rates = {
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": balanced,
    }
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        **{key: round(value, 6) if value is not None else None for key, value in rates.items()},
    }


def _metrics(labels: Sequence[bool], predictions: Sequence[bool]) -> dict[str, Any]:
    """Measure pair classification only for explicitly scored pairs."""
    outcomes = list(zip(labels, predictions, strict=True))
    return _counts_metrics(
        sum(label and prediction for label, prediction in outcomes),
        sum(not label and not prediction for label, prediction in outcomes),
        sum(not label and prediction for label, prediction in outcomes),
        sum(label and not prediction for label, prediction in outcomes),
    )


def _recommend_threshold(scored: Sequence[tuple[bool, float]]) -> dict[str, Any]:
    """Suggest a training-only threshold; adequate counts still do not prove quality."""
    positive = sum(label for label, _ in scored)
    negative = len(scored) - positive
    base = {
        "positive_pairs": positive,
        "negative_pairs": negative,
        "minimum_positive_pairs": MIN_POSITIVE_PAIRS,
        "minimum_negative_pairs": MIN_NEGATIVE_PAIRS,
        "training_only": True,
    }
    if positive < MIN_POSITIVE_PAIRS or negative < MIN_NEGATIVE_PAIRS:
        return {**base, "status": "insufficient_scored_samples", "threshold": None}
    values = sorted({score for _, score in scored})
    thresholds = {0.0, 1.0}
    thresholds.update(
        (left + right) / 2
        for left, right in zip(values, values[1:], strict=False)
        if 0.0 <= (left + right) / 2 <= 1.0
    )
    ordered = sorted(scored, key=lambda item: item[1])
    tp, tn, fp, fn, position = positive, 0, negative, 0, 0
    choices: list[tuple[float, float, float, dict[str, Any]]] = []
    for threshold in sorted(thresholds):
        while position < len(ordered) and ordered[position][1] < threshold:
            if ordered[position][0]:
                tp -= 1
                fn += 1
            else:
                fp -= 1
                tn += 1
            position += 1
        metrics = _counts_metrics(tp, tn, fp, fn)
        choices.append((metrics["balanced_accuracy"], metrics["f1"] or 0.0, threshold, metrics))
    _, _, threshold, metrics = max(choices)
    return {
        **base,
        "status": "training_suggestion_not_production_calibration",
        "threshold": threshold,
        "in_sample_metrics": metrics,
    }


def _scored(
    pairs: Sequence[FeedbackPair], kind: str, face_min_reliability: float
) -> list[tuple[bool, float]]:
    if kind == "body":
        return [
            (pair.same_person, pair.body_score) for pair in pairs if pair.body_score is not None
        ]
    return [
        (pair.same_person, pair.face_score)
        for pair in pairs
        if pair.face_score is not None
        and pair.face_reliability is not None
        and pair.face_reliability >= face_min_reliability
    ]


def _partition_report(
    pairs: Sequence[FeedbackPair],
    body_threshold: float,
    face_threshold: float,
    face_min_reliability: float,
    attribute_hard_conflicts: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "labelled_pairs": len(pairs),
        "positive_pairs": sum(pair.same_person for pair in pairs),
    }
    result["negative_pairs"] = len(pairs) - result["positive_pairs"]
    for kind, threshold in (("body", body_threshold), ("face", face_threshold)):
        scored = _scored(pairs, kind, face_min_reliability)
        result[kind] = {
            "threshold": threshold,
            "scored_pairs": len(scored),
            "unscored_pairs": len(pairs) - len(scored),
            "coverage": round(len(scored) / len(pairs), 6) if pairs else None,
            "metrics": _metrics(
                [label for label, _ in scored], [score >= threshold for _, score in scored]
            ),
        }
    result["face"]["minimum_snapshot_reliability"] = face_min_reliability
    result["face"]["missing_score"] = sum(pair.face_score is None for pair in pairs)
    result["face"]["missing_reliability"] = sum(
        pair.face_score is not None and pair.face_reliability is None for pair in pairs
    )
    result["face"]["low_reliability"] = sum(
        pair.face_score is not None
        and pair.face_reliability is not None
        and pair.face_reliability < face_min_reliability
        for pair in pairs
    )
    tagged = [
        pair
        for pair in pairs
        if pair.comparable_count is not None
        and pair.comparable_count > 0
        and pair.conflict_count is not None
    ]
    result["attributes"] = {
        "diagnostic_only_not_runtime_fusion": True,
        "hard_conflicts": attribute_hard_conflicts,
        "comparable_pairs": len(tagged),
        "unscored_pairs": len(pairs) - len(tagged),
        "coverage": round(len(tagged) / len(pairs), 6) if pairs else None,
        "metrics": _metrics(
            [pair.same_person for pair in tagged],
            [
                pair.conflict_count < attribute_hard_conflicts
                for pair in tagged
                if pair.conflict_count is not None
            ],
        ),
    }
    return result


def evaluate(
    data: FeedbackData,
    *,
    body_threshold: float = 0.5,
    face_threshold: float = 0.45,
    face_min_reliability: float = 0.7,
    attribute_hard_conflicts: int = 2,
    group_column: str | None = None,
    holdout_groups: Sequence[str] = (),
) -> dict[str, Any]:
    """Separate fixed-threshold descriptions from training-only recommendations."""
    for value in (body_threshold, face_threshold, face_min_reliability):
        _require(math.isfinite(value) and 0 <= value <= 1, "invalid_threshold_parameter")
    _require(
        type(attribute_hard_conflicts) is int and 1 <= attribute_hard_conflicts <= 20,
        "invalid_attribute_threshold",
    )
    _require(bool(group_column) == bool(holdout_groups), "group_split_incomplete")
    selected = frozenset(holdout_groups)
    train = data.pairs
    holdout: tuple[FeedbackPair, ...] = ()
    if selected:
        _require(selected <= {pair.group for pair in data.pairs}, "unknown_holdout_group")
        train = tuple(pair for pair in data.pairs if pair.group not in selected)
        holdout = tuple(pair for pair in data.pairs if pair.group in selected)
        _require(bool(train) and bool(holdout), "empty_train_or_holdout")
        train_ids = {crop for pair in train for crop in (pair.query_id, pair.candidate_id)}
        test_ids = {crop for pair in holdout for crop in (pair.query_id, pair.candidate_id)}
        _require(train_ids.isdisjoint(test_ids), "crop_leakage_between_splits")
    parameters = (body_threshold, face_threshold, face_min_reliability, attribute_hard_conflicts)
    report: dict[str, Any] = {
        "schema_version": 2,
        "mode": "offline_human_feedback_snapshot",
        "dataset_sha256": data.sha256,
        "labelled_pairs": len(data.pairs),
        "deduplicated_identical_rows": data.deduplicated_rows,
        "truth_source": "explicit_same_person_operator_review_required",
        "parameters_loaded_from_production": False,
        "parameters": {
            "body_threshold": body_threshold,
            "face_threshold": face_threshold,
            "face_min_reliability": face_min_reliability,
            "attribute_hard_conflicts": attribute_hard_conflicts,
        },
        "identity_disjoint_test_proven": False,
        "split": "operator_group_crop_disjoint_holdout" if selected else "no_independent_holdout",
        "holdout_group_count": len(selected),
        "fixed_threshold_evaluation": _partition_report(data.pairs, *parameters),
        "training_threshold_suggestions": {
            kind: _recommend_threshold(_scored(train, kind, face_min_reliability))
            for kind in ("body", "face")
        },
        "limitations": [
            "CSV verdicts must be human reviewed; scores/person IDs never establish truth.",
            "Exported scores are historical snapshots, not new-model or end-to-end inference.",
            "Pair classification is not full-gallery Rank-1/mAP or unbiased search recall.",
            "No independent identity-disjoint accuracy is claimed, even with operator groups.",
            "Missing scores/faces/tags abstain; subgroup metrics do not represent all pairs.",
            "Face score diagnostics omit live identity/geometry gates and face priority.",
            "Repeated people/reversed pairs may be correlated; "
            "minimum pair counts are not identities.",
            "Tag conflict counts omit time windows/weights and do not reproduce live fusion.",
            "Threshold suggestions are training-only and never update production settings.",
        ],
    }
    if selected:
        report["training_fixed_threshold_evaluation"] = _partition_report(train, *parameters)
        report["holdout_fixed_threshold_evaluation"] = _partition_report(holdout, *parameters)
    return report


def write_report(path: Path, report: dict[str, Any]) -> None:
    """Create a new 0600 report exclusively; never overwrite or follow links."""
    _require(".." not in path.parts, "unsafe_output_path")
    for parent in (path.parent, *path.parent.parents):
        _require(not parent.is_symlink(), "output_parent_link")
    content = (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as handle:
        os.fchmod(handle.fileno(), 0o600)
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def main(argv: Sequence[str] | None = None) -> int:
    """Print aggregate evidence only, with safe bounded errors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", type=Path, help="exported human-feedback CSV; never auto labels")
    parser.add_argument("--output", type=Path, default=Path("reid-calibration-report.json"))
    parser.add_argument("--body-threshold", type=float, default=0.5)
    parser.add_argument("--face-threshold", type=float, default=0.45)
    parser.add_argument("--face-min-reliability", type=float, default=0.7)
    parser.add_argument("--attribute-hard-conflicts", type=int, default=2)
    parser.add_argument("--group-column", help="operator-defined CSV grouping column")
    parser.add_argument(
        "--holdout-group",
        action="append",
        default=[],
        help="group explicitly held out; repeat as needed",
    )
    args = parser.parse_args(argv)
    try:
        _require(
            args.group_column is None
            or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", args.group_column) is not None,
            "invalid_group_column",
        )
        data = read_feedback(args.labels, args.group_column)
        report = evaluate(
            data,
            body_threshold=args.body_threshold,
            face_threshold=args.face_threshold,
            face_min_reliability=args.face_min_reliability,
            attribute_hard_conflicts=args.attribute_hard_conflicts,
            group_column=args.group_column,
            holdout_groups=args.holdout_group,
        )
        write_report(args.output, report)
        print(
            json.dumps(
                {
                    "ok": True,
                    "mode": report["mode"],
                    "labelled_pairs": report["labelled_pairs"],
                    "parameters": report["parameters"],
                    "parameters_loaded_from_production": False,
                    "fixed_threshold_evaluation": report["fixed_threshold_evaluation"],
                    "training_threshold_suggestions": report["training_threshold_suggestions"],
                    "split": report["split"],
                    "identity_disjoint_test_proven": False,
                },
                allow_nan=False,
            )
        )
        return 0
    except (EvaluationError, OSError) as error:
        code = str(error) if isinstance(error, EvaluationError) else "input_or_output_unavailable"
        print(json.dumps({"ok": False, "code": code}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
