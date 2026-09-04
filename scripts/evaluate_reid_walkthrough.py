import argparse
import csv
import json
import math
import sys
import uuid
from collections import defaultdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

def _metrics(labels: list[bool], predictions: list[bool]) -> dict[str, float | int]:
    tp = sum(label and prediction for label, prediction in zip(labels, predictions, strict=True))
    tn = sum(
        not label and not prediction
        for label, prediction in zip(labels, predictions, strict=True)
    )
    fp = sum(
        not label and prediction
        for label, prediction in zip(labels, predictions, strict=True)
    )
    fn = sum(
        label and not prediction
        for label, prediction in zip(labels, predictions, strict=True)
    )
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "specificity": round(specificity, 4),
        "f1": round(f1, 4),
        "balanced_accuracy": round((recall + specificity) / 2, 4),
    }


def _recommend_threshold(
    scored: list[tuple[bool, float]],
) -> dict[str, object] | None:
    if not scored or not any(label for label, _ in scored) or all(label for label, _ in scored):
        return None
    values = sorted({score for _, score in scored})
    thresholds = [values[0] - 1e-6, values[-1] + 1e-6]
    thresholds.extend(
        (left + right) / 2 for left, right in zip(values, values[1:], strict=False)
    )
    labels = [label for label, _ in scored]
    choices = []
    for threshold in thresholds:
        metrics = _metrics(labels, [score >= threshold for _, score in scored])
        choices.append((metrics["balanced_accuracy"], metrics["f1"], threshold, metrics))
    _balanced, _f1, threshold, metrics = max(choices)
    return {"threshold": round(threshold, 4), "metrics": metrics, "samples": len(scored)}


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return math.nan
    return sum(a * b for a, b in zip(left, right, strict=True))


def main() -> None:
    from sqlalchemy import select

    from app.config.settings import get_settings
    from app.db.session import SessionLocal
    from app.models.media import PersonCrop
    from app.services.faces import FaceRecognitionService
    from app.services.reid_attributes import compare_reid_attributes
    from app.services.reid_index import REID_OBJECT_TYPE, ReidIndexService

    parser = argparse.ArgumentParser(
        description="Calibrate ReID body, face and structured-tag thresholds from a walk-through."
    )
    parser.add_argument("labels", type=Path, help="CSV ground truth file")
    parser.add_argument("--output", type=Path, default=Path("reid-calibration-report.json"))
    args = parser.parse_args()
    rows = list(csv.DictReader(args.labels.open(encoding="utf-8")))
    required = {"query_crop_id", "candidate_crop_id", "same_person"}
    if not rows or not required.issubset(rows[0]):
        parser.error(f"CSV must contain {', '.join(sorted(required))}")

    settings = get_settings()
    pair_ids = {
        uuid.UUID(row[field])
        for row in rows
        for field in ("query_crop_id", "candidate_crop_id")
    }
    with SessionLocal() as db:
        crops = {
            crop.id: crop
            for crop in db.scalars(select(PersonCrop).where(PersonCrop.id.in_(pair_ids)))
        }
        missing = sorted(str(crop_id) for crop_id in pair_ids if crop_id not in crops)
        if missing:
            raise SystemExit(f"Unknown crop ids: {', '.join(missing)}")
        reid = ReidIndexService(db, settings)
        vectors = reid.index.fetch_vectors(REID_OBJECT_TYPE, list(pair_ids))
        face = FaceRecognitionService(db, settings)
        face_scores: dict[tuple[uuid.UUID, uuid.UUID], float] = {}
        by_query: dict[uuid.UUID, list[uuid.UUID]] = defaultdict(list)
        for row in rows:
            by_query[uuid.UUID(row["query_crop_id"])].append(
                uuid.UUID(row["candidate_crop_id"])
            )
        for query_id, candidate_ids in by_query.items():
            compared = face.compare_person_crops(
                crops[query_id],
                [crops[candidate_id] for candidate_id in candidate_ids],
                min_quality=settings.reid_face_min_quality,
            )
            for candidate_id, evidence in compared.items():
                face_scores[(query_id, candidate_id)] = evidence.similarity

        evaluated = []
        body_scored: list[tuple[bool, float]] = []
        face_scored: list[tuple[bool, float]] = []
        for row in rows:
            query_id = uuid.UUID(row["query_crop_id"])
            candidate_id = uuid.UUID(row["candidate_crop_id"])
            same_person = row["same_person"].strip().lower() in {"1", "true", "yes", "y"}
            body_score = _cosine(vectors.get(query_id, []), vectors.get(candidate_id, []))
            compatibility = compare_reid_attributes(
                crops[query_id].attributes,
                crops[candidate_id].attributes,
                min_confidence=settings.reid_attribute_min_confidence,
            )
            face_score = face_scores.get((query_id, candidate_id))
            if not math.isnan(body_score):
                body_scored.append((same_person, body_score))
            if face_score is not None:
                face_scored.append((same_person, face_score))
            evaluated.append(
                {
                    **row,
                    "same_person": same_person,
                    "body_similarity": None if math.isnan(body_score) else round(body_score, 4),
                    "face_similarity": None if face_score is None else round(face_score, 4),
                    "attribute_agreement": compatibility.agreement,
                    "attribute_conflicts": list(compatibility.conflicted),
                }
            )

    tag_choices = []
    labels = [bool(row["same_person"]) for row in evaluated]
    for hard_conflicts in range(1, 4):
        predictions = [
            len(row["attribute_conflicts"]) < hard_conflicts for row in evaluated
        ]
        tag_choices.append(
            {
                "hard_conflicts": hard_conflicts,
                "metrics": _metrics(labels, predictions),
            }
        )
    report = {
        "samples": len(evaluated),
        "positive_pairs": sum(labels),
        "negative_pairs": len(labels) - sum(labels),
        "body": _recommend_threshold(body_scored),
        "face": _recommend_threshold(face_scored),
        "attributes": tag_choices,
        "rows": evaluated,
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
