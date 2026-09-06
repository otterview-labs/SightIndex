import csv
import io
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from test_reid import load_app


@pytest.fixture
def feedback_app(monkeypatch, tmp_path):
    main = load_app(monkeypatch, tmp_path, "test-reid-feedback")

    from app.db.session import SessionLocal
    from app.models.media import Image, PersonCrop

    with TestClient(main.create_app()) as client:
        with SessionLocal() as db:
            image = Image(image_url="/data/frames/feedback.jpg", source_type="stream_frame")
            db.add(image)
            db.flush()
            query = PersonCrop(
                image_id=image.id,
                crop_url="/data/crops/query.jpg",
                bbox={"label": "person"},
            )
            candidate = PersonCrop(
                image_id=image.id,
                crop_url="/data/crops/candidate.jpg",
                bbox={"label": "person"},
            )
            db.add_all([query, candidate])
            db.commit()
            db.refresh(query)
            db.refresh(candidate)
            ids = {"query": query.id, "candidate": candidate.id}
        yield client, ids


def test_feedback_is_saved_listed_and_replaced(feedback_app):
    client, ids = feedback_app
    payload = {
        "query_crop_id": str(ids["query"]),
        "candidate_crop_id": str(ids["candidate"]),
        "same_person": True,
        "source": "camera_link",
        "body_score": 0.46,
        "face_similarity": 0.82,
        "face_reliability": 0.91,
        "face_match": True,
        "attribute_agreement": 0.75,
        "attribute_comparable_count": 4,
        "attribute_match_count": 3,
        "attribute_conflict_count": 1,
        "fusion_score": 0.53,
        "evidence_level": "reliable",
        "decision_reason": "可靠人脸吻合",
    }

    created = client.put("/api/reid/feedback", json=payload)
    assert created.status_code == 200
    assert created.json()["same_person"] is True
    assert created.json()["source"] == "camera_link"
    assert created.json()["body_score"] == 0.46
    assert created.json()["face_match"] is True

    payload.update({"same_person": False, "source": "search"})
    replaced = client.put("/api/reid/feedback", json=payload)
    assert replaced.status_code == 200
    assert replaced.json()["id"] == created.json()["id"]
    assert replaced.json()["same_person"] is False

    listed = client.get(
        "/api/reid/feedback",
        params={"query_crop_id": str(ids["query"])},
    )
    assert listed.status_code == 200
    assert listed.json() == [replaced.json()]

    from app.db.session import SessionLocal
    from app.models.reid import ReidMatchFeedback

    with SessionLocal() as db:
        assert db.scalar(select(func.count()).select_from(ReidMatchFeedback)) == 1


def test_feedback_rejects_self_pairs_unknown_crops_and_unknown_sources(feedback_app):
    client, ids = feedback_app
    base = {
        "query_crop_id": str(ids["query"]),
        "candidate_crop_id": str(ids["candidate"]),
        "same_person": True,
        "source": "search",
    }

    self_pair = client.put(
        "/api/reid/feedback",
        json={**base, "candidate_crop_id": str(ids["query"])},
    )
    assert self_pair.status_code == 422
    assert "must be different" in self_pair.json()["detail"]

    missing = client.put(
        "/api/reid/feedback",
        json={**base, "candidate_crop_id": str(uuid.uuid4())},
    )
    assert missing.status_code == 404
    assert "Candidate crop not found" in missing.json()["detail"]

    invalid_source = client.put(
        "/api/reid/feedback",
        json={**base, "source": "automatic"},
    )
    assert invalid_source.status_code == 422

    unknown_query = client.get(
        "/api/reid/feedback",
        params={"query_crop_id": str(uuid.uuid4())},
    )
    assert unknown_query.status_code == 404


def test_feedback_csv_matches_the_walkthrough_calibrator_format(feedback_app):
    client, ids = feedback_app
    client.put(
        "/api/reid/feedback",
        json={
            "query_crop_id": str(ids["query"]),
            "candidate_crop_id": str(ids["candidate"]),
            "same_person": True,
            "source": "search",
        },
    ).raise_for_status()

    response = client.get("/api/reid/feedback/export.csv")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["content-disposition"] == 'attachment; filename="reid-feedback.csv"'
    rows = list(csv.DictReader(io.StringIO(response.text)))
    assert rows == [
        {
            "query_crop_id": str(ids["query"]),
            "candidate_crop_id": str(ids["candidate"]),
            "same_person": "true",
            "source": "search",
            "body_score": "",
            "face_similarity": "",
            "face_reliability": "",
            "face_match": "",
            "attribute_agreement": "",
            "attribute_comparable_count": "0",
            "attribute_match_count": "0",
            "attribute_conflict_count": "0",
            "fusion_score": "",
            "evidence_level": "",
            "decision_reason": "",
            "created_at": rows[0]["created_at"],
            "updated_at": rows[0]["updated_at"],
        }
    ]
