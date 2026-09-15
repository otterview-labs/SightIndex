import uuid
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from test_reid import load_app


@pytest.fixture
def identity_context(monkeypatch, tmp_path):
    main = load_app(
        monkeypatch,
        tmp_path,
        "identity-assignment",
        MILVUS_ENABLED="false",
        REID_ENABLED="false",
        VECTOR_INDEX_ON_INGEST="false",
        VLM_PROVIDER="none",
        STREAM_AUTOSTART_RUNNING="false",
    )
    from app.db.session import SessionLocal
    from app.models.media import Image, PersonCrop
    from app.models.persons import Person

    with TestClient(main.create_app()), SessionLocal() as db:
        image = Image(image_url="/data/frames/test.jpg")
        confirmed = Person(name="Confirmed")
        predicted = Person(name="Predicted")
        db.add_all([image, confirmed, predicted])
        db.flush()
        crop = PersonCrop(
            image_id=image.id,
            crop_url="/data/crops/test.jpg",
            bbox={"label": "person"},
        )
        db.add(crop)
        db.commit()
        yield db, main.get_settings(), image, crop, confirmed, predicted


def _face_service(monkeypatch, db, settings, person):
    from app.models.vectors import FaceEmbedding
    from app.services.faces import FaceCandidate, FaceMatch, FaceRecognitionService

    candidate = FaceCandidate(
        embedding=[1.0, 0.0],
        bbox={"x": 1.0, "y": 1.0, "width": 50.0, "height": 50.0},
        quality_score=0.9,
        model="test",
    )
    face = FaceEmbedding(person_id=person.id, face_model="test", embedding=candidate.embedding)
    db.add(face)
    db.commit()
    match = FaceMatch(person=person, face_embedding=face, similarity=0.95)
    service = FaceRecognitionService(db, settings)
    monkeypatch.setattr(service, "has_known_faces", lambda: True)
    monkeypatch.setattr(service, "_best_candidate", lambda *args, **kwargs: candidate)
    monkeypatch.setattr(service, "_search_matches", lambda *args, **kwargs: [match])
    return service


def test_recognition_preserves_manual_identity_and_separate_evidence(identity_context, monkeypatch):
    from sqlalchemy import select

    from app.models.media import PersonObservationIndex
    from app.services.persons import PersonService

    db, settings, image, crop, confirmed, predicted = identity_context
    PersonService(db, settings).label_crop(confirmed, crop)
    service = _face_service(monkeypatch, db, settings, predicted)

    event = service.recognize_crop(crop, image, require_ingest_enabled=False)

    assert crop.person_id == confirmed.id
    assert crop.person_id_source == "manual"
    assert event.person_id == predicted.id
    observation = db.scalar(
        select(PersonObservationIndex).where(PersonObservationIndex.crop_id == crop.id)
    )
    assert observation.person_id == confirmed.id
    assert observation.recognition_result_type is None
    assert observation.face_similarity is None
    assert observation.face_confidence is None
    assert PersonService(db, settings)._trajectory_seed_crops(
        predicted, [event], limit=10
    ) == []
    assert PersonService(db, settings)._is_allowed_vector_match(crop, confirmed, event)


def test_manual_unlabel_does_not_restore_old_face_identity(identity_context, monkeypatch):
    from sqlalchemy import select

    from app.models.media import PersonObservationIndex
    from app.services.persons import PersonService

    db, settings, image, crop, confirmed, predicted = identity_context
    service = _face_service(monkeypatch, db, settings, predicted)
    event = service.recognize_crop(crop, image, require_ingest_enabled=False)
    persons = PersonService(db, settings)
    persons.unlabel_crop(crop)
    persons.label_crop(confirmed, crop)
    persons.unlabel_crop(crop)

    service.recognize_crop(crop, image, existing_event=event, require_ingest_enabled=False)

    assert crop.person_id is None
    assert crop.person_id_source == "manual"
    observation = db.scalar(
        select(PersonObservationIndex).where(PersonObservationIndex.crop_id == crop.id)
    )
    assert observation.person_id is None
    assert observation.face_similarity is None
    assert observation.has_face_embedding is False
    assert persons._trajectory_seed_crops(predicted, [event], limit=10) == []


@pytest.mark.parametrize("source", ["manual", None])
def test_no_face_does_not_clear_protected_identity(identity_context, monkeypatch, source):
    from app.models.events import RecognitionEvent

    db, settings, image, crop, confirmed, predicted = identity_context
    crop.person_id = confirmed.id
    crop.person_id_source = source
    event = RecognitionEvent(
        image_id=image.id,
        crop_id=crop.id,
        person_id=confirmed.id,
        result_type="known",
        recognized_at=datetime.now(UTC),
    )
    db.add(event)
    db.commit()
    service = _face_service(monkeypatch, db, settings, predicted)
    monkeypatch.setattr(service, "_best_candidate", lambda *args, **kwargs: None)

    service.recognize_crop(crop, image, existing_event=event, require_ingest_enabled=False)

    assert crop.person_id == confirmed.id
    assert crop.person_id_source == source
    assert event.result_type == "no_face"


@pytest.mark.parametrize("source,protected", [(None, True), ("manual", True), ("face", False)])
def test_assignment_preserves_legacy_and_manual_but_updates_automatic(
    identity_context, source, protected
):
    from app.services.faces import FaceRecognitionService

    db, settings, _, crop, confirmed, predicted = identity_context
    crop.person_id = confirmed.id
    crop.person_id_source = source
    db.commit()

    FaceRecognitionService(db, settings)._assign_recognized_person(crop, predicted.id)
    db.commit()

    assert crop.person_id == (confirmed.id if protected else predicted.id)


def test_stale_recognizer_cannot_overwrite_a_committed_manual_label(identity_context):
    from app.db.session import SessionLocal
    from app.models.media import PersonCrop
    from app.services.faces import FaceRecognitionService
    from app.services.persons import PersonService

    db, settings, _, crop, confirmed, predicted = identity_context
    crop_id, confirmed_id, predicted_id = crop.id, confirmed.id, predicted.id
    with SessionLocal() as stale_db:
        stale_crop = stale_db.get(PersonCrop, crop_id)
        stale_db.expunge(stale_crop)
        stale_db.rollback()
        PersonService(db, settings).label_crop(confirmed, crop)
        stale_db.add(stale_crop)
        FaceRecognitionService(stale_db, settings)._assign_recognized_person(
            stale_crop, predicted_id
        )
        stale_db.commit()
        assert stale_crop.person_id == confirmed_id
        assert stale_crop.person_id_source == "manual"


def test_identity_source_migration_is_additive_and_idempotent(identity_context, monkeypatch):
    from sqlalchemy import create_engine, inspect, text

    from app.db import session

    legacy_id = str(uuid.uuid4())
    engine = create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE person_crops (id VARCHAR PRIMARY KEY, person_id VARCHAR, "
                "created_at TIMESTAMP, captured_at TIMESTAMP, attributes JSON)"
            )
        )
        connection.execute(
            text("INSERT INTO person_crops (id, person_id) VALUES ('crop', :person_id)"),
            {"person_id": legacy_id},
        )
    monkeypatch.setattr(session, "engine", engine)
    try:
        session._ensure_compatible_schema()
        session._ensure_compatible_schema()
        assert "person_id_source" in {
            column["name"] for column in inspect(engine).get_columns("person_crops")
        }
        with engine.connect() as connection:
            assert connection.execute(
                text("SELECT person_id, person_id_source FROM person_crops")
            ).one() == (legacy_id, None)
    finally:
        engine.dispose()


def test_counting_uses_manual_identity_instead_of_conflicting_face(identity_context):
    from sqlalchemy import select

    from app.models.events import CountingEvent, RecognitionEvent
    from app.services.persons import PersonService
    from app.services.video_processing import LineCrossing, VideoProcessingService

    db, settings, image, crop, confirmed, predicted = identity_context
    PersonService(db, settings).label_crop(confirmed, crop)
    db.add(RecognitionEvent(
        image_id=image.id,
        crop_id=crop.id,
        person_id=predicted.id,
        result_type="known",
        recognized_at=datetime.now(UTC),
    ))
    db.commit()
    service = VideoProcessingService(db, settings)

    service._create_line_crossing_events(
        [LineCrossing(detection_index=0, direction="a_to_b")],
        {0: crop},
        datetime.now(UTC),
        camera_id=None,
        location_id=None,
        image=image,
    )
    db.commit()

    event = db.scalar(select(CountingEvent).where(CountingEvent.crop_id == crop.id))
    assert event.person_id == confirmed.id
    assert event.recognition_event_id is None
