import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config.settings import Settings
from app.db.session import Base
from app.models import persons  # noqa: F401  person_crops.person_id needs this mapper
from app.models.media import Image, PersonCrop, VideoStream
from app.models.vectors import VectorIndexCapacityLock
from app.services.observation_index import ObservationIndexService


@pytest.fixture
def db(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'observations.db'}")
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine, autoflush=False)()
    # init_db() seeds this against the global engine; upsert_crop refuses to run without it.
    session.add(VectorIndexCapacityLock(target="observation_index", revision=0))
    session.commit()
    try:
        yield session
    finally:
        session.close()


def _crop(db, camera_id, location_id):
    image = Image(image_url="/data/frame.jpg", source_type="stream_frame")
    db.add(image)
    db.flush()
    crop = PersonCrop(
        image_id=image.id,
        crop_url="/data/crops/one.jpg",
        bbox={"x": 0, "y": 0, "w": 40, "h": 80},
        camera_id=camera_id,
        location_id=location_id,
    )
    db.add(crop)
    db.flush()
    return crop


def test_location_name_comes_from_the_stream(db, tmp_path):
    camera_id, location_id = uuid.uuid4(), uuid.uuid4()
    db.add(
        VideoStream(
            name="产品部门口",
            stream_url="rtsp://example/1",
            camera_id=camera_id,
            location_id=location_id,
            location_name="研发中心 3F 产品部",
        )
    )
    row = ObservationIndexService(db, Settings(data_dir=tmp_path)).upsert_crop(
        _crop(db, camera_id, location_id)
    )

    assert row.camera_name == "产品部门口"
    assert row.location_name == "研发中心 3F 产品部"


def test_location_name_falls_back_to_the_raw_id(db, tmp_path):
    """Without a named stream the UUID is all there is -- ugly, but it must not become NULL."""
    location_id = uuid.uuid4()
    row = ObservationIndexService(db, Settings(data_dir=tmp_path)).upsert_crop(
        _crop(db, uuid.uuid4(), location_id)
    )

    assert row.location_name == str(location_id)
