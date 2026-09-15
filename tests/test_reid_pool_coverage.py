"""Diagnostics for the global ReID candidate pool."""

import uuid

from fastapi.testclient import TestClient
from test_reid import load_app


def test_indexed_camera_count_uses_current_reid_space(monkeypatch, tmp_path):
    """Coverage diagnostics count distinct current-marker cameras, not stale rows."""

    main = load_app(monkeypatch, tmp_path, "test-reid-camera-coverage")
    from app.config.settings import get_settings
    from app.db.session import SessionLocal
    from app.models.media import Image, PersonCrop
    from app.models.vectors import VLEmbedding
    from app.services.reid_index import REID_OBJECT_TYPE, ReidIndexService

    with TestClient(main.create_app()):
        with SessionLocal() as db:
            camera_a, camera_b, camera_c, camera_d = (uuid.uuid4() for _ in range(4))
            image = Image(
                image_url="/data/frame.jpg",
                source_type="stream_frame",
                camera_id=camera_d,
            )
            db.add(image)
            db.commit()
            db.refresh(image)
            crops = [
                PersonCrop(
                    image_id=image.id,
                    crop_url=f"/data/{index}.jpg",
                    bbox={"label": "person"},
                    camera_id=camera,
                )
                for index, camera in enumerate(
                    [camera_a, camera_a, camera_b, camera_c, camera_d, None]
                )
            ]
            db.add_all(crops)
            db.commit()
            for crop in crops:
                db.refresh(crop)
            service = ReidIndexService(db, get_settings())
            db.add_all(
                [
                    VLEmbedding(
                        object_type=REID_OBJECT_TYPE,
                        object_id=crop.id,
                        embedding_model=(service.fingerprint if index != 4 else "stale-space"),
                        embedding_dim=service.settings.reid_embedding_dim,
                    )
                    for index, crop in enumerate(crops)
                ]
            )
            db.commit()

            # The final crop inherits camera D from its image; it is still counted once.
            assert service.indexed_camera_count() == 4
            assert service.indexed_camera_count(exclude_camera_id=camera_a) == 3
