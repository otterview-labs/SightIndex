"""Load the reviewed public sample snapshot; never accept visitor media or infer identities."""

import hashlib
import json
import shutil
from pathlib import Path

from app.config.settings import get_settings
from app.db.session import SessionLocal, init_db
from app.models.media import PersonCrop
from app.public_demo import validate_public_demo
from app.services.media import MediaService
from app.services.observation_index import ObservationIndexService
from app.services.storage import StorageService


def seed() -> None:
    settings = get_settings()
    if not settings.public_demo_read_only:
        raise RuntimeError("Seeding requires PUBLIC_DEMO_READ_ONLY=true")
    validate_public_demo(settings)
    database_path = Path(settings.database_url.removeprefix("sqlite:///"))
    if database_path.exists() and database_path.stat().st_size:
        raise RuntimeError("Demo database already exists; refusing to modify it")
    samples = Path(__file__).resolve().parents[1] / "samples"
    snapshot = samples / "demo"
    manifest = json.loads((snapshot / "manifest.json").read_text())
    sample = samples / "nasa-walk.jpg"
    if hashlib.sha256(sample.read_bytes()).hexdigest() != manifest["sample_sha256"]:
        raise RuntimeError("Public sample does not match the reviewed snapshot")
    for name, digest in manifest["files_sha256"].items():
        if Path(name).name != name:
            raise RuntimeError("Snapshot files must be direct children")
        if hashlib.sha256((snapshot / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError("Snapshot media checksum mismatch")
    storage = StorageService(settings)
    storage.ensure_dirs()
    shutil.copyfile(sample, settings.uploads_dir / "nasa-walk.jpg")
    shutil.copyfile(
        snapshot / manifest["thumbnail"], settings.thumbnails_dir / "demo-annotated.png",
    )
    init_db()
    with SessionLocal() as db:
        image = MediaService(db, settings).create_image_from_url(
            image_url="/data/uploads/nasa-walk.jpg", source_type="public_demo",
            original_file_name="nasa-walk.jpg", image_size=sample.stat().st_size,
        )
        image.thumbnail_url = "/data/thumbnails/demo-annotated.png"
        for item in manifest["crops"]:
            name = item["file"]
            shutil.copyfile(snapshot / name, settings.crops_dir / name)
            crop = PersonCrop(
                image_id=image.id, crop_url=f"/data/crops/{name}",
                bbox=item["bbox"], attributes=item["attributes"],
            )
            db.add(crop)
            db.flush()
            ObservationIndexService(db, settings).upsert_crop(crop)
        db.commit()
        print(f"Loaded {len(manifest['crops'])} reviewed, preprocessed public sample crops")


if __name__ == "__main__":
    seed()
