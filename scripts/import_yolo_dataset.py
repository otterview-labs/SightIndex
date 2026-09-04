from __future__ import annotations

import argparse
import shutil
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config.settings import get_settings  # noqa: E402
from app.db.session import SessionLocal, init_db  # noqa: E402
from app.models.media import Image, PersonCrop  # noqa: E402
from app.services.vector_index import VectorIndexingService  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_CLASS_NAMES = {0: "phone", 1: "smoking"}


@dataclass(frozen=True)
class ImportStats:
    seen: int
    images_created: int
    crops_created: int
    skipped_existing: int
    indexed: int
    errors: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import a YOLO image dataset into SightIndex.")
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Directory containing train/valid images and labels.",
    )
    parser.add_argument("--source-type", default="dataset_phone_smoking")
    parser.add_argument("--limit", type=int, default=0, help="Max images to import. 0 imports all.")
    parser.add_argument("--copy-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--index", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    init_db()
    stats = import_yolo_dataset(
        dataset_dir=args.dataset_dir,
        source_type=args.source_type,
        limit=args.limit,
        copy_images=args.copy_images,
        should_index=args.index,
    )
    print(
        "seen={seen} images_created={images_created} crops_created={crops_created} "
        "skipped_existing={skipped_existing} indexed={indexed} errors={errors} "
        "data_dir={data_dir}".format(
            **stats.__dict__,
            data_dir=settings.data_dir,
        )
    )


def import_yolo_dataset(
    dataset_dir: Path,
    source_type: str,
    limit: int,
    copy_images: bool,
    should_index: bool,
) -> ImportStats:
    settings = get_settings()
    dataset_dir = dataset_dir.resolve()
    image_paths = list(_iter_image_paths(dataset_dir))
    if limit > 0:
        image_paths = image_paths[:limit]

    import cv2  # type: ignore[import-not-found]

    images_created = 0
    crops_created = 0
    skipped_existing = 0
    indexed = 0
    errors = 0
    target_root = settings.data_dir / "datasets" / "phone_smoking" / "yolo"
    target_root.mkdir(parents=True, exist_ok=True)
    settings.crops_dir.mkdir(parents=True, exist_ok=True)

    with SessionLocal() as db:
        indexer = VectorIndexingService(db, settings) if should_index else None
        for image_path in image_paths:
            existing = db.scalar(
                select(Image.id)
                .where(Image.source_type == source_type)
                .where(Image.image_url.like(f"%/{image_path.name}"))
                .limit(1)
            )
            if existing is not None:
                skipped_existing += 1
                continue

            frame = cv2.imread(str(image_path))
            if frame is None:
                errors += 1
                continue
            height, width = frame.shape[:2]
            labels = list(_read_yolo_labels(image_path, dataset_dir, width, height))
            if not labels:
                continue

            stored_image_path = _stored_image_path(image_path, dataset_dir, target_root)
            stored_image_path.parent.mkdir(parents=True, exist_ok=True)
            if copy_images:
                shutil.copy2(image_path, stored_image_path)
            image_url = _data_url(stored_image_path, settings.data_dir)
            image = Image(image_url=image_url, source_type=source_type)
            db.add(image)
            db.flush()
            images_created += 1

            crops: list[PersonCrop] = []
            for label in labels:
                crop_url = _write_crop(cv2, frame, label, image_path.suffix, settings.crops_dir)
                crop = PersonCrop(
                    image_id=image.id,
                    crop_url=crop_url,
                    bbox=label,
                    captured_at=image.captured_at,
                )
                db.add(crop)
                crops.append(crop)
            db.commit()

            for crop in crops:
                db.refresh(crop)
                crops_created += 1
                if indexer is not None:
                    try:
                        indexer.index_crop(crop)
                        indexed += 1
                    except Exception:
                        errors += 1
            db.commit()

    return ImportStats(
        seen=len(image_paths),
        images_created=images_created,
        crops_created=crops_created,
        skipped_existing=skipped_existing,
        indexed=indexed,
        errors=errors,
    )


def _iter_image_paths(dataset_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for split in ("train", "valid", "val", "test"):
        image_dir = dataset_dir / split / "images"
        if image_dir.exists():
            paths.extend(
                path
                for path in sorted(image_dir.iterdir())
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
    if paths:
        return paths
    return [
        path
        for path in sorted(dataset_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def _read_yolo_labels(
    image_path: Path,
    dataset_dir: Path,
    image_width: int,
    image_height: int,
) -> list[dict[str, object]]:
    label_path = _label_path(image_path, dataset_dir)
    if not label_path.exists():
        return []
    labels: list[dict[str, object]] = []
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        class_id = int(float(parts[0]))
        center_x, center_y, width, height = (float(value) for value in parts[1:5])
        box_width = max(1, round(width * image_width))
        box_height = max(1, round(height * image_height))
        x = round((center_x * image_width) - box_width / 2)
        y = round((center_y * image_height) - box_height / 2)
        x = max(0, min(x, image_width - 1))
        y = max(0, min(y, image_height - 1))
        box_width = max(1, min(box_width, image_width - x))
        box_height = max(1, min(box_height, image_height - y))
        labels.append(
            {
                "x": x,
                "y": y,
                "width": box_width,
                "height": box_height,
                "label": DEFAULT_CLASS_NAMES.get(class_id, f"class_{class_id}"),
                "class_id": class_id,
                "confidence": 1.0,
                "dataset": "phone_smoking",
            }
        )
    return labels


def _label_path(image_path: Path, dataset_dir: Path) -> Path:
    relative = image_path.relative_to(dataset_dir)
    parts = list(relative.parts)
    try:
        images_index = parts.index("images")
        parts[images_index] = "labels"
    except ValueError:
        return image_path.with_suffix(".txt")
    return dataset_dir / Path(*parts).with_suffix(".txt")


def _stored_image_path(image_path: Path, dataset_dir: Path, target_root: Path) -> Path:
    relative = image_path.relative_to(dataset_dir)
    return target_root / relative


def _write_crop(
    cv2: object,
    frame: object,
    bbox: dict[str, object],
    suffix: str,
    crops_dir: Path,
) -> str:
    x = int(bbox["x"])
    y = int(bbox["y"])
    width = int(bbox["width"])
    height = int(bbox["height"])
    crop = frame[y : y + height, x : x + width]
    filename = f"{uuid.uuid4()}{suffix or '.jpg'}"
    target = crops_dir / filename
    cv2.imwrite(str(target), crop)
    return f"/data/crops/{filename}"


def _data_url(path: Path, data_dir: Path) -> str:
    return f"/data/{path.relative_to(data_dir).as_posix()}"


if __name__ == "__main__":
    main()
