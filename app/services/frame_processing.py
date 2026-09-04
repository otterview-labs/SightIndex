import json
import logging
import shutil
import uuid
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib import error, request

from sqlalchemy.orm import Session

from app.config.settings import Settings
from app.models.media import Image, PersonCrop

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    bbox: dict[str, Any]
    confidence: float
    label: str = "person"


class PersonDetector:
    def detect(self, image_path: Path) -> list[Detection]:
        raise NotImplementedError


class WholeFramePersonDetector(PersonDetector):
    """Temporary detector that keeps the ingestion pipeline complete before YOLO is connected."""

    def detect(self, image_path: Path) -> list[Detection]:
        width, height = self._read_size(image_path)
        return [
            Detection(
                bbox={"x": 0, "y": 0, "width": width, "height": height},
                # This detector asserts the frame is a person rather than estimating it, so 0.0
                # said the opposite of what it means. Harmless while the confidence floor was 0;
                # the moment one exists, every crop it produces is thrown away.
                confidence=1.0,
                label="person",
            )
        ]

    def _read_size(self, image_path: Path) -> tuple[int, int]:
        try:
            import cv2  # type: ignore[import-not-found]
        except Exception:
            return 0, 0
        image = cv2.imread(str(image_path))
        if image is None:
            return 0, 0
        height, width = image.shape[:2]
        return int(width), int(height)


class OpenCVHogPersonDetector(PersonDetector):
    def __init__(self, hit_threshold: float = 0.0) -> None:
        try:
            import cv2  # type: ignore[import-not-found]
        except Exception as exc:
            raise RuntimeError(f"OpenCV is not installed: {exc}") from exc
        self.cv2 = cv2
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.hit_threshold = hit_threshold

    def detect(self, image_path: Path) -> list[Detection]:
        image = self.cv2.imread(str(image_path))
        if image is None:
            return []
        resized, scale = self._resize_for_detection(image)
        boxes, weights = self.hog.detectMultiScale(
            resized,
            hitThreshold=self.hit_threshold,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05,
        )
        if len(boxes) == 0:
            return []

        indices = self.cv2.dnn.NMSBoxes(
            bboxes=[tuple(map(int, box)) for box in boxes],
            scores=[float(weight) for weight in weights],
            score_threshold=max(self.hit_threshold, 0.0),
            nms_threshold=0.35,
        )
        flat_indices = self._flatten_indices(indices)
        detections: list[Detection] = []
        for index in flat_indices:
            x, y, width, height = [int(value / scale) for value in boxes[index]]
            detections.append(
                Detection(
                    bbox={"x": x, "y": y, "width": width, "height": height},
                    confidence=float(weights[index]),
                )
            )
        return detections

    def _resize_for_detection(self, image: Any) -> tuple[Any, float]:
        height, width = image.shape[:2]
        max_width = 960
        if width <= max_width:
            return image, 1.0
        scale = max_width / width
        target = (max_width, int(height * scale))
        return self.cv2.resize(image, target), scale

    def _flatten_indices(self, indices: Any) -> list[int]:
        if indices is None or len(indices) == 0:
            return []
        return [int(index[0] if hasattr(index, "__len__") else index) for index in indices]


class YoloPersonDetector(PersonDetector):
    def __init__(
        self,
        model_name: str,
        confidence: float,
        image_size: int,
        device: str | None = None,
    ) -> None:
        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise RuntimeError(f"Ultralytics YOLO is not available: {exc}") from exc
        self.model = YOLO(model_name)
        self.confidence = confidence
        self.image_size = image_size
        self.device = device or None

    def detect(self, image_path: Path) -> list[Detection]:
        if not self._is_readable_image(image_path):
            return []
        results = self.model.predict(
            source=str(image_path),
            classes=[0],
            conf=self.confidence,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        detections: list[Detection] = []
        for box in boxes:
            x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
            confidence = float(box.conf[0].item()) if box.conf is not None else 0.0
            detections.append(
                Detection(
                    bbox={
                        "x": x1,
                        "y": y1,
                        "width": max(1, x2 - x1),
                        "height": max(1, y2 - y1),
                    },
                    confidence=confidence,
                    label="person",
                )
            )
        return detections

    def _is_readable_image(self, image_path: Path) -> bool:
        try:
            import cv2  # type: ignore[import-not-found]
        except Exception:
            return True
        return cv2.imread(str(image_path)) is not None


class YoloServicePersonDetector(PersonDetector):
    def __init__(
        self,
        base_url: str,
        confidence: float,
        iou: float,
        image_size: int,
        max_det: int,
        timeout_seconds: int,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.confidence = confidence
        self.iou = iou
        self.image_size = image_size
        self.max_det = max_det
        self.timeout_seconds = timeout_seconds

    def detect(self, image_path: Path) -> list[Detection]:
        payload = json.dumps(
            {
                "task": "det",
                "image_path": str(image_path),
                "conf": self.confidence,
                "iou": self.iou,
                "imgsz": self.image_size,
                "max_det": self.max_det,
            }
        ).encode("utf-8")
        req = request.Request(
            self.base_url + "/predict",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (OSError, error.URLError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"YOLO service request failed: {exc}") from exc

        image_size = data.get("image_size")
        width = int(data.get("width") or 0)
        height = int(data.get("height") or 0)
        if isinstance(image_size, dict):
            width = width or int(image_size.get("width") or 0)
            height = height or int(image_size.get("height") or 0)
        if width <= 0 or height <= 0:
            width, height = self._read_size(image_path)

        raw_boxes = data.get("boxes") or data.get("detections") or []
        detections: list[Detection] = []
        if not isinstance(raw_boxes, list):
            return detections
        for item in raw_boxes:
            if not isinstance(item, dict):
                continue
            if not self._is_person_item(item):
                continue
            bbox = self._parse_bbox(item.get("bbox"), width, height)
            if bbox is None:
                continue
            confidence = self._parse_confidence(item)
            detections.append(Detection(bbox=bbox, confidence=confidence, label="person"))
        return detections

    def _is_person_item(self, item: dict[str, object]) -> bool:
        label = str(item.get("label") or item.get("class_name") or item.get("name") or "")
        if label:
            return label.lower() in {"person", "0"}

        class_value = item.get("class_id") or item.get("class") or item.get("cls")
        if class_value is None:
            return True
        try:
            return int(float(str(class_value))) == 0
        except (TypeError, ValueError):
            return False

    def _parse_confidence(self, item: dict[str, object]) -> float:
        try:
            return float(item.get("score") or item.get("confidence") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _parse_bbox(self, value: object, width: int, height: int) -> dict[str, int] | None:
        if width <= 0 or height <= 0:
            return None

        if isinstance(value, list | tuple):
            if len(value) != 4:
                return None
            try:
                x1, y1, x2, y2 = [float(item) for item in value]
            except (TypeError, ValueError):
                return None
        elif isinstance(value, dict):
            try:
                x1 = float(value.get("x1", value.get("x", 0)))
                y1 = float(value.get("y1", value.get("y", 0)))
                if "x2" in value and "y2" in value:
                    x2 = float(value["x2"])
                    y2 = float(value["y2"])
                else:
                    x2 = x1 + float(value.get("width", value.get("w", 0)))
                    y2 = y1 + float(value.get("height", value.get("h", 0)))
            except (TypeError, ValueError):
                return None
        else:
            return None

        if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
            x1 *= width
            x2 *= width
            y1 *= height
            y2 *= height
        left = max(0, min(int(round(x1)), width))
        top = max(0, min(int(round(y1)), height))
        right = max(left + 1, min(int(round(x2)), width))
        bottom = max(top + 1, min(int(round(y2)), height))
        return {
            "x": left,
            "y": top,
            "width": max(1, right - left),
            "height": max(1, bottom - top),
        }

    def _read_size(self, image_path: Path) -> tuple[int, int]:
        try:
            import cv2  # type: ignore[import-not-found]
        except Exception:
            return 0, 0
        image = cv2.imread(str(image_path))
        if image is None:
            return 0, 0
        height, width = image.shape[:2]
        return int(width), int(height)


@lru_cache(maxsize=8)
def _cached_yolo_detector(
    model_name: str,
    confidence: float,
    image_size: int,
    device: str | None,
) -> YoloPersonDetector:
    return YoloPersonDetector(
        model_name=model_name,
        confidence=confidence,
        image_size=image_size,
        device=device,
    )


def create_person_detector(settings: Settings) -> PersonDetector:
    detector_name = settings.person_detector.lower()
    if detector_name == "whole_frame":
        return WholeFramePersonDetector()
    if detector_name == "hog":
        return OpenCVHogPersonDetector(hit_threshold=settings.hog_hit_threshold)
    if detector_name == "yolo":
        return _cached_yolo_detector(
            settings.yolo_model,
            settings.yolo_confidence,
            settings.yolo_image_size,
            settings.yolo_device,
        )
    if detector_name == "yolo_service":
        return YoloServicePersonDetector(
            base_url=settings.yolo_service_url,
            confidence=settings.yolo_service_confidence,
            iou=settings.yolo_service_iou,
            image_size=settings.yolo_service_image_size,
            max_det=settings.yolo_service_max_det,
            timeout_seconds=settings.yolo_service_timeout_seconds,
        )
    raise ValueError(f"Unsupported person detector: {settings.person_detector}")


class FrameProcessingService:
    def __init__(
        self,
        db: Session,
        settings: Settings,
        detector: PersonDetector | None = None,
    ) -> None:
        self.db = db
        self.settings = settings
        self.detector = detector or create_person_detector(settings)

    def detect_image_path(self, image_path: Path) -> list[Detection]:
        return self.detector.detect(image_path)

    def quality_filter_detections(self, detections: list[Detection]) -> list[Detection]:
        return [detection for detection in detections if self._is_quality_detection(detection)]

    def process_image(
        self,
        image: Image,
        detections: list[Detection] | None = None,
    ) -> list[PersonCrop]:
        image_path = self._resolve_data_url(image.image_url)
        if image_path is None or not image_path.exists():
            return []
        original_thumbnail_url = image.thumbnail_url
        annotated_url: str | None = None

        raw_detections = (
            detections if detections is not None else self.detector.detect(image_path)
        )
        image_detections = self.quality_filter_detections(raw_detections)
        if image_detections:
            annotated_url = self._create_annotated_frame_file(image_path, image_detections)
            if annotated_url:
                image.thumbnail_url = annotated_url
                self.db.add(image)

        crops: list[PersonCrop] = []
        # Stature needs the frame's edges to tell a whole person from a clipped one, and reading
        # them back off disk per crop is a file open the capture loop does not need.
        frame_width, frame_height = self._read_image_size(image.image_url)
        for detection in image_detections:
            crop_url = self._create_crop_file(image_path, detection)
            crop_width, crop_height = self._read_image_size(crop_url)
            crop = PersonCrop(
                image_id=image.id,
                crop_url=crop_url,
                bbox={
                    **detection.bbox,
                    "confidence": detection.confidence,
                    "label": detection.label,
                    "crop_width": crop_width,
                    "crop_height": crop_height,
                    "frame_width": frame_width,
                    "frame_height": frame_height,
                    "quality_pass": True,
                },
                camera_id=image.camera_id,
                location_id=image.location_id,
                captured_at=image.captured_at,
            )
            self.db.add(crop)
            crops.append(crop)
        try:
            enqueued = self._enqueue_index_jobs(image, crops)
            self.db.commit()
        except Exception:
            self.db.rollback()
            # The database transaction is rolled back by the caller, but crop/annotation files
            # are external side effects. Remove only files created by this attempt so queue
            # pressure cannot accumulate orphaned media on a long-running stream.
            for crop in crops:
                self._remove_data_file(crop.crop_url)
            if annotated_url and annotated_url != original_thumbnail_url:
                self._remove_data_file(annotated_url)
            image.thumbnail_url = original_thumbnail_url
            raise
        if enqueued:
            from app.services.vector_index_queue import vector_index_queue

            vector_index_queue.wake(self.settings)
        for crop in crops:
            self.db.refresh(crop)
        if crops:
            self._try_recognize_faces(image, crops)
        if self.settings.appearance_tone_on_ingest and crops:
            self._try_read_clothing_tone(crops)
        if self.settings.vlm_structured_on_ingest and crops:
            self._try_analyze_crop_attributes(crops)
        if self.settings.vector_index_on_ingest and crops:
            self._try_index_crops(crops)
        return crops

    def _remove_data_file(self, url: str | None) -> None:
        if not url:
            return
        path = self._resolve_data_url(url)
        if path is None:
            return
        try:
            path.unlink(missing_ok=True)
        except OSError:
            return

    def _resolve_data_url(self, url: str) -> Path | None:
        prefix = "/data/"
        if not url.startswith(prefix):
            return None
        relative_path = url.removeprefix(prefix)
        return self.settings.data_dir / relative_path

    def _create_crop_file(self, image_path: Path, detection: Detection) -> str:
        self.settings.crops_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{uuid.uuid4()}{image_path.suffix or '.jpg'}"
        target = self.settings.crops_dir / filename

        if not self._try_crop_with_cv2(image_path, target, detection):
            shutil.copyfile(image_path, target)
        return f"/data/crops/{filename}"

    def _create_annotated_frame_file(
        self,
        image_path: Path,
        detections: list[Detection],
    ) -> str | None:
        try:
            import cv2  # type: ignore[import-not-found]
        except Exception:
            return None

        image = cv2.imread(str(image_path))
        if image is None:
            return None

        self.settings.thumbnails_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{uuid.uuid4()}{image_path.suffix or '.jpg'}"
        target = self.settings.thumbnails_dir / filename
        for detection in detections:
            self._draw_detection_box(cv2, image, detection)
        if not self._write_jpeg(cv2, target, image, self.settings.thumbnail_jpeg_quality):
            return None
        return f"/data/thumbnails/{filename}"

    def _draw_detection_box(self, cv2: Any, image: Any, detection: Detection) -> None:
        height, width = image.shape[:2]
        x1, y1, x2, y2 = self._bounded_bbox(image, detection)
        thickness = max(2, round(min(width, height) / 360))
        font_scale = max(0.45, min(0.9, width / 1280))
        color = (92, 215, 189)
        label_color = (8, 31, 25)
        label = f"{detection.label} {detection.confidence:.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness,
        )
        label_height = text_height + baseline + 8
        label_y1 = max(0, y1 - label_height)
        label_y2 = min(height - 1, label_y1 + label_height)
        label_x2 = min(width - 1, x1 + text_width + 10)
        cv2.rectangle(image, (x1, label_y1), (label_x2, label_y2), color, -1)
        cv2.putText(
            image,
            label,
            (x1 + 5, label_y2 - baseline - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            label_color,
            max(1, thickness - 1),
            cv2.LINE_AA,
        )

    def _try_crop_with_cv2(self, image_path: Path, target: Path, detection: Detection) -> bool:
        try:
            import cv2  # type: ignore[import-not-found]
        except Exception:
            return False

        image = cv2.imread(str(image_path))
        if image is None:
            return False

        x1, y1, x2, y2 = self._bounded_bbox(image, detection, padded=True)
        crop = image[y1:y2, x1:x2]
        crop = self._enhance_crop(cv2, crop)
        return self._write_jpeg(cv2, target, crop, self.settings.person_crop_jpeg_quality)

    def _is_quality_detection(self, detection: Detection) -> bool:
        bbox = detection.bbox
        width = int(bbox.get("width", 0) or 0)
        height = int(bbox.get("height", 0) or 0)
        if detection.confidence < self.settings.person_crop_min_confidence:
            return False
        if width < self.settings.person_crop_min_bbox_width:
            return False
        if height < self.settings.person_crop_min_bbox_height:
            return False
        return True

    def _enhance_crop(self, cv2: Any, crop: Any) -> Any:
        height, width = crop.shape[:2]
        if width <= 0 or height <= 0:
            return crop

        scale = 1.0
        min_width = self.settings.person_crop_upscale_min_width
        min_height = self.settings.person_crop_upscale_min_height
        if min_width and width < min_width:
            scale = max(scale, self.settings.person_crop_upscale_min_width / width)
        if min_height and height < min_height:
            scale = max(scale, self.settings.person_crop_upscale_min_height / height)
        scale = min(scale, self.settings.person_crop_upscale_max_factor)
        if scale > 1.01:
            crop = cv2.resize(
                crop,
                (max(1, round(width * scale)), max(1, round(height * scale))),
                # Lanczos keeps diagonal clothing and face edges better than cubic when a small
                # doorway subject has to be enlarged for inspection.
                interpolation=cv2.INTER_LANCZOS4,
            )

        amount = self.settings.person_crop_sharpen_amount
        if amount > 0:
            blurred = cv2.GaussianBlur(crop, (0, 0), 1.0)
            sharpened = cv2.addWeighted(crop, 1.0 + amount, blurred, -amount, 0)
            threshold = self.settings.person_crop_sharpen_threshold
            if threshold > 0:
                # Blend the sharpened image only around meaningful edges. Camera noise and JPEG
                # blocks live below the threshold and therefore do not get amplified.
                detail = cv2.absdiff(crop, blurred)
                detail_gray = cv2.cvtColor(detail, cv2.COLOR_BGR2GRAY)
                _, edge_mask = cv2.threshold(
                    detail_gray,
                    threshold,
                    255,
                    cv2.THRESH_BINARY,
                )
                edge_mask = cv2.GaussianBlur(edge_mask, (0, 0), 0.8)
                mask = cv2.cvtColor(edge_mask, cv2.COLOR_GRAY2BGR).astype("float32") / 255.0
                crop = cv2.convertScaleAbs(
                    crop.astype("float32") * (1.0 - mask)
                    + sharpened.astype("float32") * mask
                )
            else:
                crop = sharpened
        return crop

    def _write_jpeg(self, cv2: Any, target: Path, image: Any, quality: int) -> bool:
        if target.suffix.lower() in {".jpg", ".jpeg"}:
            params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
            return bool(cv2.imwrite(str(target), image, params))
        return bool(cv2.imwrite(str(target), image))

    def _read_image_size(self, data_url: str) -> tuple[int | None, int | None]:
        path = self._resolve_data_url(data_url)
        if path is None:
            return None, None
        try:
            import cv2  # type: ignore[import-not-found]
        except Exception:
            return None, None
        image = cv2.imread(str(path))
        if image is None:
            return None, None
        height, width = image.shape[:2]
        return int(width), int(height)

    def _bounded_bbox(
        self,
        image: Any,
        detection: Detection,
        *,
        padded: bool = False,
    ) -> tuple[int, int, int, int]:
        height, width = image.shape[:2]
        x = int(detection.bbox.get("x", 0))
        y = int(detection.bbox.get("y", 0))
        crop_width = int(detection.bbox.get("width", width))
        crop_height = int(detection.bbox.get("height", height))
        x1 = max(0, min(x, width))
        y1 = max(0, min(y, height))
        x2 = max(x1 + 1, min(x1 + crop_width, width))
        y2 = max(y1 + 1, min(y1 + crop_height, height))
        if padded:
            box_width = x2 - x1
            box_height = y2 - y1
            pad_x = round(box_width * self.settings.person_crop_padding_x_ratio)
            pad_top = round(box_height * self.settings.person_crop_padding_top_ratio)
            pad_bottom = round(box_height * self.settings.person_crop_padding_bottom_ratio)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_top)
            x2 = min(width, x2 + pad_x)
            y2 = min(height, y2 + pad_bottom)
        return x1, y1, x2, y2

    def _enqueue_index_jobs(self, image: Image, crops: list[PersonCrop]) -> bool:
        """Writes job rows into the crop's own transaction, so both commit or neither does.

        The worker is woken by the caller only after that commit. Capacity is checked for the
        complete crop/target set before any outbox row is staged. A full queue therefore raises
        and rolls the caller transaction back instead of committing a crop with half its required
        indexing jobs. Background inference itself remains queue-only.
        """

        from app.services.vector_index_queue import (
            REID_OBJECT_TYPE,
            vector_index_queue,
        )

        # SQLAlchemy's Python UUID defaults are assigned at flush time. Build requests only
        # after every new crop has its durable object id, otherwise the outbox can receive NULL.
        self.db.flush()
        crop_targets = [
            target
            for target in (REID_OBJECT_TYPE, "person_crop")
            if vector_index_queue.target_enabled(target, self.settings)
        ]
        requests: list[tuple[str, uuid.UUID]] = []
        if vector_index_queue.target_enabled("image", self.settings):
            requests.append(("image", image.id))
        requests.extend(
            (target, crop.id) for crop in crops for target in crop_targets
        )
        if not requests:
            return False
        return vector_index_queue.enqueue_many_in_session(
            self.db,
            requests,
            self.settings,
        )

    def _try_index_crops(self, crops: list[PersonCrop]) -> None:
        if self.settings.vector_index_on_ingest_background:
            return  # already enqueued in-session, atomically with the crops
        try:
            from app.services.vector_index import VectorIndexingService

            indexer = VectorIndexingService(self.db, self.settings)
            for crop in crops:
                indexer.write_crop_vector(crop, flush=False)
            indexer.index.flush("person_crop")
            for crop in crops:
                indexer.record_crop_index(crop)
            self.db.commit()
        except Exception:
            self.db.rollback()
            return

    def _try_read_clothing_tone(self, crops: list[PersonCrop]) -> None:
        """Fills clothing colour from the pixels, for deployments with no VLM.

        Never overwrites a VLM reading: that one knows about gender, bags and behaviour, and this
        one only knows how bright a band of pixels is. Refreshing its own earlier output is fine.
        """

        try:
            from app.services.appearance_attributes import AppearanceAttributeService
            from app.services.observation_index import ObservationIndexService
            from app.services.stature import StatureService

            stature_service = StatureService(self.db, self.settings)
            service = AppearanceAttributeService(
                saturation_floor=self.settings.appearance_tone_saturation_floor,
                hue_value_floor=self.settings.appearance_tone_hue_value_floor,
                dark_ratio=self.settings.appearance_tone_dark_ratio,
            )
            changed = False
            for crop in crops:
                existing = crop.attributes or {}
                if existing and existing.get("source") != "cv_tone":
                    continue
                path = self._resolve_crop_path(crop)
                if path is None:
                    continue
                attributes = service.describe(path)
                if attributes is None:
                    continue
                stature = stature_service.describe(crop.bbox, crop.camera_id)
                if stature:
                    attributes["stature"] = stature
                crop.attributes = attributes
                self.db.add(crop)
                changed = True
            if changed:
                self.db.commit()
                for crop in crops:
                    ObservationIndexService(self.db, self.settings).upsert_crop(crop)
                self.db.commit()
        except Exception:
            self.db.rollback()
            return

    def _resolve_crop_path(self, crop: PersonCrop) -> Path | None:
        prefix = "/data/"
        if not crop.crop_url or not crop.crop_url.startswith(prefix):
            return None
        path = self.settings.data_dir / Path(crop.crop_url.removeprefix(prefix))
        return path if path.exists() else None

    def _try_analyze_crop_attributes(self, crops: list[PersonCrop]) -> None:
        try:
            from app.services.structured_attributes import StructuredAttributeService

            analyzer = StructuredAttributeService(self.db, self.settings)
            for crop in crops:
                analyzer.analyze_person_crop(crop, persist=True)
        except Exception:
            return

    def _try_recognize_faces(self, image: Image, crops: list[PersonCrop]) -> None:
        try:
            from app.services.faces import FaceRecognitionService

            service = FaceRecognitionService(self.db, self.settings)
            if not service.has_known_faces():
                return
            for crop in crops:
                service.recognize_crop(crop, image)
        except Exception:
            return
