import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FaceAlgorithmInfo:
    provider: str
    model_pack: str
    detector: str
    detector_model_file: str
    recognizer: str
    recognizer_model_file: str
    embedding_dim: int
    model_root: str | None
    device: str
    ctx_id: int
    requested_providers: list[str]
    available_providers: list[str]


@dataclass(frozen=True)
class FaceAlgorithmCandidate:
    embedding: list[float]
    bbox: dict[str, float]
    quality_score: float
    model: str


class InsightFaceCudaRecognizer:
    """InsightFace face detector + embedding extractor with explicit CUDA provider selection."""

    def __init__(
        self,
        model_name: str = "buffalo_l",
        det_size: int = 640,
        device: str | None = None,
        root: Path | str | None = None,
        allow_download: bool = False,
    ) -> None:
        self.model_name = model_name
        self.det_size = det_size
        self.device = _normalize_device(device)
        self.root = Path(root).expanduser() if root else None
        self.allow_download = allow_download
        self.ctx_id = _ctx_id(self.device)
        self.providers = _providers(self.device)

    def model_dir(self) -> Path:
        root = self.root or Path.home() / ".insightface"
        return root / "models" / self.model_name

    def _require_model_present(self) -> None:
        """Refuses to pull 281MB inline in whoever happened to ask first.

        InsightFace downloads on first use, inside the calling thread, holding whatever that
        thread holds. Measured here at 10KB/s against GitHub -- an eight-hour request that kept
        a database session open the whole time. The pose weights are seeded deliberately for the
        same reason; this makes the face pack match.
        """

        if self.allow_download or self.model_dir().is_dir():
            return
        raise ValueError(
            f"InsightFace model pack '{self.model_name}' is not present at {self.model_dir()}. "
            "Seed it there, or set FACE_INSIGHTFACE_ALLOW_DOWNLOAD=true to fetch it inline."
        )

    def info(self) -> FaceAlgorithmInfo:
        detector = "RetinaFace-10GF"
        detector_model_file = "det_10g.onnx"
        recognizer = "ArcFace ResNet50@WebFace600K"
        recognizer_model_file = "w600k_r50.onnx"
        if self.model_name != "buffalo_l":
            detector = "InsightFace detector"
            detector_model_file = "model pack detector"
            recognizer = "InsightFace recognizer"
            recognizer_model_file = "model pack recognizer"
        return FaceAlgorithmInfo(
            provider="insightface",
            model_pack=self.model_name,
            detector=detector,
            detector_model_file=detector_model_file,
            recognizer=recognizer,
            recognizer_model_file=recognizer_model_file,
            embedding_dim=512,
            model_root=str(self.root) if self.root else None,
            device=self.device,
            ctx_id=self.ctx_id,
            requested_providers=list(self.providers),
            available_providers=_available_onnx_providers(),
        )

    def extract(self, image_path: Path) -> list[FaceAlgorithmCandidate]:
        try:
            import cv2  # type: ignore[import-not-found]
            from insightface.app import FaceAnalysis  # type: ignore[import-not-found]
        except Exception as exc:
            raise ValueError(f"InsightFace runtime is not installed: {exc}") from exc

        image = cv2.imread(str(image_path))
        if image is None:
            return []

        self._require_model_present()
        app = _insightface_app(
            self.device,
            self.model_name,
            self.det_size,
            str(self.root) if self.root else None,
            FaceAnalysis,
            tuple(self.providers),
        )
        faces = app.get(image)
        candidates: list[FaceAlgorithmCandidate] = []
        for face in faces:
            embedding = [float(value) for value in getattr(face, "embedding", [])]
            normalized = _normalize(embedding)
            if not normalized:
                continue
            bbox_values = [float(value) for value in getattr(face, "bbox", [0, 0, 0, 0])]
            x1, y1, x2, y2 = bbox_values[:4]
            candidates.append(
                FaceAlgorithmCandidate(
                    embedding=normalized,
                    bbox={
                        "x": max(0.0, x1),
                        "y": max(0.0, y1),
                        "width": max(1.0, x2 - x1),
                        "height": max(1.0, y2 - y1),
                    },
                    quality_score=float(getattr(face, "det_score", 0.0) or 0.0),
                    model=f"insightface-{self.model_name}",
                )
            )
        return candidates


def _normalize_device(device: str | None) -> str:
    if device is None or not str(device).strip():
        return "cuda:0"
    value = str(device).strip().lower()
    if value in {"gpu", "cuda"}:
        return "cuda:0"
    if value in {"cpu", "-1"}:
        return "cpu"
    return value


def _ctx_id(device: str) -> int:
    if device == "cpu":
        return -1
    if device.startswith("cuda:"):
        suffix = device.split(":", 1)[1]
        return int(suffix) if suffix.isdigit() else 0
    if device.isdigit():
        return int(device)
    return 0


def _providers(device: str) -> list[str]:
    if device == "cpu":
        return ["CPUExecutionProvider"]
    return ["CUDAExecutionProvider", "CPUExecutionProvider"]


def _available_onnx_providers() -> list[str]:
    try:
        import onnxruntime as ort  # type: ignore[import-not-found]
    except Exception:
        return []
    return [str(provider) for provider in ort.get_available_providers()]


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0:
        return []
    return [value / norm for value in vector]


@lru_cache(maxsize=4)
def _insightface_app(
    device: str,
    model_name: str,
    det_size: int,
    root: str | None,
    face_analysis_class: Any,
    providers: tuple[str, ...],
) -> Any:
    ctx_id = _ctx_id(device)
    kwargs: dict[str, object] = {"name": model_name}
    if root is not None:
        kwargs["root"] = root
    try:
        app = face_analysis_class(**kwargs, providers=list(providers))
    except TypeError:
        app = face_analysis_class(**kwargs)
    app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))
    return app
