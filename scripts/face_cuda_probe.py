from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def main() -> None:
    from app.config.settings import get_settings
    from app.face_algorithms import InsightFaceCudaRecognizer

    parser = argparse.ArgumentParser(description="Probe InsightFace CUDA face recognition runtime.")
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        help="Optional image path to run detection on.",
    )
    args = parser.parse_args()

    settings = get_settings()
    recognizer = InsightFaceCudaRecognizer(
        model_name=settings.face_insightface_model,
        det_size=settings.face_insightface_det_size,
        device=settings.face_embedding_device,
        root=settings.face_insightface_root,
    )
    payload: dict[str, object] = {"algorithm": recognizer.info().__dict__}
    if args.image is not None:
        candidates = recognizer.extract(args.image)
        payload["faces"] = [
            {
                "bbox": candidate.bbox,
                "quality_score": candidate.quality_score,
                "embedding_dim": len(candidate.embedding),
                "model": candidate.model,
            }
            for candidate in candidates
        ]
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
