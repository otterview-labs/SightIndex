from __future__ import annotations

import os
from pathlib import Path

from modelscope import snapshot_download


def main() -> None:
    model_id = os.environ.get("MODEL_ID", "Qwen/Qwen3-VL-Reranker-2B")
    cache_dir = Path(os.environ.get("MODEL_CACHE_DIR", "data/models/modelscope"))
    marker = Path(os.environ.get("MODEL_PATH_FILE", "data/models/Qwen3-VL-Reranker-2B.path"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    marker.parent.mkdir(parents=True, exist_ok=True)
    model_path = snapshot_download(model_id, cache_dir=str(cache_dir))
    marker.write_text(str(model_path))
    print(model_path)


if __name__ == "__main__":
    main()
