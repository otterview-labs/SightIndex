import base64
import json
import os
from collections.abc import Mapping
from urllib import request


def check_api(environment: Mapping[str, str] | None = None) -> None:
    environment = os.environ if environment is None else environment
    credentials = (
        f"{environment['APP_BASIC_AUTH_USERNAME']}:{environment['APP_BASIC_AUTH_PASSWORD']}"
    )
    headers = {
        "Authorization": f"Basic {base64.b64encode(credentials.encode()).decode('ascii')}"
    }
    counts_request = request.Request("http://127.0.0.1:8000/api/media/counts", headers=headers)
    with request.urlopen(counts_request, timeout=5) as response:
        counts = json.load(response)
    if not isinstance(counts, dict) or not all(
        isinstance(counts.get(key), int) and not isinstance(counts[key], bool) and counts[key] >= 0
        for key in ("image_with_crops_count", "person_crop_count")
    ):
        raise RuntimeError("API database check returned invalid media counts")
    if environment.get("REID_ENABLED", "false").strip().lower() in {"true", "1", "yes", "on"}:
        status_request = request.Request(
            "http://127.0.0.1:8000/api/reid/status", headers=headers
        )
        with request.urlopen(status_request, timeout=8) as response:
            status = json.load(response)
        if not isinstance(status, dict) or status.get("ready") is not True:
            raise RuntimeError("ReID service or index is not ready")


if __name__ == "__main__":
    check_api()
