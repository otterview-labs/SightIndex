#!/usr/bin/env python3
"""Run the documented upload/process/inspect workflow against your own instance."""

import argparse
import getpass
import json
from pathlib import Path

import httpx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--image", type=Path,
        default=Path(__file__).resolve().parents[1] / "samples" / "nasa-walk.jpg",
    )
    parser.add_argument("--username", help="Prompt for a Basic Auth password if required")
    args = parser.parse_args()
    auth = (args.username, getpass.getpass()) if args.username else None
    # A local evaluation must not accidentally send sample media through a system proxy.
    with httpx.Client(
        base_url=args.base_url.rstrip("/"), auth=auth, timeout=120, trust_env=False,
    ) as client:
        client.get("/health").raise_for_status()
        with args.image.open("rb") as sample:
            uploaded = client.post(
                "/api/images/upload", files={"file": (args.image.name, sample, "image/jpeg")},
            )
        uploaded.raise_for_status()
        record = uploaded.json()
        image_id = record["id"]
        processed = client.post(f"/api/images/{image_id}/process")
        processed.raise_for_status()
        crops = processed.json()
        inspected = client.get(f"/api/images/{image_id}")
        inspected.raise_for_status()
        if inspected.json()["id"] != image_id:
            raise RuntimeError("The inspected image does not match the upload")
        repeated = client.post(f"/api/images/{image_id}/process")
        repeated.raise_for_status()
        if {item["id"] for item in crops} != {item["id"] for item in repeated.json()}:
            raise RuntimeError("Processing retry created a different crop set")
        for item in crops:
            if item["image_id"] != image_id:
                raise RuntimeError("Crop refers to a different image")
            client.get(item["crop_url"]).raise_for_status()
        print(json.dumps({
            "image_id": image_id,
            "image_url": record["image_url"],
            "crops": len(crops),
            "retry_reused_crops": True,
            "console": args.base_url.rstrip("/") + "/observations",
            "note": "Detection counts vary by detector and version. This is not a benchmark.",
        }, indent=2))


if __name__ == "__main__":
    main()
