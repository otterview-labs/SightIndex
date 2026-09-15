import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

MODEL_ID = "Qwen/Qwen3-VL-Embedding-2B"
WEIGHTS_SHA256 = "c73fa9caeddeb3ff831d46c085a7a5708343248ca777e90f2d486964464509c1"
API_ROOT = f"https://modelscope.cn/api/v1/models/{MODEL_ID}"


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            checksum.update(chunk)
    return checksum.hexdigest()


def read_manifest(destination: Path) -> dict:
    manifest_path = destination / "download-manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        request = Request(
            f"{API_ROOT}/repo/files?Revision=master&Recursive=true",
            headers={"User-Agent": "SightIndex-model-download"},
        )
        with urlopen(request, timeout=30) as response:
            result = json.load(response)
        if result.get("Code") != 200:
            raise RuntimeError("ModelScope did not return a successful file manifest")
        files = [
            item for item in result["Data"]["Files"]
            if item["Type"] == "blob" and item.get("Sha256")
        ]
        manifest = {"model": MODEL_ID, "files": files}
    weights = next(
        item for item in manifest["files"] if item["Path"] == "model.safetensors"
    )
    if manifest["model"] != MODEL_ID or weights["Sha256"] != WEIGHTS_SHA256:
        raise RuntimeError("Unexpected model or weights revision; refusing an unreviewed download")
    for item in manifest["files"]:
        relative = Path(item["Path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("Unsafe model path")
        if len(item["Sha256"]) != 64 or not item.get("Revision"):
            raise RuntimeError("Missing file integrity or revision metadata")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def download(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(destination)
    files = sorted(manifest["files"], key=lambda item: item["Size"])
    print(f"MODEL {MODEL_ID} files={len(files)} bytes={sum(item['Size'] for item in files)}",
          flush=True)
    for item in files:
        target = destination / item["Path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.stat().st_size == item["Size"] and digest(target) == item["Sha256"]:
                print(f"VERIFIED_EXISTING {item['Path']}", flush=True)
                continue
            raise RuntimeError(f"Existing file does not match manifest: {item['Path']}")
        partial = target.with_name(target.name + ".partial")
        url = (
            f"{API_ROOT}/repo?Revision={quote(item['Revision'], safe='')}"
            f"&FilePath={quote(item['Path'], safe='')}"
        )
        if not partial.exists() or partial.stat().st_size != item["Size"]:
            print(f"DOWNLOADING {item['Path']} bytes={item['Size']}", flush=True)
            subprocess.run(
                [
                    "curl", "--fail", "--location", "--show-error", "--silent",
                    "--connect-timeout", "20", "--max-time", "7200",
                    "--retry", "4", "--retry-delay", "5", "--continue-at", "-",
                    "--output", str(partial), url,
                ],
                check=True,
                timeout=7400,
            )
        if partial.stat().st_size != item["Size"] or digest(partial) != item["Sha256"]:
            raise RuntimeError(f"Download integrity mismatch: {item['Path']}")
        partial.replace(target)
        print(f"VERIFIED {item['Path']} sha256={item['Sha256']}", flush=True)
    (destination / "verified.sha256").write_text(
        "".join(f"{item['Sha256']}  {item['Path']}\n" for item in files),
        encoding="utf-8",
    )
    print("ALL_MODEL_FILES_VERIFIED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    download(parser.parse_args().destination)
