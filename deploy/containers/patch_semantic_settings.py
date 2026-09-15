import hashlib
from pathlib import Path

BASE_FILES = {
    "app/config/settings.py": "4f6cd2c6235413425bf7f0da22d3e1e28e0dd739df97c6e8b2734ac8d92a6342",
    "app/api/search.py": "5ca13404a066548576cedce279a7d5b682aa83bf9395c9157138c35f167ba48e",
    "app/services/vector_index.py": (
        "fc33668b95e8f69a83c1bfdbab4a10ee5f26d5bf505ed09319067a8b3e00aa67"
    ),
}


def patch_runtime(root: Path) -> None:
    for relative_path, expected in BASE_FILES.items():
        actual = hashlib.sha256((root / relative_path).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Unreviewed base file: {relative_path}")
    settings_file = root / "app/config/settings.py"
    original = settings_file.read_text()
    anchor = "    visual_search_min_score: float = Field(default=0.0, ge=0.0, le=1.0)\n"
    if original.count(anchor) != 1 or "semantic_search_enabled" in original:
        raise RuntimeError("Base settings layout changed")
    modified = original.replace(
        anchor,
        anchor
        + "    semantic_search_enabled: bool = False\n"
        + "    semantic_search_min_score: float = Field(default=0.25, ge=0.0, le=1.0)\n"
        + "    semantic_search_max_scope: int = Field(default=10000, ge=1, le=10000)\n",
    )
    compile(modified, str(settings_file), "exec")
    settings_file.write_text(modified)


if __name__ == "__main__":
    patch_runtime(Path("/opt/sightindex"))
