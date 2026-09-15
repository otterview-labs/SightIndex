from pathlib import Path


def patch_runtime(root: Path) -> None:
    """Backport only upstream-key separation into an already reviewed API image."""
    settings_path = root / "app/config/settings.py"
    embeddings_path = root / "app/services/embeddings.py"
    settings = settings_path.read_text()
    embeddings = embeddings_path.read_text()
    setting_line = "    visual_embedding_service_api_key: str | None = None\n"
    original = (
        "        elif provider in self.qwen3_vl_http_providers:\n"
        "            vector = _Qwen3VLHTTPVisualRuntime(\n"
        "                service_url=self.settings.visual_embedding_service_url,\n"
        "                api_key=self.settings.visual_embedding_service_api_key,\n"
    )
    replacement = original.replace(
        "api_key=self.settings.visual_embedding_service_api_key,",
        "api_key=(self.settings.visual_embedding_upstream_api_key\n"
        "                         or self.settings.visual_embedding_service_api_key),",
    )
    if "visual_embedding_upstream_api_key" in settings:
        raise RuntimeError("Base image already contains upstream-key support; review the backport")
    if settings.count(setting_line) != 1 or embeddings.count(original) != 2:
        raise RuntimeError("Base image layout changed; refusing an unreviewed backport")
    settings = settings.replace(
        setting_line, setting_line + "    visual_embedding_upstream_api_key: str | None = None\n"
    )
    embeddings = embeddings.replace(original, replacement)
    compile(settings, str(settings_path), "exec")
    compile(embeddings, str(embeddings_path), "exec")
    settings_path.write_text(settings)
    embeddings_path.write_text(embeddings)


if __name__ == "__main__":
    patch_runtime(Path("/opt/sightindex"))
