"""Public configuration names retain the local capture setting as a fallback."""

import pytest
from pydantic import ValidationError

from app.config.settings import Settings


@pytest.fixture(autouse=True)
def clean_idle_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LINE_CROSSING_TRACK_IDLE_SECONDS", raising=False)
    monkeypatch.delenv("COUNTING_TRACK_IDLE_SECONDS", raising=False)


@pytest.mark.parametrize(
    "name", ["line_crossing_track_idle_seconds", "counting_track_idle_seconds"]
)
def test_constructor_alias(name: str) -> None:
    settings = Settings(_env_file=None, **{name: 12.0})
    assert settings.line_crossing_track_idle_seconds == 12.0
    assert settings.counting_track_idle_seconds == 12.0


@pytest.mark.parametrize(
    "name", ["LINE_CROSSING_TRACK_IDLE_SECONDS", "COUNTING_TRACK_IDLE_SECONDS"]
)
def test_environment_alias(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    monkeypatch.setenv(name, "13")
    assert Settings(_env_file=None).line_crossing_track_idle_seconds == 13.0


def test_public_environment_name_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LINE_CROSSING_TRACK_IDLE_SECONDS", "11")
    monkeypatch.setenv("COUNTING_TRACK_IDLE_SECONDS", "29")
    settings = Settings(_env_file=None)
    assert settings.line_crossing_track_idle_seconds == 11.0
    assert settings.counting_track_idle_seconds == 11.0


def test_explicit_constructor_setting_overrides_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LINE_CROSSING_TRACK_IDLE_SECONDS", "11")
    settings = Settings(_env_file=None, counting_track_idle_seconds=14.0)
    assert settings.line_crossing_track_idle_seconds == 14.0


@pytest.mark.parametrize("value", [0.0, 601.0])
def test_legacy_alias_preserves_public_validation(value: float) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, counting_track_idle_seconds=value)
