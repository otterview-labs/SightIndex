"""Bounded, non-reflective failure descriptions for optional attribute extraction."""

from __future__ import annotations

import errno
import json
import uuid
from dataclasses import dataclass
from types import MappingProxyType
from urllib.error import HTTPError, URLError


@dataclass(frozen=True)
class AttributeFailure:
    code: str
    message: str
    retryable: bool

    @property
    def stored(self) -> str:
        return f"attribute_error:{self.code}"


FAILURES = MappingProxyType(
    {
        item.code: item
        for item in (
            AttributeFailure(
                "unknown", "Attribute analysis failed; inspect the upstream service.", False
            ),
            AttributeFailure("timeout", "The attribute service timed out.", True),
            AttributeFailure(
                "unavailable", "The attribute service is temporarily unavailable.", True
            ),
            AttributeFailure("rate_limited", "The attribute service rate limit was reached.", True),
            AttributeFailure(
                "authentication", "The attribute service rejected authentication.", False
            ),
            AttributeFailure(
                "configuration", "Attribute service configuration needs review.", False
            ),
            AttributeFailure(
                "invalid_response", "The attribute service returned an invalid response.", True
            ),
            AttributeFailure("media_missing", "The crop image is unavailable.", False),
            AttributeFailure("media_unreadable", "The crop image cannot be read.", False),
        )
    }
)
_EXACT_MESSAGES = {
    "VLM structured analysis is not configured": "configuration",
    "VLM structured analysis did not return JSON": "invalid_response",
    "VLM structured JSON root must be an object": "invalid_response",
    "VLM structured analysis response is invalid": "invalid_response",
    "VLM response did not include choices": "invalid_response",
    "VLM response choice is invalid": "invalid_response",
    "VLM response did not include a message": "invalid_response",
    "VLM response content is invalid": "invalid_response",
}


def stored_attribute_failure(value: object) -> AttributeFailure:
    """Whitelist known codes only; legacy exception text is never reflected."""

    if isinstance(value, str):
        if value.startswith("attribute_error:"):
            return FAILURES.get(value.removeprefix("attribute_error:"), FAILURES["unknown"])
        if value in _EXACT_MESSAGES:
            return FAILURES[_EXACT_MESSAGES[value]]
        if value.startswith("Person crop image does not exist:"):
            return FAILURES["media_missing"]
    return FAILURES["unknown"]


def classify_attribute_failure(error: BaseException) -> AttributeFailure:
    """Use exception types and HTTP status, not arbitrary upstream response bodies."""

    current: BaseException | None = error
    visited: set[int] = set()
    for _ in range(8):
        if current is None or id(current) in visited:
            break
        visited.add(id(current))
        if isinstance(current, HTTPError):
            if current.code in (401, 403):
                return FAILURES["authentication"]
            if current.code == 429:
                return FAILURES["rate_limited"]
            if current.code >= 500:
                return FAILURES["unavailable"]
            return FAILURES["configuration"]
        if isinstance(current, TimeoutError):
            return FAILURES["timeout"]
        if isinstance(current, json.JSONDecodeError):
            return FAILURES["invalid_response"]
        if isinstance(current, FileNotFoundError):
            return FAILURES["media_missing"]
        if isinstance(current, PermissionError):
            return FAILURES["media_unreadable"]
        if isinstance(current, ConnectionError) or (
            isinstance(current, OSError)
            and current.errno
            in {errno.ECONNREFUSED, errno.ECONNRESET, errno.EHOSTUNREACH, errno.ENETUNREACH}
        ):
            return FAILURES["unavailable"]
        if isinstance(current, URLError) and isinstance(current.reason, BaseException):
            current = current.reason
            continue
        # These are locally generated, fixed VLM errors; matching them selects a fixed code
        # without returning the message or any path that follows its known prefix.
        if (
            type(current).__module__ == "app.services.vlm"
            and type(current).__name__ == "VLMRuntimeError"
        ):
            known = stored_attribute_failure(str(current))
            if known.code != "unknown":
                return known
        current = current.__cause__ or (
            None if current.__suppress_context__ else current.__context__
        )
    return FAILURES["unknown"]


def safe_checkpoint_error(value: object) -> str:
    """Keep a valid crop UUID prefix when present, never an untrusted error body."""

    if isinstance(value, str):
        prefix, separator, error = value.partition(": ")
        if separator:
            try:
                crop_id = uuid.UUID(prefix)
            except ValueError:
                pass
            else:
                return f"{crop_id}: {stored_attribute_failure(error).stored}"
    return stored_attribute_failure(value).stored
