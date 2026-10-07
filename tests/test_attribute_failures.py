"""Failure evidence is classified without reflecting upstream bodies or local paths."""

from __future__ import annotations

import json
from urllib.error import HTTPError, URLError

import pytest

from app.services.attribute_failures import (
    classify_attribute_failure,
    safe_checkpoint_error,
    stored_attribute_failure,
)


@pytest.mark.parametrize(
    "exception,code",
    [
        (TimeoutError("private-token"), "timeout"),
        (URLError(TimeoutError("private-token")), "timeout"),
        (ConnectionResetError("private-token"), "unavailable"),
        (FileNotFoundError("/private/secret-media.jpg"), "media_missing"),
        (PermissionError("/private/secret-media.jpg"), "media_unreadable"),
        (json.JSONDecodeError("private-token", "private-response-body", 1), "invalid_response"),
        (RuntimeError("private-token /private/secret-media.jpg"), "unknown"),
    ],
)
def test_exception_classifier_never_reflects_exception_body(exception, code):
    failure = classify_attribute_failure(exception)
    assert failure.code == code
    assert "private" not in failure.message + failure.stored


@pytest.mark.parametrize(
    "status,code",
    [
        (401, "authentication"),
        (403, "authentication"),
        (404, "configuration"),
        (429, "rate_limited"),
        (500, "unavailable"),
        (503, "unavailable"),
    ],
)
def test_http_status_classification_does_not_read_body_or_url(status, code):
    error = HTTPError("https://private.test/?token=private-token", status, "private-body", {}, None)
    failure = classify_attribute_failure(error)
    assert failure.code == code
    assert "private" not in failure.message + failure.stored


def test_wrapper_exception_chain_preserves_type_classification():
    wrapper = RuntimeError("never-reflect-this-message")
    wrapper.__cause__ = TimeoutError("never-reflect-this-message")
    assert classify_attribute_failure(wrapper).code == "timeout"


def test_exception_cycles_are_bounded():
    error = RuntimeError("never-reflect-this-message")
    error.__cause__ = error
    assert classify_attribute_failure(error).code == "unknown"


@pytest.mark.parametrize(
    "value",
    [
        None,
        12,
        {"secret": "token"},
        "unknown legacy error /private/file?token=secret",
        "attribute_error:unknown?token=secret",
        "attribute_error:timeout\nsecret",
    ],
)
def test_old_or_untrusted_stored_errors_are_generic(value):
    failure = stored_attribute_failure(value)
    assert failure.code == "unknown"
    assert "secret" not in failure.message + failure.stored


def test_known_stored_code_and_checkpoint_uuid_are_preserved():
    value = "4d52b04a-ac43-4a8a-8287-0eb6cb2b22ce: attribute_error:timeout"
    assert safe_checkpoint_error(value) == value
    assert safe_checkpoint_error(value.replace("attribute_error:timeout", "legacy-secret")) == (
        "4d52b04a-ac43-4a8a-8287-0eb6cb2b22ce: attribute_error:unknown"
    )


def test_local_vlm_known_invalid_response_is_classified():
    from app.services.vlm import VLMRuntimeError

    failure = classify_attribute_failure(
        VLMRuntimeError("VLM structured analysis did not return JSON")
    )
    assert failure.code == "invalid_response"
