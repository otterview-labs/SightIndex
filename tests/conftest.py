"""Shared, synthetic image fixtures; no camera data or model downloads are needed."""

import pytest


@pytest.fixture
def sample_jpeg() -> bytes:
    """Return decodable pixels for tests that exercise actual person-crop creation."""

    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    frame = np.full((48, 64, 3), 96, dtype=np.uint8)
    succeeded, encoded = cv2.imencode(".jpg", frame)
    assert succeeded
    return bytes(encoded)
