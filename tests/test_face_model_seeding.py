import pytest

from app.config.settings import Settings
from app.face_algorithms import InsightFaceCudaRecognizer
from app.services.faces import face_runtime_status


def test_a_missing_model_pack_fails_fast_instead_of_downloading(tmp_path):
    """The download runs inline in the caller's thread; an eight-hour request is not an option."""

    recognizer = InsightFaceCudaRecognizer(root=tmp_path, allow_download=False)

    with pytest.raises(ValueError) as excinfo:
        recognizer._require_model_present()

    message = str(excinfo.value)
    assert "buffalo_l" in message
    assert str(tmp_path) in message, "the error must say where to put the model"
    assert "FACE_INSIGHTFACE_ALLOW_DOWNLOAD" in message, "and how to opt in"


def test_a_seeded_model_pack_passes(tmp_path):
    (tmp_path / "models" / "buffalo_l").mkdir(parents=True)

    InsightFaceCudaRecognizer(root=tmp_path, allow_download=False)._require_model_present()


def test_opting_in_skips_the_check(tmp_path):
    InsightFaceCudaRecognizer(root=tmp_path, allow_download=True)._require_model_present()


def test_face_runtime_status_requires_model_files_and_requested_provider(
    monkeypatch, tmp_path
):
    model_dir = tmp_path / "models" / "buffalo_l"
    model_dir.mkdir(parents=True)
    (model_dir / "det_10g.onnx").write_bytes(b"detector")
    (model_dir / "w600k_r50.onnx").write_bytes(b"recognizer")
    monkeypatch.setitem(
        face_runtime_status.__globals__["InsightFaceCudaRecognizer"].info.__globals__,
        "_available_onnx_providers",
        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    status = face_runtime_status(
        Settings(
            data_dir=tmp_path,
            face_insightface_root=tmp_path,
            face_embedding_device="cuda:0",
        )
    )

    assert status.ready is True
    assert status.error is None
    assert status.model == "insightface-buffalo_l"


def test_face_runtime_status_explains_cuda_degradation(monkeypatch, tmp_path):
    monkeypatch.setitem(
        face_runtime_status.__globals__["InsightFaceCudaRecognizer"].info.__globals__,
        "_available_onnx_providers",
        lambda: ["CPUExecutionProvider"],
    )

    status = face_runtime_status(
        Settings(
            data_dir=tmp_path,
            face_insightface_root=tmp_path,
            face_embedding_device="cuda:0",
        )
    )

    assert status.ready is False
    assert "missing model files" in status.error
    assert "CUDAExecutionProvider is unavailable" in status.error
