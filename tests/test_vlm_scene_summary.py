import base64
import importlib
import json
import sys
from urllib import request

from fastapi.testclient import TestClient

from app.services.label_catalog import normalize_labels


class _FakeHTTPResponse:
    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "labels": [
                                        "category:person",
                                        "upper_color:White",
                                        "lower_color.black",
                                        "carried_item:backpack",
                                        "white",
                                        "upper_white",
                                        "unknown_field:value",
                                    ]
                                }
                            )
                        }
                    }
                ]
            }
        ).encode("utf-8")


def test_normalize_labels_accepts_only_catalog_labels() -> None:
    assert normalize_labels(
        [
            "category:person",
            "upper_color.White",
            "lower_color.black",
            "sports_car",
            "vehicle_type:sports-car",
            "accessory:mask",
            "accessory:glasses",
            "gender:male",
            "gender:female",
        ]
    ) == [
        "category:person",
        "upper_color:white",
        "lower_color:black",
        "vehicle_type:sports_car",
        "accessory:mask",
        "accessory:glasses",
        "gender:male",
    ]


def test_scene_summary_returns_normalized_labels(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'scene-summary.db'}")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("VLM_BASE_URL", "http://vlm.example/v1")
    monkeypatch.setenv("VLM_MODEL", "qwen-vl")
    captured: dict[str, object] = {}

    def fake_urlopen(req: request.Request, timeout: int) -> _FakeHTTPResponse:
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeHTTPResponse()

    monkeypatch.setattr("app.services.vlm.request.urlopen", fake_urlopen)

    for module_name in list(sys.modules):
        if module_name == "main" or module_name.startswith("app."):
            sys.modules.pop(module_name)
    main = importlib.import_module("main")

    image_base64 = base64.b64encode(b"fake image").decode("ascii")
    with TestClient(main.create_app()) as client:
        response = client.post(
            "/api/vlm/scene-summary",
            json={"image_base64": image_base64, "image_filename": "frame.jpg"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "labels": [
            "category:person",
            "upper_color:white",
            "lower_color:black",
            "carried_item:backpack",
        ]
    }
    assert captured["url"] == "http://vlm.example/v1/chat/completions"
    assert captured["body"]["messages"][1]["content"][1]["type"] == "image_url"
