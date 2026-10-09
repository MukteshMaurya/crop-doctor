"""API tests for the AI Plant Disease Detector backend.

Inference is exercised against a tiny generated checkpoint (see
conftest.py) so the tests are fast and deterministic; the production
model is exercised separately in test_production_model.py.
"""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from tests.conftest import TEST_CLASSES, make_image_bytes


def test_root_info(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "AI Plant Disease Detector API"
    assert data["status"] == "online"
    assert "/api/predict" in data["endpoints"]["predict"]


def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["device"] == "cpu"
    assert data["architecture"] == "resnet50"
    assert data["num_classes"] == len(TEST_CLASSES)
    assert data["model_path"]
    # The effective CORS allowlist must be visible so browser
    # connection problems can be diagnosed from one endpoint.
    assert isinstance(data["cors_origins"], list)
    assert data["cors_origins"]


def test_health_exposes_dev_default_origins(client, monkeypatch):
    monkeypatch.delenv("FRONTEND_ORIGIN", raising=False)
    response = client.get("/health")
    assert response.status_code == 200
    origins = response.json()["cors_origins"]
    assert "http://localhost:5500" in origins
    assert "http://127.0.0.1:5500" in origins
    for origin in origins:
        assert origin.startswith("http://localhost") or origin.startswith(
            "http://127.0.0.1"
        )


def test_health_reports_configured_frontend_origin(tiny_checkpoint, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", tiny_checkpoint)
    monkeypatch.setenv("FRONTEND_ORIGIN", "https://app.example.com")
    service = __import__("app.model_loader", fromlist=["get_model_service"]).get_model_service()
    service.loaded = False
    from app.main import app

    with TestClient(app) as test_client:
        response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json()["cors_origins"] == ["https://app.example.com"]


def test_health_fails_when_model_missing(broken_client):
    response = broken_client.get("/health")
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "degraded"
    assert data["model_loaded"] is False
    assert data["error"]


def test_classes_endpoint(client):
    response = client.get("/api/classes")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["num_classes"] == 3
    assert data["classes"] == TEST_CLASSES


def test_classes_unavailable_when_model_missing(broken_client):
    response = broken_client.get("/api/classes")
    assert response.status_code == 503
    assert response.json()["success"] is False


def test_predict_valid_image(client):
    files = {"file": ("leaf.jpg", make_image_bytes("JPEG"), "image/jpeg")}
    response = client.post("/api/predict", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["predicted_class"] in TEST_CLASSES
    assert 0.0 <= data["confidence"] <= 1.0
    assert len(data["top_predictions"]) == 3
    for item in data["top_predictions"]:
        assert item["class_name"] in TEST_CLASSES
        assert 0.0 <= item["confidence"] <= 1.0
    # top_predictions must be sorted descending and include the winner
    confs = [item["confidence"] for item in data["top_predictions"]]
    assert confs == sorted(confs, reverse=True)
    assert data["top_predictions"][0]["class_name"] == data["predicted_class"]


def test_predict_png_image(client):
    files = {"file": ("leaf.png", make_image_bytes("PNG"), "image/png")}
    response = client.post("/api/predict", files=files)
    assert response.status_code == 200
    assert response.json()["success"] is True


def test_predict_non_image_upload(client):
    files = {"file": ("notes.txt", b"not an image at all", "text/plain")}
    response = client.post("/api/predict", files=files)
    assert response.status_code == 400
    assert response.json()["success"] is False
    assert "image" in response.json()["error"].lower()


def test_predict_corrupt_image_bytes(client):
    files = {"file": ("leaf.jpg", b"\xff\xd8\xff garbage bytes", "image/jpeg")}
    response = client.post("/api/predict", files=files)
    assert response.status_code == 400


def test_predict_empty_upload(client):
    files = {"file": ("empty.jpg", b"", "image/jpeg")}
    response = client.post("/api/predict", files=files)
    assert response.status_code == 400
    assert "empty" in response.json()["error"].lower()


def test_predict_missing_file_field(client):
    response = client.post("/api/predict")
    assert response.status_code == 422
    assert response.json()["success"] is False


def test_predict_oversized_image(client):
    # 10 MB + 1 byte of valid-looking content; the size guard must
    # reject it before image decoding.
    oversize = b"\x00" * (10 * 1024 * 1024 + 1)
    files = {"file": ("big.jpg", oversize, "image/jpeg")}
    response = client.post("/api/predict", files=files)
    assert response.status_code == 413
    assert "limit" in response.json()["error"].lower()


def test_predict_reports_oversize_via_header(client, monkeypatch):
    # Simulate a Content-Length header that exceeds the limit.
    oversize = b"\x00" * 1024
    response = client.post(
        "/api/predict",
        files={"file": ("big.jpg", oversize, "image/jpeg")},
        headers={"content-length": str(10 * 1024 * 1024 + 1)},
    )
    assert response.status_code == 413


def test_predict_unavailable_when_model_missing(broken_client):
    files = {"file": ("leaf.jpg", make_image_bytes(), "image/jpeg")}
    response = broken_client.post("/api/predict", files=files)
    assert response.status_code == 503
    assert response.json()["success"] is False


def test_cors_allows_configured_origin(client, monkeypatch):
    # The dev-default origins include localhost:5500.
    response = client.get(
        "/health", headers={"Origin": "http://localhost:5500"}
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == (
        "http://localhost:5500"
    )


def test_cors_does_not_allow_unregistered_origin(client):
    response = client.get(
        "/health", headers={"Origin": "https://evil.example.com"}
    )
    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers


def test_cors_env_var_origins(tiny_checkpoint, monkeypatch):
    monkeypatch.setenv("FRONTEND_ORIGIN", "https://app.example.com,https://staging.example.com")
    monkeypatch.setenv("MODEL_PATH", tiny_checkpoint)
    service = __import__("app.model_loader", fromlist=["get_model_service"]).get_model_service()
    service.loaded = False
    from app.main import _cors_origins

    assert _cors_origins() == [
        "https://app.example.com",
        "https://staging.example.com",
    ]


def test_cors_empty_env_var_falls_back_to_dev_defaults(monkeypatch):
    monkeypatch.setenv("FRONTEND_ORIGIN", "   ")
    from app.main import _cors_origins

    origins = _cors_origins()
    assert origins
    assert "http://localhost:5500" in origins


def test_cors_blank_string_is_ignored(monkeypatch):
    monkeypatch.setenv("FRONTEND_ORIGIN", " , https://app.example.com , ")
    from app.main import _cors_origins

    assert _cors_origins() == ["https://app.example.com"]


def test_cors_strips_trailing_slash(monkeypatch):
    # Browsers never send a trailing slash in Origin; tolerate
    # it in configuration so a "https://host/" value still
    # matches requests from "https://host".
    monkeypatch.setenv("FRONTEND_ORIGIN", "https://app.example.com/")
    from app.main import _cors_origins

    assert _cors_origins() == ["https://app.example.com"]
