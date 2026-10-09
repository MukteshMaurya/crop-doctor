"""Integration tests against the real production checkpoint.

These run only when the 270 MB production model is present at
model_assets/unified_plant_resnet50.pth; otherwise they are skipped.
They verify the actual class mapping and that a real prediction
round-trips through the API.
"""

import os
import sys

import pytest

BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_ROOT)

from tests.conftest import make_image_bytes  # noqa: E402

PRODUCTION_MODEL = os.path.join(BACKEND_ROOT, "model_assets", "unified_plant_resnet50.pth")

EXPECTED_CLASSES = [
    "apple", "banana", "blueberry", "cherry", "citrus", "corn",
    "grape", "other_crops", "peach", "pepper_bell", "potato",
    "raspberry", "soybean", "squash", "strawberry", "tomato", "wheat",
]

requires_production_model = pytest.mark.skipif(
    not os.path.exists(PRODUCTION_MODEL),
    reason="production checkpoint not available",
)


@requires_production_model
def test_production_health_and_classes(monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app
    from app.model_loader import get_model_service

    monkeypatch.setenv("MODEL_PATH", PRODUCTION_MODEL)
    service = get_model_service()
    service.loaded = False
    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health["model_loaded"] is True
        assert health["num_classes"] == 17

        classes = client.get("/api/classes").json()
        assert classes["classes"] == EXPECTED_CLASSES

        files = {"file": ("leaf.jpg", make_image_bytes(), "image/jpeg")}
        result = client.post("/api/predict", files=files).json()
        assert result["success"] is True
        assert result["predicted_class"] in EXPECTED_CLASSES
        assert len(result["top_predictions"]) == 3


@requires_production_model
def test_production_checkpoint_class_names():
    """The API classes must match the checkpoint's own class_names."""
    import torch

    checkpoint = torch.load(PRODUCTION_MODEL, map_location="cpu", weights_only=True)
    assert checkpoint["num_classes"] == 17
    assert checkpoint["class_names"] == EXPECTED_CLASSES


PRODUCTION_ONNX = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "model_assets",
    "unified_plant_resnet50.onnx",
)

requires_production_onnx = pytest.mark.skipif(
    not os.path.exists(PRODUCTION_ONNX),
    reason="production ONNX model not available",
)


@requires_production_onnx
def test_production_onnx_health_classes_and_predict(monkeypatch):
    """The production ONNX model must serve the real API:
    healthy status, 17 classes in checkpoint order, and a
    real prediction."""
    from fastapi.testclient import TestClient

    from app.main import app
    from app.model_loader import get_model_service

    monkeypatch.setenv("MODEL_PATH", PRODUCTION_ONNX)
    service = get_model_service()
    service.loaded = False
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        health_json = health.json()
        assert health_json["model_loaded"] is True
        assert health_json["backend"] == "onnx"
        assert health_json["num_classes"] == 17

        classes = client.get("/api/classes").json()
        assert classes["classes"] == EXPECTED_CLASSES

        files = {"file": ("leaf.jpg", make_image_bytes(), "image/jpeg")}
        result = client.post("/api/predict", files=files).json()
        assert result["success"] is True
        assert result["predicted_class"] in EXPECTED_CLASSES
        assert len(result["top_predictions"]) == 3
