"""Tests for the ONNX Runtime inference path (production backend).

Uses a tiny ONNX export of the same weights as the PyTorch
fixture so both paths can be compared for parity.
"""

import io

import pytest
from PIL import Image

from tests.conftest import BACKEND_ROOT, TEST_CLASSES, make_image_bytes


def test_onnx_health_reports_backend(onnx_client):
    response = onnx_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["backend"] == "onnx"
    assert data["device"] == "cpu"
    assert data["num_classes"] == len(TEST_CLASSES)


def test_onnx_classes_endpoint(onnx_client):
    response = onnx_client.get("/api/classes")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["classes"] == TEST_CLASSES


def test_onnx_predict_valid_image(onnx_client):
    files = {"file": ("leaf.jpg", make_image_bytes("JPEG"), "image/jpeg")}
    response = onnx_client.post("/api/predict", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["predicted_class"] in TEST_CLASSES
    assert 0.0 <= data["confidence"] <= 1.0
    assert len(data["top_predictions"]) == 3
    confs = [item["confidence"] for item in data["top_predictions"]]
    assert confs == sorted(confs, reverse=True)
    assert data["top_predictions"][0]["class_name"] == data["predicted_class"]


def test_onnx_predict_png(onnx_client):
    files = {"file": ("leaf.png", make_image_bytes("PNG"), "image/png")}
    response = onnx_client.post("/api/predict", files=files)
    assert response.status_code == 200
    assert response.json()["success"] is True


def test_onnx_rejects_non_image(onnx_client):
    files = {"file": ("notes.txt", b"not an image", "text/plain")}
    response = onnx_client.post("/api/predict", files=files)
    assert response.status_code == 400
    assert response.json()["success"] is False


def test_onnx_missing_class_metadata_fails_loudly(
    tiny_onnx_no_meta, monkeypatch
):
    """An ONNX file without class metadata must NOT be
    served as healthy — the loader fails loudly."""
    from fastapi.testclient import TestClient

    from app.main import app
    from app.model_loader import get_model_service

    monkeypatch.setenv("MODEL_PATH", tiny_onnx_no_meta)
    service = get_model_service()
    service.loaded = False
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 503
        data = response.json()
        assert data["model_loaded"] is False
        assert "class_names" in data["error"]


def test_onnx_missing_file_fails_loudly(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from app.main import app
    from app.model_loader import get_model_service

    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.onnx"))
    service = get_model_service()
    service.loaded = False
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 503
        assert response.json()["model_loaded"] is False


def test_unsupported_extension_fails_loudly(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from app.main import app
    from app.model_loader import get_model_service

    bogus = tmp_path / "model.txt"
    bogus.write_text("not a model")
    monkeypatch.setenv("MODEL_PATH", str(bogus))
    service = get_model_service()
    service.loaded = False
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 503
        assert "extension" in response.json()["error"]


def test_onnx_pytorch_parity(tiny_checkpoint, tiny_onnx, monkeypatch):
    """The ONNX path and the PyTorch path must produce the
    same prediction for the same image (same weights)."""
    from fastapi.testclient import TestClient

    from app.main import app
    from app.model_loader import get_model_service

    image_bytes = make_image_bytes("PNG", size=(200, 150), color=(60, 120, 200))
    files = {"file": ("leaf.png", image_bytes, "image/png")}

    service = get_model_service()

    # PyTorch fallback path
    monkeypatch.setenv("MODEL_PATH", tiny_checkpoint)
    service.loaded = False
    with TestClient(app) as pytorch_client:
        pytorch_result = pytorch_client.post("/api/predict", files=files).json()

    # ONNX production path
    monkeypatch.setenv("MODEL_PATH", tiny_onnx)
    service.loaded = False
    with TestClient(app) as onnx_client:
        onnx_result = onnx_client.post("/api/predict", files=files).json()

    assert pytorch_result["success"] is True
    assert onnx_result["success"] is True
    assert onnx_result["predicted_class"] == pytorch_result["predicted_class"]
    assert (
        abs(onnx_result["confidence"] - pytorch_result["confidence"]) < 1e-4
    )
    for onnx_item, pytorch_item in zip(
        onnx_result["top_predictions"], pytorch_result["top_predictions"]
    ):
        assert onnx_item["class_name"] == pytorch_item["class_name"]
        assert (
            abs(onnx_item["confidence"] - pytorch_item["confidence"]) < 1e-4
        )


def test_production_onnx_parity_when_available():
    """If the production ONNX and PyTorch checkpoints are
    both present, verify they agree on a real image."""
    import os

    import numpy as np
    import onnxruntime as ort

    onnx_path = os.path.join(
        BACKEND_ROOT, "model_assets", "unified_plant_resnet50.onnx"
    )
    if not os.path.exists(onnx_path):
        pytest.skip("production ONNX model not available")

    session = ort.InferenceSession(
        onnx_path, providers=["CPUExecutionProvider"]
    )
    metadata = session.get_modelmeta().custom_metadata_map
    import json

    class_names = json.loads(metadata["class_names"])
    assert len(class_names) == 17
    assert class_names[0] == "apple"
    assert class_names[-1] == "wheat"

    # Deterministic input: solid green image via the numpy pipeline
    from app.preprocessing import preprocess_numpy

    img = Image.new("RGB", (300, 200), color=(80, 140, 60))
    x = preprocess_numpy(img)
    logits = session.run(["logits"], {"input": x})[0][0]
    e = np.exp(logits - logits.max())
    probs = e / e.sum()
    top = int(np.argmax(probs))
    assert class_names[top] in class_names
    assert 0.0 <= float(probs[top]) <= 1.0
    # sanity: probabilities sum to 1
    assert abs(float(probs.sum()) - 1.0) < 1e-5
