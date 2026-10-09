"""Shared fixtures: build a tiny ResNet50 checkpoint (and
an ONNX export of the same weights) so tests exercise the
real model-loading and inference code paths without the
production model files."""

import json
import os
import sys

import pytest
import torch
import torch.nn as nn
from fastapi.testclient import TestClient
from torchvision import models

BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_ROOT)

from app.main import app  # noqa: E402
from app.model_loader import get_model_service  # noqa: E402

TEST_CLASSES = ["alpha_crop", "beta_crop", "gamma_crop"]


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory):
    """Create a small valid checkpoint in the production format."""
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(TEST_CLASSES))
    path = tmp_path_factory.mktemp("models") / "tiny_resnet50.pth"
    torch.save(
        {
            "num_classes": len(TEST_CLASSES),
            "class_names": TEST_CLASSES,
            "model_state_dict": model.state_dict(),
        },
        str(path),
    )
    return str(path)


@pytest.fixture(scope="session")
def tiny_onnx(tiny_checkpoint, tmp_path_factory):
    """Export the same tiny weights to ONNX with the class
    mapping embedded as metadata (production format)."""
    import onnx

    checkpoint = torch.load(
        tiny_checkpoint, map_location="cpu", weights_only=True
    )
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(
        model.fc.in_features, len(checkpoint["class_names"])
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    path = tmp_path_factory.mktemp("models") / "tiny_resnet50.onnx"
    torch.onnx.export(
        model,
        torch.randn(1, 3, 224, 224),
        str(path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    onnx_model = onnx.load(str(path))
    onnx_model.metadata_props.append(
        onnx.StringStringEntryProto(
            key="class_names",
            value=json.dumps(checkpoint["class_names"]),
        )
    )
    onnx_model.metadata_props.append(
        onnx.StringStringEntryProto(
            key="num_classes",
            value=str(len(checkpoint["class_names"])),
        )
    )
    onnx.save(onnx_model, str(path))
    return str(path)


@pytest.fixture(scope="session")
def tiny_onnx_no_meta(tiny_onnx, tmp_path_factory):
    """Same ONNX file but with the class metadata removed,
    to verify the loader fails loudly instead of guessing."""
    import onnx

    onnx_model = onnx.load(tiny_onnx)
    del onnx_model.metadata_props[:]
    path = tmp_path_factory.mktemp("models") / "tiny_no_meta.onnx"
    onnx.save(onnx_model, str(path))
    return str(path)


@pytest.fixture()
def client(tiny_checkpoint, monkeypatch):
    """TestClient with the tiny PyTorch checkpoint loaded
    via the real lifespan (exercises the fallback path)."""
    monkeypatch.setenv("MODEL_PATH", tiny_checkpoint)
    service = get_model_service()
    service.loaded = False
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def onnx_client(tiny_onnx, monkeypatch):
    """TestClient with the tiny ONNX model loaded via the
    real lifespan (exercises the production path)."""
    monkeypatch.setenv("MODEL_PATH", tiny_onnx)
    service = get_model_service()
    service.loaded = False
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def broken_client(monkeypatch, tmp_path):
    """TestClient whose MODEL_PATH points to a missing file."""
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "does_not_exist.pth"))
    service = get_model_service()
    service.loaded = False
    with TestClient(app) as test_client:
        yield test_client


def make_image_bytes(fmt="JPEG", size=(64, 64), color=(80, 140, 60)):
    """Return bytes of a small valid in-memory image."""
    from PIL import Image

    buf = __import__("io").BytesIO()
    Image.new("RGB", size, color=color).save(buf, format=fmt)
    return buf.getvalue()
