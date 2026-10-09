"""Shared fixtures: build a tiny ResNet50 checkpoint so tests exercise
the real model-loading and inference code paths without the 270 MB
production file."""

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


@pytest.fixture()
def client(tiny_checkpoint, monkeypatch):
    """TestClient with the tiny checkpoint loaded via the real lifespan."""
    monkeypatch.setenv("MODEL_PATH", tiny_checkpoint)
    service = get_model_service()
    service.loaded = False  # force reload with the new path
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
