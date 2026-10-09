"""Preprocessing parity: the numpy pipeline (ONNX path) must
reproduce the torchvision pipeline (PyTorch path) exactly."""

import numpy as np
import pytest
from PIL import Image

from app.preprocessing import INPUT_SIZE, preprocess_numpy, preprocess_torch


@pytest.mark.parametrize("size,color", [
    ((64, 64), (80, 140, 60)),
    ((300, 200), (200, 100, 50)),
    ((1000, 400), (10, 20, 30)),
    ((50, 500), (255, 255, 255)),
])
def test_numpy_matches_torchvision(size, color):
    image = Image.new("RGB", size, color=color)

    numpy_batch = preprocess_numpy(image)
    torch_batch = preprocess_torch(image)

    assert numpy_batch.shape == (1, 3, *INPUT_SIZE)
    assert tuple(torch_batch.shape) == (1, 3, *INPUT_SIZE)

    numpy_arr = numpy_batch.astype(np.float64)
    torch_arr = torch_batch.numpy().astype(np.float64)
    max_diff = float(np.max(np.abs(numpy_arr - torch_arr)))
    assert max_diff < 1e-5, f"preprocessing mismatch: {max_diff}"


def test_numpy_preprocess_dtype_and_range():
    image = Image.new("RGB", (64, 64), color=(128, 64, 32))
    batch = preprocess_numpy(image)
    assert batch.dtype == np.float32
    assert batch.shape == (1, 3, 224, 224)
    # normalized values stay in a sane range
    assert float(batch.min()) > -3.0
    assert float(batch.max()) < 3.0


def test_preprocess_converts_grayscale_to_rgb():
    image = Image.new("L", (64, 64), color=128)
    batch = preprocess_numpy(image)
    assert batch.shape == (1, 3, 224, 224)
