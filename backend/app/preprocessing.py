"""Image preprocessing for the unified plant ResNet50 model.

Both paths reproduce the exact training pipeline from
``crop doctor/app1.py``:
    Resize((224, 224)) -> ToTensor -> Normalize(ImageNet mean/std)
on an RGB-converted image.

* ``preprocess_numpy``  - ONNX path (PIL + numpy only)
* ``preprocess_torch``  - PyTorch fallback (torchvision)

The two paths are parity-tested in tests/test_preprocessing.py.
"""

from typing import Any

import numpy as np
from PIL import Image

IMAGENET_MEAN: np.ndarray = np.array(
    [0.485, 0.456, 0.406], dtype=np.float32
)
IMAGENET_STD: np.ndarray = np.array(
    [0.229, 0.224, 0.225], dtype=np.float32
)
INPUT_SIZE = (224, 224)

_torch_transform: Any = None


def preprocess_numpy(image: Image.Image) -> np.ndarray:
    """PIL + numpy pipeline returning a (1, 3, 224, 224)
    float32 batch, matching the torchvision pipeline."""
    img = image.convert("RGB").resize(INPUT_SIZE, Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC, 0..1
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr.transpose(2, 0, 1)[None, ...]  # 1CHW


def preprocess_torch(image: Image.Image):
    """Original torchvision pipeline (lazy import keeps torch
    out of memory when the ONNX path is used)."""
    global _torch_transform
    if _torch_transform is None:
        from torchvision import transforms

        _torch_transform = transforms.Compose(
            [
                transforms.Resize(INPUT_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
    return _torch_transform(image.convert("RGB")).unsqueeze(0)
