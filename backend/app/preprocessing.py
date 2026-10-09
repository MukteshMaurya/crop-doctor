"""Image preprocessing for the unified plant ResNet50 model.

Preprocessing is identical to the training/gradio pipeline in
``crop doctor/app1.py``:
    Resize((224, 224)) -> ToTensor -> Normalize(ImageNet mean/std)
Input images are converted to RGB before the transform.
"""

from typing import List

import torch
from PIL import Image
from torchvision import transforms

IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: List[float] = [0.229, 0.224, 0.225]
INPUT_SIZE = (224, 224)

_transform: transforms.Compose = transforms.Compose(
    [
        transforms.Resize(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


def preprocess(image: Image.Image) -> torch.Tensor:
    """Convert a PIL image (any mode) to a normalized batch tensor."""
    rgb = image.convert("RGB")
    return _transform(rgb).unsqueeze(0)
