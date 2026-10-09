"""Inference wrapper for the unified plant ResNet50 model.

Supports both runtimes selected by the loaded backend:
* onnxruntime (primary, CPU, ~170 MB peak RSS)
* PyTorch (fallback, CPU)

Both produce identical predictions (parity-tested).
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from PIL import Image

from app.model_loader import ModelService
from app.preprocessing import preprocess_numpy, preprocess_torch


@dataclass
class PredictionResult:
    predicted_class: str
    confidence: float
    top_predictions: List[Tuple[str, float]]


def predict_image(
    service: ModelService, image: Image.Image, top_k: int = 3
) -> PredictionResult:
    """Run inference on a PIL image and return the top-k
    softmax predictions."""
    if not service.loaded:
        raise RuntimeError("Model is not loaded")

    if service.backend == "onnx":
        return _predict_onnx(service, image, top_k)
    return _predict_pytorch(service, image, top_k)


def _predict_onnx(
    service: ModelService, image: Image.Image, top_k: int
) -> PredictionResult:
    x = preprocess_numpy(image)
    logits = service.onnx_session.run(["logits"], {"input": x})[0][0]

    # Softmax over the logits (same convention as the
    # CrossEntropyLoss-trained PyTorch model).
    e = np.exp(logits - logits.max())
    probs = e / e.sum()

    k = min(top_k, service.num_classes)
    order = np.argsort(-probs, kind="stable")[:k]
    top_predictions = [
        (service.class_names[int(i)], float(probs[i])) for i in order
    ]
    predicted_class, confidence = top_predictions[0]
    return PredictionResult(
        predicted_class=predicted_class,
        confidence=confidence,
        top_predictions=top_predictions,
    )


def _predict_pytorch(
    service: ModelService, image: Image.Image, top_k: int
) -> PredictionResult:
    import torch
    import torch.nn.functional as F

    tensor = preprocess_torch(image).to(service.device)
    num_classes = len(service.class_names)
    k = min(top_k, num_classes)

    with torch.inference_mode():
        outputs = service.model(tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        top_probs, top_indices = torch.topk(probabilities, k=k)

    top_predictions = [
        (service.class_names[int(idx)], float(prob))
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist())
    ]
    predicted_class, confidence = top_predictions[0]
    return PredictionResult(
        predicted_class=predicted_class,
        confidence=confidence,
        top_predictions=top_predictions,
    )
