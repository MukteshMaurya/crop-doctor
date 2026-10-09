"""Inference wrapper for the unified plant ResNet50 model."""

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

from app.model_loader import ModelService
from app.preprocessing import preprocess


@dataclass
class PredictionResult:
    predicted_class: str
    confidence: float
    top_predictions: List[Tuple[str, float]]


def predict_image(service: ModelService, image: Image.Image, top_k: int = 3) -> PredictionResult:
    """Run inference on a PIL image and return the top-k softmax predictions."""
    if not service.loaded or service.model is None:
        raise RuntimeError("Model is not loaded")

    tensor = preprocess(image).to(service.device)
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
