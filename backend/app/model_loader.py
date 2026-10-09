"""Model loading for the unified plant ResNet50 checkpoint.

The production checkpoint (``unified_plant_resnet50.pth``) is a full dict with:
    - ``num_classes``       : int (17)
    - ``class_names``       : list[str], authoritative class order
    - ``model_state_dict``  : state_dict of torchvision ResNet50 with a
                              replaced ``fc`` layer (Linear(2048, 17))
    - ``optimizer_state_dict``: Adam state (not needed for inference)

Architecture and preprocessing mirror ``crop doctor/app1.py`` exactly.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torchvision import models

logger = logging.getLogger("uvicorn.error")

DEFAULT_MODEL_PATH = "model_assets/unified_plant_resnet50.pth"
DEFAULT_DEVICE = "cpu"
ARCHITECTURE = "resnet50"


class ModelService:
    """Loads and holds the trained model. Safe to call ``load()`` repeatedly."""

    def __init__(self) -> None:
        self.model: Optional[nn.Module] = None
        self.class_names: List[str] = []
        self.device: str = DEFAULT_DEVICE
        self.model_path: str = ""
        self.num_classes: int = 0
        self.loaded: bool = False
        self.load_error: Optional[str] = None

    def load(self) -> None:
        """Load the checkpoint from MODEL_PATH (env) into memory.

        Never raises: failures are recorded in ``load_error`` so the API can
        start and report a degraded health status instead of crashing.
        """
        self.loaded = False
        self.load_error = None
        self.model_path = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
        self.device = self._resolve_device()

        try:
            checkpoint = self._read_checkpoint(self.model_path)
            class_names = self._extract_class_names(checkpoint)
            state_dict = self._extract_state_dict(checkpoint)

            model = models.resnet50(weights=None)
            model.fc = nn.Linear(model.fc.in_features, len(class_names))
            model.load_state_dict(state_dict, strict=True)
            model.to(self.device)
            model.eval()

            self.model = model
            self.class_names = list(class_names)
            self.num_classes = len(class_names)
            self.loaded = True
            logger.info(
                "Model loaded: %s classes from %s on %s",
                self.num_classes,
                self.model_path,
                self.device,
            )
        except Exception as exc:  # noqa: BLE001 - reported via /health
            self.model = None
            self.class_names = []
            self.num_classes = 0
            self.load_error = f"{type(exc).__name__}: {exc}"
            logger.error("Model loading failed: %s", self.load_error)

    @staticmethod
    def _resolve_device() -> str:
        requested = os.getenv("DEVICE", DEFAULT_DEVICE).lower()
        if requested in ("cuda", "gpu") and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    @staticmethod
    def _read_checkpoint(path: str) -> Dict[str, Any]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model checkpoint not found: {path}")
        try:
            # weights_only=True is safe and supported on torch>=2.6 (default)
            # and available as an explicit flag on older versions.
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # pragma: no cover - very old torch
            return torch.load(path, map_location="cpu")

    @staticmethod
    def _extract_class_names(checkpoint: Dict[str, Any]) -> List[str]:
        class_names = checkpoint.get("class_names")
        if not class_names:
            raise ValueError("Checkpoint is missing the 'class_names' entry")
        return [str(name) for name in class_names]

    @staticmethod
    def _extract_state_dict(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        state_dict = checkpoint.get("model_state_dict")
        if state_dict is None:
            raise ValueError("Checkpoint is missing the 'model_state_dict' entry")
        # Tolerate DataParallel checkpoints ("module." prefix).
        cleaned: Dict[str, Any] = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                key = key.replace("module.", "", 1)
            cleaned[key] = value
        return cleaned


_service: Optional[ModelService] = None


def get_model_service() -> ModelService:
    """Singleton accessor so the API and tests share one service instance."""
    global _service
    if _service is None:
        _service = ModelService()
    return _service
