"""Model loading for the unified plant ResNet50 checkpoint.

Two inference backends are supported, selected by the
extension of MODEL_PATH:

* ``.onnx`` (default, production) - ONNX Runtime, CPU only.
  The ONNX file carries the trained weights and the class
  mapping in its metadata, so torch is never imported and
  peak memory stays near 170 MB (vs ~670 MB for the legacy
  torch.load of the optimizer-laden checkpoint).
* ``.pth`` (fallback) - PyTorch. Loads the checkpoint with
  mmap (when available), builds the model first, copies the
  state dict into it, then frees the checkpoint immediately.

Architecture and preprocessing mirror ``crop doctor/app1.py``
exactly. The model is loaded once per process, from the
FastAPI lifespan (never at module import), and the app
reports a degraded health status until loading succeeds.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger("uvicorn.error")

DEFAULT_MODEL_PATH = "model_assets/unified_plant_resnet50.onnx"
DEFAULT_DEVICE = "cpu"
ARCHITECTURE = "resnet50"


class ModelService:
    """Loads and holds the trained model. Safe to call ``load()``
    repeatedly; the model is loaded once per process via the
    application lifespan."""

    def __init__(self) -> None:
        self.onnx_session: Any = None  # onnxruntime.InferenceSession
        self.model: Optional[Any] = None  # torch.nn.Module (fallback)
        self.class_names: List[str] = []
        self.device: str = DEFAULT_DEVICE
        self.model_path: str = ""
        self.num_classes: int = 0
        self.loaded: bool = False
        self.load_error: Optional[str] = None
        self.backend: Optional[str] = None  # "onnx" | "pytorch"

    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load the model from MODEL_PATH (env).

        Never raises: failures are recorded in ``load_error`` so
        the API can start and report a degraded health status
        instead of crashing or reporting a false healthy state.
        """
        self.loaded = False
        self.load_error = None
        self.backend = None
        self.model_path = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
        self.device = self._resolve_device()

        ext = os.path.splitext(self.model_path)[1].lower()
        try:
            if ext == ".onnx":
                self._load_onnx()
            elif ext in (".pth", ".pt", ".ckpt"):
                self._load_pytorch()
            else:
                raise ValueError(
                    f"Unsupported model file extension '{ext}' "
                    f"for MODEL_PATH: {self.model_path}. "
                    "Expected a .onnx or .pth file."
                )
        except Exception as exc:  # noqa: BLE001 - reported via /health
            self._record_failure(exc)

    def _record_failure(self, exc: Exception) -> None:
        self.onnx_session = None
        self.model = None
        self.class_names = []
        self.num_classes = 0
        self.backend = None
        self.loaded = False
        self.load_error = f"{type(exc).__name__}: {exc}"
        logger.error("Model loading failed: %s", self.load_error)

    # ------------------------------------------------------------------
    # ONNX Runtime path (production, CPU)
    # ------------------------------------------------------------------

    @staticmethod
    def _ort_thread_count() -> int:
        raw = os.getenv("ORT_INTRA_OP_THREADS", "2").strip()
        try:
            return max(1, int(raw))
        except ValueError:
            return 2

    def _load_onnx(self) -> None:
        # Imported lazily so the ONNX path never pays torch's
        # ~250 MB import cost.
        import onnxruntime as ort

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"ONNX model not found: {self.model_path}"
            )

        options = ort.SessionOptions()
        options.intra_op_num_threads = self._ort_thread_count()

        session = ort.InferenceSession(
            self.model_path,
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

        metadata = session.get_modelmeta().custom_metadata_map
        raw_names = metadata.get("class_names")
        if not raw_names:
            raise ValueError(
                "ONNX model is missing the 'class_names' metadata "
                "entry. Re-export it with tools/export_onnx.py."
            )
        class_names = [str(name) for name in json.loads(raw_names)]
        if not class_names:
            raise ValueError("ONNX metadata 'class_names' is empty")

        self.onnx_session = session
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.backend = "onnx"
        self.device = "cpu"  # ONNX path always runs on CPU
        self.loaded = True
        logger.info(
            "Model loaded (onnxruntime): %s classes from %s",
            self.num_classes,
            self.model_path,
        )

    # ------------------------------------------------------------------
    # PyTorch fallback path
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device() -> str:
        requested = os.getenv("DEVICE", DEFAULT_DEVICE).lower()
        if requested in ("cuda", "gpu"):
            try:
                import torch

                if torch.cuda.is_available():
                    return "cuda"
            except Exception:  # noqa: BLE001 - torch optional
                pass
        return "cpu"

    @staticmethod
    def _read_checkpoint(path: str) -> Dict[str, Any]:
        import torch

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model checkpoint not found: {path}"
            )
        try:
            # mmap keeps the (unused) optimizer-state tensors
            # out of RSS; falls back for older torch versions.
            return torch.load(
                path, map_location="cpu", weights_only=True, mmap=True
            )
        except TypeError:  # pragma: no cover - torch < 2.1
            return torch.load(path, map_location="cpu", weights_only=True)

    def _load_pytorch(self) -> None:
        import gc

        import torch.nn as nn
        from torchvision import models

        checkpoint = self._read_checkpoint(self.model_path)
        class_names = checkpoint.get("class_names")
        if not class_names:
            raise ValueError(
                "Checkpoint is missing the 'class_names' entry"
            )
        class_names = [str(name) for name in class_names]

        state_dict = checkpoint.get("model_state_dict")
        if state_dict is None:
            raise ValueError(
                "Checkpoint is missing the 'model_state_dict' entry"
            )
        # Tolerate DataParallel checkpoints ("module." prefix).
        cleaned: Dict[str, Any] = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                key = key.replace("module.", "", 1)
            cleaned[key] = value

        # Build the model first so the state dict is copied
        # straight into it and can be freed immediately.
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(class_names))
        model.load_state_dict(cleaned, strict=True)
        model.to(self.device)
        model.eval()

        del checkpoint, state_dict, cleaned
        gc.collect()

        self.model = model
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.backend = "pytorch"
        self.loaded = True
        logger.info(
            "Model loaded (pytorch fallback): %s classes from %s on %s",
            self.num_classes,
            self.model_path,
            self.device,
        )


_service: Optional[ModelService] = None


def get_model_service() -> ModelService:
    """Singleton accessor so the API and tests share one service
    instance (one model per process)."""
    global _service
    if _service is None:
        _service = ModelService()
    return _service
