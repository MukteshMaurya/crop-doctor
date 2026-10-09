"""One-time tool: convert the lean PyTorch checkpoint to ONNX
for low-memory CPU inference on Render.

The trained weights are NOT modified — only the format changes.
Class names and architecture are embedded as ONNX metadata so
the backend never needs to import torch at runtime.

Usage (run once, from the backend/ directory):
    python tools/export_onnx.py

Requires: torch, torchvision, onnx (see requirements-dev.txt).
"""

import json
import os
import sys

import torch
import torch.nn as nn
from torchvision import models

BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEAN_CKPT = os.path.join(
    BACKEND_ROOT, "model_assets", "unified_plant_resnet50_lean.pth"
)
ONNX_PATH = os.path.join(
    BACKEND_ROOT, "model_assets", "unified_plant_resnet50.onnx"
)


def main() -> int:
    if not os.path.exists(LEAN_CKPT):
        print(f"Lean checkpoint not found: {LEAN_CKPT}")
        print("Create it first (see model_assets/README.md).")
        return 1

    checkpoint = torch.load(LEAN_CKPT, map_location="cpu", weights_only=True)
    class_names = [str(name) for name in checkpoint["class_names"]]
    num_classes = int(checkpoint.get("num_classes", len(class_names)))

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy,
        ONNX_PATH,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    # Embed the class mapping and architecture as ONNX metadata
    # so the runtime can serve /api/classes without torch.
    import onnx

    onnx_model = onnx.load(ONNX_PATH)
    onnx_model.metadata_props.append(
        onnx.StringStringEntryProto(
            key="class_names", value=json.dumps(class_names)
        )
    )
    onnx_model.metadata_props.append(
        onnx.StringStringEntryProto(
            key="num_classes", value=str(num_classes)
        )
    )
    onnx_model.metadata_props.append(
        onnx.StringStringEntryProto(key="architecture", value="resnet50")
    )
    onnx_model.metadata_props.append(
        onnx.StringStringEntryProto(
            key="preprocessing",
            value="Resize((224,224)) bilinear; ToTensor; "
            "Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]); RGB",
        )
    )
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, ONNX_PATH)

    print(f"Exported: {ONNX_PATH}")
    print(f"Size: {os.path.getsize(ONNX_PATH) / 1e6:.1f} MB")
    print(f"Classes ({num_classes}): {', '.join(class_names)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
