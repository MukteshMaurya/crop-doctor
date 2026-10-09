# Model assets

All three artifacts contain the **same trained ResNet50
weights** (17 crop/plant-type classes) in different
formats. The trained model is reused as-is — it is never
retrained or modified.

| File | Size | Purpose |
| ---- | ---- | ------- |
| `unified_plant_resnet50.onnx` | 94.1 MB | **Production model** (default `MODEL_PATH`). ONNX Runtime, CPU only, ~186 MB peak server RSS. Class names and architecture are embedded as ONNX metadata. |
| `unified_plant_resnet50.pth` | 283 MB | Original checkpoint (includes the Adam `optimizer_state_dict`, which is never used at inference). PyTorch fallback — loads with mmap, ~466 MB peak RSS; requires a 1 GB+ instance. |
| `unified_plant_resnet50_lean.pth` | 94.5 MB | Same weights without the optimizer state. Lighter PyTorch fallback (smaller download), still ~460 MB peak RSS at inference time. |

- **Classes (17, in order):** apple, banana, blueberry,
  cherry, citrus, corn, grape, other_crops, peach,
  pepper_bell, potato, raspberry, soybean, squash,
  strawberry, tomato, wheat.
- **Preprocessing:** Resize((224, 224)) bilinear →
  ToTensor → Normalize(mean=[0.485, 0.456, 0.406],
  std=[0.229, 0.224, 0.225]), RGB.
- **Checkpoint provenance:** SHA256
  `CBCD4DC5BF7D5B138A70F914C6C02721EC2FF43EF15761BEBF18374ADEA903A1`
  (original `crop doctor/unified_plant_resnet50.pth`).
- **ONNX export:** regenerate with
  `python tools/export_onnx.py` (requires torch + onnx;
  see `requirements-dev.txt`). Prediction parity between
  the ONNX and PyTorch paths is verified by
  `tests/test_onnx.py` (max softmax difference ≤ 1.2e-07).
- All files are tracked with Git LFS (see `.gitattributes`).
