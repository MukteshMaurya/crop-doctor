# AI Plant Disease Detector — Backend

FastAPI backend serving the pretrained
`unified_plant_resnet50` model (ResNet50, 17
crop/plant-type classes).

## Run locally

```bash
cd backend
pip install -r requirements-dev.txt   # includes test deps
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

The model is loaded once at startup from `MODEL_PATH`
(default: `model_assets/unified_plant_resnet50.onnx`,
the ONNX export of the trained weights).

## Inference backends

* **ONNX Runtime (default, production)** — `MODEL_PATH`
  ends in `.onnx`. CPU only. Never imports torch.
  Measured peak RSS for the whole server
  (uvicorn + FastAPI + onnxruntime + model):
  **~186 MB**. Fits a 512 MB Render instance.
* **PyTorch fallback** — `MODEL_PATH` ends in
  `.pth`/`.pt`/`.ckpt`. Loads with mmap, builds the
  model first, frees the checkpoint immediately.
  Measured peak RSS: **~466 MB** (bare process);
  with the server stack this needs a **1 GB+**
  instance. Both paths produce identical predictions
  (parity-tested, max difference ≤ 1.2e-07).

The ONNX file embeds the class names as metadata, so
`/api/classes` works without loading the PyTorch
checkpoint. An ONNX file without class metadata fails
 loudly (health → 503) rather than guessing.

## Environment variables

| Variable              | Default                                     | Purpose                              |
| --------------------- | ------------------------------------------- | ------------------------------------ |
| `FRONTEND_ORIGIN`     | localhost dev origins                       | Allowed CORS origin(s), comma-separated |
| `MODEL_PATH`          | `model_assets/unified_plant_resnet50.onnx`  | Model artifact to load (.onnx or .pth) |
| `DEVICE`              | `cpu`                                       | PyTorch fallback only: `cpu` or `cuda` |
| `ORT_INTRA_OP_THREADS`| `2`                                         | ONNX Runtime CPU threads             |

## Test

```bash
cd backend
pytest tests -v
```

API tests use a tiny generated ResNet50 checkpoint and
a tiny ONNX export of the same weights; production-model
tests run automatically when the real model files are
present in `model_assets/`.

## Deploy to Render

See `render.yaml` and `PROJECT_PROGRESS.md`.
CPU only; no GPU. The ONNX path is the one that fits
the 512 MB instance — keep `MODEL_PATH` pointing at the
`.onnx` file in production.
