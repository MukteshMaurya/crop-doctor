# AI Plant Disease Detector — Backend

FastAPI backend serving the pretrained
`unified_plant_resnet50.pth` checkpoint (ResNet50, 17
crop/plant-type classes).

## Run locally

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

The model is loaded once at startup from `MODEL_PATH`
(default: `model_assets/unified_plant_resnet50.pth`).

## Environment variables

| Variable          | Default                            | Purpose                          |
| ----------------- | ---------------------------------- | -------------------------------- |
| `FRONTEND_ORIGIN` | localhost dev origins              | Allowed CORS origin(s), comma-separated |
| `MODEL_PATH`      | `model_assets/unified_plant_resnet50.pth` | Checkpoint location |
| `DEVICE`          | `cpu`                              | `cpu` or `cuda` (cuda only if available) |

## Test

```bash
cd backend
pytest tests -v
```

Tests run against a tiny generated checkpoint for speed;
production-model tests run automatically when the real
checkpoint is present.

## Deploy to Render

See `render.yaml` and `PROJECT_PROGRESS.md` (Phase 5).
The service needs no GPU; a standard Render instance is
sufficient.
