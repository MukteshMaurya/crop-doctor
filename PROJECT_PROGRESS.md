# PROJECT_PROGRESS — AI Plant Disease Detector

**Final updated:** Phase 5 preparation complete (local). Render/Vercel
deployment steps documented below require the repository owner's
account — they were **not** performed by the agent and are **not**
claimed as done.

---

## Phase 1: Inspection and Model Verification ✅

**Objective:** Inspect the project, locate the `.pth` model(s), determine
architecture, input size, normalization, class names/order, checkpoint
format, and document the inference pipeline before writing code.

### Findings

**Designated model: `unified_plant_resnet50.pth` (269.87 MB)**
(an identical copy exists in `crop doctor/` — SHA256
`CBCD4DC5BF7D5B138A70F914C6C02721EC2FF43EF15761BEBF18374ADEA903A1`)

- Checkpoint type: **full dict** (not a raw state_dict). Keys:
  `['num_classes', 'class_names', 'model_state_dict', 'optimizer_state_dict']`
  (the Adam optimizer state explains the ~270 MB size; only
  `model_state_dict` is used for inference).
- `num_classes = 17`. Architecture: `torchvision.models.resnet50(weights=None)`
  with `model.fc = nn.Linear(2048, 17)` — strict `load_state_dict`
  succeeds with 0 missing / 0 unexpected keys (320 tensors).
- **Class names (exact order, read from the checkpoint — authoritative):**
  apple, banana, blueberry, cherry, citrus, corn, grape, other_crops,
  peach, pepper_bell, potato, raspberry, soybean, squash, strawberry,
  tomato, wheat.
- **These are crop/plant TYPES, not diseases.** The model identifies
  which crop a leaf image belongs to. This was confirmed by direct
  checkpoint inspection and by the user, who chose to keep this model.
- Preprocessing (from `crop doctor/app1.py`, identical to standard
  ImageNet ResNet50 training): `Resize((224, 224))` → `ToTensor()` →
  `Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])`,
  RGB conversion.
- Confidence: `F.softmax(logits, dim=1)` (model trained with
  CrossEntropyLoss). Softmax value = model's predicted probability,
  **not** real-world accuracy.
- Loads with `torch.load(..., weights_only=True)` on torch 2.13.0+cpu
  (Python 3.12). Dummy-inference sanity check passed; output shape
  `[1, 17]`, softmax sums to 1.0.

**Other checkpoints found (NOT used — kept for reference):**
- `plant disease/mobilenetv2_plant.pth` (8.9 MB): MobileNetV2,
  38 PlantVillage **disease** classes (matches `disease_prescriptions.csv`
  and HF repo `Daksh159/plant-disease-mobilenetv2`).
- `plant disease/plant_disease_efficientnet.pth` (78 MB): EfficientNet V2 S,
  10 tomato-only disease classes (matches `class_labels.json`).

**Decision point raised with user:** the designated model classifies
crop type, not disease. User chose: **"Keep unified_plant_resnet50.pth
(17 crop types)"**. The app therefore honestly reports
"predicted crop / plant type" and states clearly that it does not
diagnose disease. No model was retrained, replaced, or downloaded.

**Environment:** Python 3.12.3 (`C:\Users\ABCD\AppData\Local\Programs\
Python\Python312\python.exe`) with torch 2.13.0+cpu, torchvision
0.28.0+cpu, fastapi, uvicorn, python-multipart, pillow, httpx;
pytest 9.1.1 installed by the agent. Git 2.55.0, Git LFS 3.7.1.
The `.venv` inside the working dir contains only pip (unused).

---

## Phase 2: Backend Development ✅

**Objective:** FastAPI + PyTorch backend around the existing checkpoint.

### Files created (`backend/`)

- `app/__init__.py` — package + version
- `app/model_loader.py` — `ModelService`: loads checkpoint once at
  startup via FastAPI lifespan; `weights_only=True`; extracts
  `class_names`/`model_state_dict`; builds ResNet50 with matching
  `fc`; strict load; `model.eval()`; graceful failure recorded in
  `load_error` (app still starts, `/health` reports degraded/503).
  Honors `MODEL_PATH` and `DEVICE` env vars (CPU default).
- `app/preprocessing.py` — exact training pipeline (224×224,
  ImageNet mean/std, RGB).
- `app/predictor.py` — `torch.inference_mode()`, softmax, top-3.
- `app/schemas.py` — Pydantic request/response models.
- `app/main.py` — endpoints + validation + CORS:
  - `GET /` — service info
  - `GET /health` — 200 healthy / **503 degraded** (model_loaded,
    device, architecture, model_path, num_classes, error)
  - `GET /api/classes` — the 17 classes from the checkpoint
  - `POST /api/predict` — multipart upload; validates Content-Length
    header and actual bytes (10 MB limit → 413), rejects empty (400),
    decodes with PIL `verify()` and checks real image format
    (JPEG/PNG/WEBP/BMP → 400 on failure), runs inference (500 on
    failure), returns `{success, predicted_class, confidence,
    top_predictions[3]}`.
  - CORS: origins from `FRONTEND_ORIGIN` (comma-separated). When
    unset, localhost dev origins only — **never** unrestricted `*`.
- `model_assets/unified_plant_resnet50.pth` — copy of the production
  checkpoint (hash-verified) so the Render service root
  (`backend/`) is self-contained.
- `model_assets/README.md`, `backend/README.md` — model + run docs
- `requirements.txt` — CPU PyTorch index + pinned ranges
- `render.yaml`, `runtime.txt` (`python-3.12`), `.gitignore`
- `tests/conftest.py`, `tests/test_api.py`,
  `tests/test_production_model.py`, `tests/__init__.py`

### Tests performed (actual results)

`pytest tests -v` → **19 passed** (16 API tests + 3 production-model
tests). API tests use a tiny generated ResNet50 checkpoint in the
production format; production tests load the real 270 MB model and
verify the 17 class names match the checkpoint exactly.

- Health OK / degraded (503 when model missing) ✅
- Class list matches checkpoint `class_names` ✅
- Valid JPEG + PNG prediction, response structure, sorted top-3 ✅
- Non-image upload → 400 ✅; corrupt bytes → 400 ✅; empty → 400 ✅
- Missing file field → 422 ✅; oversized → 413 (body and header) ✅
- Prediction unavailable → 503 when model not loaded ✅
- CORS: allowed origin echoed; unregistered origin gets no
  CORS header; `FRONTEND_ORIGIN` parsing ✅

**Bug found & fixed by tests:** `/health` initially returned HTTP 200
in the degraded case; now returns 503 so load balancers/readiness
probes fail correctly.

---

## Phase 3: Frontend Development ✅

**Objective:** Responsive vanilla JS interface connected to the backend.

### Files created (`frontend/`)

- `index.html` — title "AI Plant Disease Detector", how-it-works,
  drag-and-drop zone + browse button, image preview,
  "Detect Plant Disease" button, loading spinner, result card
  (predicted crop/plant type, confidence % with animated bar,
  top-3 list), "Analyze another image" reset, error alert,
  AI disclaimer. States clearly the model classifies crop/plant
  type (17 classes) and does not diagnose disease.
- `styles.css` — agricultural green theme, cards, responsive
  breakpoints (mobile/tablet/desktop), accessible focus states.
- `app.js` — client-side type/size validation (10 MB), `FormData`
  upload (no manual Content-Type), disables repeated submissions,
  handles HTTP 400/413/503/5xx, network/CORS failures, cold-start
  messaging, reset.
- `config.js` — single source for `API_BASE_URL` (currently
  `http://127.0.0.1:8000` for local dev; **must be set to the
  Render URL for production**).
- `vercel.json` — static security headers (nosniff, DENY framing).
- `README.md` — config + Vercel deployment guide.

**JS syntax validated** with `node --check` (app.js, config.js).

---

## Phase 4: Testing and Integration ✅

**Objective:** Verify model, API, frontend integration, invalid inputs,
error handling, and label consistency.

### Live-server verification (actual)

Backend started with `uvicorn app.main:app --host 127.0.0.1 --port 8000`
and `MODEL_PATH=model_assets/unified_plant_resnet50.pth`; frontend
served with `python -m http.server 5500`.

- `GET /health` → `{"status":"healthy","model_loaded":true,"device":"cpu",
  "architecture":"resnet50","num_classes":17}` ✅
- `GET /api/classes` → 17 classes in checkpoint order ✅
- `POST /api/predict` with real images:
  - `wheat.jpg` → **wheat** 0.569 (top-3: wheat, other_crops, tomato) ✅
  - banana photo → **banana** 0.999 ✅
  - soybean leaf photo → **soybean** 0.999 ✅
- Invalid uploads: JSON file → 400; empty file → 400; corrupt "jpeg"
  → 400; no file field → 422; 11 MB file → 413 ✅
- Cross-origin request simulation (Origin: `http://localhost:5500`,
  exact multipart pattern the browser uses): CORS header echoed,
  predictions returned, invalid → 400. **All E2E checks passed** ✅
- All static assets (index.html, styles.css, app.js, config.js)
  served with HTTP 200 ✅

**Consistency check:** API class list == checkpoint `class_names` ==
frontend-observed predictions (wheat/banana/soybean all in list). ✅

---

## Phase 5: Production Deployment — PREPARED (pending user action) 🟡

**Objective:** GitHub + Render (backend) + Vercel (frontend).

### Done locally

- `git init` in the working directory (`crop disease/`); Git LFS
  installed; `.gitattributes` tracks `*.pth` via LFS (both model
  copies deduplicate to a single LFS object, content hash
  `cbcd4dc5bf…`).
- Root `.gitignore` (venv, caches, secrets, `.env`, `.gradio/`, etc.).
- Local commit `3460b1f` — 27 files. **Nothing has been pushed.**
  Git identity already configured (Muktesh Maurya).

### Step 1 — Push to GitHub (user action)

```bash
cd "C:\Python files\AIMD\AIMD\expression detection\crop doctor\crop disease"
git remote add origin https://github.com/<USER>/<REPO>.git
git branch -M main
git push -u origin main        # pushes LFS objects automatically
```

Verify LFS (not a pointer) is in the repo after push:
`git lfs ls-files` should list both `.pth` paths. On Render, confirm
the deployed file is ~270 MB, not ~130 bytes (LFS pointer size).

### Step 2 — Deploy backend to Render (user action)

1. Render dashboard → **New + → Web Service → Git repository**.
2. **Root Directory:** `backend` (the repo root is the folder that
   contains `backend/` and `frontend/`).
3. **Build command:**
   `pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt`
4. **Start command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   (matches `render.yaml`; or connect the repo and Render reads
   `render.yaml` directly).
5. **Plan:** Starter (or Free) — CPU only, no GPU needed. Model is
   ~270 MB on disk; inference for one image takes <1 s on CPU.
6. **Environment variables:**
   - `FRONTEND_ORIGIN` = `https://YOUR-VERCEL-URL.vercel.app`
     (**required** — otherwise only localhost origins are allowed)
   - `MODEL_PATH` = `model_assets/unified_plant_resnet50.pth`
   - `DEVICE` = `cpu`
7. Deploy, then verify: `https://YOUR-RENDER-URL.onrender.com/health`
   must return `{"status":"healthy","model_loaded":true,...}`.
   First request after a cold start may take a few seconds while
   the 270 MB model loads.

### Step 3 — Deploy frontend to Vercel (user action)

1. Vercel dashboard → **Add New → Project → import the GitHub repo**.
2. **Root Directory:** `frontend`.
3. **Framework preset:** **Other** (plain HTML/CSS/JS).
4. **Build Command:** leave empty. **Output Directory:** `.`
5. **Before deploying:** edit `frontend/config.js` →
   `API_BASE_URL: "https://YOUR-RENDER-URL.onrender.com"`.
   (Never leave `http://127.0.0.1:8000` in production.)
6. Deploy. Open the live URL, upload a leaf image, and confirm the
   result comes from Render (DevTools → Network → `/api/predict`
   should show the Render domain, and the response contains the
   real model prediction).

### Alternative model-storage note

If LFS bandwidth/storage is a concern, the checkpoint's
`optimizer_state_dict` (~170 MB of the 270 MB) is unused at
inference; a lean `{"num_classes":…, "class_names":…,
"model_state_dict":…}` copy (~100 MB) can be generated on request.
The original checkpoint is preserved untouched either way.

---

## Problems discovered and fixes

1. **Model/type mismatch (Phase 1):** the designated `.pth` is a
   17-class crop-type classifier, while `disease_prescriptions.csv`
   and the Gradio app assumed 38 disease classes. Raised with the
   user instead of guessing; user chose to keep the crop-type model.
   UI/API/docs now describe the model honestly.
2. **`/health` returned 200 when the model failed to load** —
   caught by `test_health_fails_when_model_missing`; fixed to 503.
3. **Local `tests/` package shadowed** by an installed `tests`
   package in site-packages — fixed by adding `tests/__init__.py`.
4. **`.venv` in the working dir has no torch** — used the system
   Python 3.12 environment (torch 2.13.0+cpu) instead.

## Current deployment status

- Local backend: ✅ verified (uvicorn, real predictions, all error
  paths, CORS).
- Local frontend: ✅ verified (all assets serve; E2E API pattern
  passes).
- Render: 🟡 prepared (`render.yaml`, `runtime.txt`,
  `requirements.txt`, LFS) — **awaiting user's Render account**.
- Vercel: 🟡 prepared (`vercel.json`, config docs) — **awaiting
  user's Vercel account + `config.js` update**.
- GitHub: 🟡 local commit ready — **awaiting user's remote + push**.

## Remaining limitations

- The model identifies **crop/plant type, not disease** (user's
  choice of checkpoint). Disease diagnosis requires a disease-class
  checkpoint (e.g., the 38-class MobileNetV2 already on disk).
- Softmax confidence is the model's own probability estimate; no
  validation dataset with ground truth was available in the project
  to compute accuracy/precision/recall/F1, so no real-world
  accuracy is claimed.
- Single-image CPU inference only; no batching, no GPU.
- The 270 MB checkpoint includes unused optimizer state; see the
  lean-copy option in Phase 5 if LFS limits matter.
- Render/Vercel/GitHub push could not be executed by the agent
  (no credentials); those steps are documented for the user.
