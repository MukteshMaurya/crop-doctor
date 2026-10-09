# PROJECT_PROGRESS — AI Plant Disease Detector

## Phase 1: Inspection and Model Verification

**Objective:** Inspect the existing project, locate and analyze the `.pth` model(s), determine architecture, input size, normalization, class names/order, and checkpoint format. Document the inference pipeline before writing any application code.

### Files inspected

- `crop disease/unified_plant_resnet50.pth` (269.87 MB) — user-designated model
- `crop doctor/unified_plant_resnet50.pth` — identical copy (SHA256 `CBCD4DC5...A1`, matches the `crop disease` copy)
- `crop doctor/app1.py` — original Gradio inference app for the ResNet50 checkpoint
- `crop doctor/disease_prescriptions.csv` — 38 PlantVillage disease-class reference rows
- `plant disease/app1.py` — MobileNetV2 38-class disease-classifier script (HF repo `Daksh159/plant-disease-mobilenetv2`)
- `plant disease/main.py` — EfficientNet FastAPI prototype (YOLO gatekeeper, 0.85 confidence threshold)
- `plant disease/train_tomato.py` — training script for the tomato EfficientNet
- `plant disease/class_labels.json` — 10 tomato class mapping
- `plant disease/mobilenetv2_plant.pth` (8.91 MB), `plant disease/plant_disease_efficientnet.pth` (77.88 MB), `plant disease/yolov8n.pt` (6.25 MB)
- `plant disease/temp_data/metadata.json` — sample metadata (tomato late blight)

### Findings

**A. `unified_plant_resnet50.pth` (the file the user pointed to)**
- Checkpoint type: full dict (NOT a raw state_dict). Keys: `['num_classes', 'class_names', 'model_state_dict', 'optimizer_state_dict']`
- `num_classes = 17`; the 270 MB size is explained by the stored Adam `optimizer_state_dict`
- Architecture: `torchvision.models.resnet50(weights=None)` with `model.fc = nn.Linear(2048, 17)` — strict `load_state_dict` succeeds with zero missing/unexpected keys (320 tensors)
- **Class names (exact order, extracted from the checkpoint itself — authoritative):**
  0. apple, 1. banana, 2. blueberry, 3. cherry, 4. citrus, 5. corn, 6. grape, 7. other_crops, 8. peach, 9. pepper_bell, 10. potato, 11. raspberry, 12. soybean, 13. squash, 14. strawberry, 15. tomato, 16. wheat
- These are **crop types / plant species, NOT diseases**. The model identifies which crop a leaf image belongs to.
- Preprocessing (from `crop doctor/app1.py`, matches standard ImageNet ResNet50 pipeline): `Resize((224,224))` → `ToTensor()` → `Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])`, RGB conversion
- Confidence: `F.softmax(outputs, dim=1)` over raw logits (CrossEntropyLoss-trained model)
- Loads successfully with `torch.load(..., weights_only=True)` on torch 2.13.0+cpu (Python 3.12)
- Dummy-inference sanity check passed: output shape `[1, 17]`, softmax sums to 1.0
- Live sample tests: wheat.jpg → wheat (0.57), wheatte(1).jpg → wheat (1.00), images.jpg → soybean (0.999), kzMYKaqQhtY2EzL7GABdah.jpg → banana (0.999)

**B. `plant disease/mobilenetv2_plant.pth` (8.91 MB) — actual disease model**
- Raw `state_dict` (OrderedDict, 314 tensors), loads with `weights_only=True`
- Architecture: `models.mobilenet_v2(weights=None)` with `classifier[1] = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, 38))`
- **38 PlantVillage disease classes** (standard order: Apple scab/black rot/cedar rust/healthy, Blueberry healthy, Cherry powdery mildew/healthy, Corn cercospora/common rust/NLB/healthy, Grape black rot/esca/leaf blight/healthy, Orange huanglongbing, Peach bacterial spot/healthy, Pepper bell bacterial spot/healthy, Potato early/late blight/healthy, Raspberry healthy, Soybean healthy, Squash powdery mildew, Strawberry leaf scorch/healthy, Tomato bacterial spot/early blight/late blight/leaf mold/septoria/spider mites/target spot/TYLCV/TMV/healthy)
- Same 224×224 ImageNet preprocessing; softmax confidence
- Verified loads + runs (e.g. images.jpg → idx4 Blueberry healthy 0.9999)

**C. `plant disease/plant_disease_efficientnet.pth` (77.88 MB)**
- Raw state_dict (782 tensors); EfficientNet V2 S with `classifier[1] = nn.Linear(1280, 10)`
- **10 tomato-only classes** (matches `class_labels.json`): Tomato Bacterial spot, Early blight, Late blight, Leaf Mold, Septoria leaf spot, Spider mites Two-spotted, Target Spot, TYLCV, TMV, healthy
- Trained by `train_tomato.py` on a tomato-only subset (100 images/class, 5 epochs, Adam lr=0.001)

**Environment**
- Python 3.12.3 at `C:\Users\ABCD\AppData\Local\Programs\Python\Python312\python.exe` with `torch 2.13.0+cpu`, `torchvision 0.28.0+cpu`
- Python 3.14 at `C:\Python314\python.exe` with `torch 2.14.0+cpu`
- The `.venv` inside `crop disease/` contains only pip (no torch) — not usable as-is
- No git repo initialized in the working directory (checked)

### Critical issue discovered

The user's stated goal is **plant disease detection**, and `disease_prescriptions.csv` lists 38 disease classes — but the designated model `unified_plant_resnet50.pth` is a **17-class crop-type classifier** (apple, banana, wheat, ...). Its outputs are crop names, not diseases. It is healthy and loads correctly, but it cannot detect disease. Two genuine disease-classifier checkpoints already exist on disk (B: 38 diseases, C: 10 tomato diseases). Per the project instructions ("If multiple models exist, identify the likely production model and ask"), this decision requires user confirmation before implementation.

### Decision needed from user

Which checkpoint should power the app?
- **A** — `unified_plant_resnet50.pth` as-is: app honestly reports "predicted crop type" (17 classes), not disease
- **B** — `mobilenetv2_plant.pth`: true 38-class PlantVillage disease detection, matches `disease_prescriptions.csv`, small and CPU-friendly (recommended for the stated project goal)
- **C** — `plant_disease_efficientnet.pth`: tomato-only disease detection (10 classes)
- **D** — user supplies a different `.pth` file

### Commands executed

- Checkpoint inspection via ad-hoc scripts: `torch.load(..., weights_only=True)`, key/shape printing, strict state_dict load, dummy inference, softmax verification
- SHA256 comparison of the two ResNet50 copies (identical)
- Sample-image inference across all three models

### Current deployment status

None — Phase 1 inspection only. No application code written yet.

### Next steps

1. Resolve the model-selection question above.
2. Phase 2: build FastAPI backend around the confirmed model with exact preprocessing.
3. Phase 3: build the frontend.
4. Phase 4: integration testing.
5. Phase 5: deployment preparation (Render + Vercel).
