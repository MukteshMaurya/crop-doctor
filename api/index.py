import io
import os
import csv
import urllib.request
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify, render_template

app = Flask(__name__, template_folder='../templates')

DEVICE = torch.device("cpu")

# Determine model path dynamically (use /tmp on Vercel to bypass bundle limits)
if os.getenv("VERCEL"):
    MODEL_PATH = "/tmp/unified_plant_resnet50.pth"
else:
    MODEL_PATH = "unified_plant_resnet50.pth"

MODEL_URL = "https://media.githubusercontent.com/media/MukteshMaurya/crop-doctor/main/unified_plant_resnet50.pth"

# Resolve CSV path relative to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if os.getenv("VERCEL") else "."
CSV_PATH = os.path.join(BASE_DIR, "prescriptions.csv") if os.getenv("VERCEL") else "prescriptions.csv"

model = None
CLASS_NAMES = []
PRESCRIPTIONS_DB = {}

# ---------------------------------------------------------
# 1. Load CSV Prescription Database into Memory
# ---------------------------------------------------------
def load_prescriptions_csv():
    global PRESCRIPTIONS_DB
    if PRESCRIPTIONS_DB:
        return

    if not os.path.exists(CSV_PATH):
        print(f"Warning: {CSV_PATH} not found.")
        return

    try:
        with open(CSV_PATH, mode='r', encoding='utf-8') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                raw_class = row["raw_class"].strip()
                PRESCRIPTIONS_DB[raw_class] = {
                    "status": row.get("status", "Unknown"),
                    "disease_name": row.get("disease_name", raw_class),
                    "description": row.get("description", "N/A"),
                    "organic_remedy": row.get("organic_remedy", "N/A"),
                    "chemical_treatment": row.get("chemical_treatment", "N/A"),
                    "prevention": row.get("prevention", "N/A")
                }
        print(f"Loaded prescriptions for {len(PRESCRIPTIONS_DB)} classes from CSV.")
    except Exception as e:
        print(f"Error reading CSV: {e}")

# ---------------------------------------------------------
# 2. Model Initialization & Dynamic Downloader
# ---------------------------------------------------------
def download_model_if_needed():
    if not os.path.exists(MODEL_PATH):
        print("Model file missing. Downloading from GitHub LFS...")
        os.makedirs(os.path.dirname(MODEL_PATH) if os.path.dirname(MODEL_PATH) else '.', exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded successfully.")

def load_model():
    global model, CLASS_NAMES
    if model is not None:
        return

    # 1. Load CSV database
    load_prescriptions_csv()

    # 2. Download model to /tmp on Vercel cold-start if missing
    download_model_if_needed()

    # 3. Load weights
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    
    if isinstance(checkpoint, dict) and 'class_names' in checkpoint:
        CLASS_NAMES = checkpoint['class_names']
    else:
        CLASS_NAMES = list(PRESCRIPTIONS_DB.keys())

    num_classes = len(CLASS_NAMES)

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(DEVICE)
    model.eval()

# Preprocessing Pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ---------------------------------------------------------
# 3. Flask API Endpoints
# ---------------------------------------------------------
@app.route('/', methods=['GET'])
def home():
    try:
        return render_template('index.html')
    except Exception:
        return jsonify({
            "status": "ready",
            "message": "Crop Doctor API operational. Send POST requests to /predict."
        }), 200

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "ready",
        "message": "33-Class Plant Disease API with CSV Prescription Lookup running."
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    try:
        load_model()
        
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            confidence, predicted_idx = torch.max(probabilities, 0)

        class_index = predicted_idx.item()
        raw_class_name = CLASS_NAMES[class_index]
        confidence_pct = round(confidence.item() * 100, 2)

        # Match prediction directly against CSV dictionary lookup
        prescription_data = PRESCRIPTIONS_DB.get(raw_class_name, {
            "status": "Unknown",
            "disease_name": raw_class_name,
            "description": "Details not found in CSV database.",
            "organic_remedy": "Consult a local agricultural expert.",
            "chemical_treatment": "N/A",
            "prevention": "N/A"
        })

        return jsonify({
            "success": True,
            "raw_class": raw_class_name,
            "confidence": f"{confidence_pct}%",
            "prescription": prescription_data
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)