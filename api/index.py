import io
import os
import csv
import urllib.request
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

torch.set_grad_enabled(False)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")

app = Flask(__name__, template_folder=TEMPLATE_DIR)
CORS(app)

DEVICE = torch.device("cpu")
MODEL_PATH = os.path.join(BASE_DIR, "unified_plant_resnet50.pth")
CSV_PATH = os.path.join(BASE_DIR, "prescriptions.csv")
MODEL_URL = "https://media.githubusercontent.com/media/MukteshMaurya/crop-doctor/main/unified_plant_resnet50.pth"

model = None
CLASS_NAMES = []
PRESCRIPTIONS_DB = {}

def load_prescriptions_csv():
    global PRESCRIPTIONS_DB
    if PRESCRIPTIONS_DB or not os.path.exists(CSV_PATH):
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
    except Exception as e:
        print(f"Error reading CSV: {e}")

def download_model_if_needed():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model weights...")
        os.makedirs(os.path.dirname(MODEL_PATH) if os.path.dirname(MODEL_PATH) else '.', exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

def get_model():
    global model, CLASS_NAMES
    if model is None:
        load_prescriptions_csv()
        download_model_if_needed()
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        
        if isinstance(checkpoint, dict) and 'class_names' in checkpoint:
            CLASS_NAMES = checkpoint['class_names']
        else:
            CLASS_NAMES = list(PRESCRIPTIONS_DB.keys())

        num_classes = len(CLASS_NAMES)
        loaded_model = models.resnet50(weights=None)
        loaded_model.fc = nn.Linear(loaded_model.fc.in_features, num_classes)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            loaded_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            loaded_model.load_state_dict(checkpoint)
            
        loaded_model.to(DEVICE)
        loaded_model.eval()
        model = loaded_model
    return model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.route('/', methods=['GET'])
def home():
    try:
        return render_template('index.html')
    except Exception as e:
        return jsonify({"status": "ready", "error": str(e)}), 200

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    try:
        current_model = get_model()
        
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        tensor = transform(image).unsqueeze(0).to(DEVICE)

        outputs = current_model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, 0)

        class_index = predicted_idx.item()
        raw_class_name = CLASS_NAMES[class_index]
        confidence_pct = round(confidence.item() * 100, 2)

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