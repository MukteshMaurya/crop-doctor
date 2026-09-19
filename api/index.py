import io
import os
import csv
import numpy as np
import onnxruntime as ort
from PIL import Image
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")

app = Flask(__name__, template_folder=TEMPLATE_DIR)
CORS(app)

MODEL_PATH = os.path.join(BASE_DIR, "model.onnx")
CSV_PATH = os.path.join(BASE_DIR, "prescriptions.csv")

session = None
PRESCRIPTIONS_DB = {}
CLASS_NAMES = []

import torch

def load_prescriptions_csv():
    global PRESCRIPTIONS_DB, CLASS_NAMES
    if PRESCRIPTIONS_DB or not os.path.exists(CSV_PATH):
        return
    try:
        # 1. First attempt to pull true class names directly from PyTorch checkpoint if present
        PTH_PATH = os.path.join(BASE_DIR, "unified_plant_resnet50.pth")
        if os.path.exists(PTH_PATH):
            try:
                checkpoint = torch.load(PTH_PATH, map_location="cpu")
                if isinstance(checkpoint, dict) and 'class_names' in checkpoint:
                    CLASS_NAMES = checkpoint['class_names']
            except Exception as e:
                print(f"Could not load class_names from pth: {e}")

        # 2. Load CSV prescription mappings
        with open(CSV_PATH, mode='r', encoding='utf-8') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            csv_classes = []
            for row in csv_reader:
                raw_class = row["raw_class"].strip()
                csv_classes.append(raw_class)
                PRESCRIPTIONS_DB[raw_class] = {
                    "status": row.get("status", "Unknown"),
                    "disease_name": row.get("disease_name", raw_class),
                    "description": row.get("description", "N/A"),
                    "organic_remedy": row.get("organic_remedy", "N/A"),
                    "chemical_treatment": row.get("chemical_treatment", "N/A"),
                    "prevention": row.get("prevention", "N/A")
                }
        
        # Fallback to CSV class order if class_names wasn't stored in checkpoint
        if not CLASS_NAMES:
            CLASS_NAMES = csv_classes

    except Exception as e:
        print(f"Error reading CSV/PTH: {e}")

def get_onnx_session():
    global session
    if session is None:
        load_prescriptions_csv()
        session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    return session

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))
    
    # Convert image to float array and normalize (ImageNet stats)
    arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)  # HWC to CHW
    arr = np.expand_dims(arr, axis=0)  # Add batch dim
    return arr

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

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
        ort_session = get_onnx_session()
        input_tensor = preprocess_image(file.read())
        
        input_name = ort_session.get_inputs()[0].name
        outputs = ort_session.run(None, {input_name: input_tensor})[0]
        
        probs = softmax(outputs[0])
        predicted_idx = int(np.argmax(probs))
        confidence_pct = round(float(probs[predicted_idx]) * 100, 2)

        raw_class_name = CLASS_NAMES[predicted_idx] if predicted_idx < len(CLASS_NAMES) else f"Class_{predicted_idx}"
        
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