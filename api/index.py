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

# Standard 33 Plant Village Class Order
CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Cherry___Powdery_mildew",
    "Cherry___healthy",
    "Corn___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight",
    "Corn___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___Healthy",
    "Tomato___healthy",
    "Strawberry___Leaf_spot"
]

session = None
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
        print(f"Loaded {len(PRESCRIPTIONS_DB)} classes into database dictionary.")
    except Exception as e:
        print(f"Error reading CSV: {e}")

def get_onnx_session():
    global session
    if session is None:
        load_prescriptions_csv()
        session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    return session

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))
    
    arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)
    arr = np.expand_dims(arr, axis=0)
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

        # Get class name by index from array
        if predicted_idx < len(CLASS_NAMES):
            raw_class_name = CLASS_NAMES[predicted_idx]
        else:
            raw_class_name = f"Unknown_Class_{predicted_idx}"

        # Fetch prescription details from dictionary using raw_class key
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