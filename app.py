import csv
import os
import streamlit as st
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "asafe51/plantvillage-disease-classifier"

# Confidence levels
HIGH_CONFIDENCE = 80.0
MODERATE_CONFIDENCE = 60.0
LOW_CONFIDENCE = 40.0

# Margin = Top-1 confidence - Top-2 confidence
STRONG_MARGIN = 20.0
GOOD_MARGIN = 10.0
VERY_SMALL_MARGIN = 5.0


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Crop Doctor",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 AI Crop Doctor")
st.write(
    "Upload a plant leaf image and the AI will predict the most likely "
    "disease and show how reliable the prediction is."
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

    model = AutoModelForImageClassification.from_pretrained(
        MODEL_NAME
    )

    model.eval()

    return processor, model


with st.spinner("Loading AI model..."):
    processor, model = load_model()


# ============================================================
# LOAD DISEASE PRESCRIPTION DATABASE
# ============================================================

PRESCRIPTION_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "disease_prescriptions.csv"
)


@st.cache_data
def load_prescriptions():

    prescriptions = {}

    if not os.path.exists(PRESCRIPTION_FILE):
        return prescriptions

    with open(
        PRESCRIPTION_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            key = row["disease_label"].strip()

            prescriptions[key] = row

    return prescriptions


prescription_db = load_prescriptions()


# ============================================================
# GET LABEL
# ============================================================

def get_label(model, index):

    id2label = model.config.id2label

    # Most models use integer keys
    if index in id2label:
        return id2label[index]

    # Some models may use string keys
    if str(index) in id2label:
        return id2label[str(index)]

    return f"Class {index}"


# ============================================================
# CLEAN LABEL
# ============================================================

def clean_label(label):

    label = label.replace("_", " ")

    return label.strip()


# ============================================================
# ANALYZE CONFIDENCE + MARGIN
# ============================================================

def analyze_prediction(confidence, margin):

    # --------------------------------------------------------
    # HIGH CONFIDENCE
    # --------------------------------------------------------

    if confidence >= HIGH_CONFIDENCE and margin >= STRONG_MARGIN:

        return {
            "level": "HIGH",
            "icon": "🟢",
            "title": "High confidence",
            "message": (
                "The model has high confidence and the top prediction "
                "is clearly separated from the next prediction."
            ),
            "treatment_allowed": True
        }

    # --------------------------------------------------------
    # GOOD / MODERATE CONFIDENCE
    # --------------------------------------------------------

    elif confidence >= MODERATE_CONFIDENCE and margin >= GOOD_MARGIN:

        return {
            "level": "MODERATE",
            "icon": "🟡",
            "title": "Moderate confidence",
            "message": (
                "The model has a reasonably strong prediction, but "
                "verification is recommended before treatment."
            ),
            "treatment_allowed": False
        }

    # --------------------------------------------------------
    # LOW ABSOLUTE CONFIDENCE BUT STRONG SEPARATION
    # --------------------------------------------------------

    elif confidence >= LOW_CONFIDENCE and margin >= STRONG_MARGIN:

        return {
            "level": "LOW BUT WELL SEPARATED",
            "icon": "🟠",
            "title": "Low confidence, strong prediction separation",
            "message": (
                "The absolute confidence is low, but the top prediction "
                "is much stronger than the second prediction. "
                "The prediction is shown, but should be verified."
            ),
            "treatment_allowed": False
        }

    # --------------------------------------------------------
    # MODERATE CONFIDENCE BUT MODEL IS CONFUSED
    # --------------------------------------------------------

    elif confidence >= MODERATE_CONFIDENCE and margin < GOOD_MARGIN:

        return {
            "level": "UNCERTAIN",
            "icon": "🟠",
            "title": "Model is somewhat uncertain",
            "message": (
                "The confidence is reasonable, but the top predictions "
                "are relatively close. Additional verification is recommended."
            ),
            "treatment_allowed": False
        }

    # --------------------------------------------------------
    # LOW CONFIDENCE
    # --------------------------------------------------------

    elif confidence >= LOW_CONFIDENCE:

        return {
            "level": "LOW",
            "icon": "🟠",
            "title": "Low confidence",
            "message": (
                "The model's prediction is uncertain. "
                "The prediction is shown for reference, but it should "
                "not be treated as a confirmed diagnosis."
            ),
            "treatment_allowed": False
        }

    # --------------------------------------------------------
    # VERY LOW CONFIDENCE
    # --------------------------------------------------------

    else:

        return {
            "level": "VERY LOW",
            "icon": "🔴",
            "title": "Very low confidence",
            "message": (
                "The model is highly uncertain about this image. "
                "The prediction should not be considered reliable."
            ),
            "treatment_allowed": False
        }


# ============================================================
# PREDICT IMAGE
# ============================================================

def predict_image(image):

    # Convert image to RGB
    image = image.convert("RGB")

    # Process image
    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    # Model prediction
    with torch.no_grad():

        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )[0]

    # --------------------------------------------------------
    # TOP 5 PREDICTIONS
    # --------------------------------------------------------

    top_probabilities, top_indices = torch.topk(
        probabilities,
        k=min(5, len(probabilities))
    )

    results = []

    for probability, index in zip(
        top_probabilities,
        top_indices
    ):

        index = index.item()

        confidence = probability.item() * 100

        label = clean_label(
            get_label(model, index)
        )

        results.append(
            {
                "label": label,
                "confidence": confidence,
                "index": index
            }
        )

    # --------------------------------------------------------
    # TOP 1
    # --------------------------------------------------------

    best_prediction = results[0]

    best_label = best_prediction["label"]

    best_raw_label = get_label(
        model,
        best_prediction["index"]
    )

    best_confidence = best_prediction["confidence"]

    # --------------------------------------------------------
    # TOP 2
    # --------------------------------------------------------

    if len(results) >= 2:

        second_confidence = results[1]["confidence"]

    else:

        second_confidence = 0.0

    # --------------------------------------------------------
    # PREDICTION MARGIN
    # --------------------------------------------------------

    margin = best_confidence - second_confidence

    # --------------------------------------------------------
    # ANALYZE
    # --------------------------------------------------------

    analysis = analyze_prediction(
        best_confidence,
        margin
    )

    return {
        "prediction": best_label,
        "raw_label": best_raw_label,
        "confidence": best_confidence,
        "second_confidence": second_confidence,
        "margin": margin,
        "top5": results,
        "analysis": analysis
    }


# ============================================================
# DISPLAY TOP 5
# ============================================================

def display_top5(results):

    st.subheader("🔎 Top 5 Predictions")

    for i, result in enumerate(results):

        st.write(
            f"**{i + 1}. {result['label']}**"
        )

        st.progress(
            min(result["confidence"] / 100, 1.0)
        )

        st.caption(
            f"{result['confidence']:.2f}%"
        )


# ============================================================
# DISPLAY DISEASE PRESCRIPTION
# ============================================================

def display_prescription(result):

    raw_label = result["raw_label"]

    confidence = result["confidence"]

    margin = result["margin"]

    analysis = result["analysis"]

    # --------------------------------------------------------
    # FIND DISEASE
    # --------------------------------------------------------

    prescription = prescription_db.get(raw_label)

    st.divider()

    st.subheader("💊 Disease Management & Prescription")

    # --------------------------------------------------------
    # DATABASE NOT FOUND
    # --------------------------------------------------------

    if prescription is None:

        st.warning(
            "No verified treatment information is available "
            "for this prediction yet."
        )

        return

    # --------------------------------------------------------
    # HEALTHY PLANT
    # --------------------------------------------------------

    if prescription["condition"].lower() == "healthy":

        st.success(
            "🌱 The model's predicted class is HEALTHY."
        )

        st.write(
            prescription["management"]
        )

        st.info(
            "💊 No pesticide prescription is recommended "
            "for a healthy classification."
        )

        return

    # --------------------------------------------------------
    # LOW / MODERATE CONFIDENCE
    # --------------------------------------------------------

    if not analysis["treatment_allowed"]:

        st.warning(
            "🔒 Prescription locked"
        )

        st.write(
            f"""
            The model predicted:

            **{result['prediction']}**

            Confidence: **{confidence:.2f}%**

            Prediction margin: **{margin:.2f}%**
            """
        )

        st.info(
            "The prediction is not sufficiently reliable for "
            "an automatic treatment recommendation. "
            "Verify the disease before applying any pesticide."
        )

        return

    # --------------------------------------------------------
    # HIGH CONFIDENCE
    # --------------------------------------------------------

    st.success(
        f"✅ Treatment information for: "
        f"{result['prediction']}"
    )

    # --------------------------------------------------------
    # SYMPTOMS
    # --------------------------------------------------------

    st.markdown("### 🔍 Typical symptoms")

    st.write(
        prescription["symptoms"]
    )

    # --------------------------------------------------------
    # MANAGEMENT
    # --------------------------------------------------------

    st.markdown("### 🌿 Recommended management")

    st.write(
        prescription["management"]
    )

    # --------------------------------------------------------
    # PRESCRIPTION
    # --------------------------------------------------------

    st.markdown("### 💊 Treatment / prescription")

    st.warning(
        prescription["prescription"]
    )

    # --------------------------------------------------------
    # SOURCE
    # --------------------------------------------------------

    st.caption(
        f"Information source: {prescription['source']}"
    )

    # --------------------------------------------------------
    # SAFETY
    # --------------------------------------------------------

    st.error(
        "⚠️ Always verify the diagnosis and use only products "
        "registered for the crop and disease in your region. "
        "Follow the pesticide label, dose, PPE requirements, "
        "and pre-harvest interval. This AI should not replace "
        "local agricultural expert advice."
    )


# ============================================================
# DISPLAY FINAL RESULT
# ============================================================

def display_final_result(result):

    analysis = result["analysis"]

    st.divider()

    st.subheader("🌿 AI Prediction")

    # --------------------------------------------------------
    # DISEASE
    # --------------------------------------------------------

    st.markdown(
        f"### {result['prediction']}"
    )

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Top-1 Confidence",
            f"{result['confidence']:.2f}%"
        )

    with col2:

        st.metric(
            "Top-1 vs Top-2 Margin",
            f"{result['margin']:.2f}%"
        )

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    st.markdown(
        f"## {analysis['icon']} {analysis['title']}"
    )

    st.info(
        analysis["message"]
    )

    # --------------------------------------------------------
    # EXPLANATION
    # --------------------------------------------------------

    st.write(
        f"""
        **Top prediction:** {result['prediction']}

        **Top-1 confidence:** {result['confidence']:.2f}%

        **Second prediction confidence:**
        {result['second_confidence']:.2f}%

        **Prediction margin:** {result['margin']:.2f}%
        """
    )

    # --------------------------------------------------------
    # TREATMENT STATUS
    # --------------------------------------------------------

    if analysis["treatment_allowed"]:

        st.success(
            "The prediction has sufficient confidence for "
            "the next disease-information stage."
        )

    else:

        st.warning(
            "⚠️ Do not automatically apply a treatment based "
            "only on this prediction. Verify the disease first."
        )


# ============================================================
# IMAGE INPUT
# ============================================================

st.subheader("📷 Select Image Source")

input_method = st.radio(
    "Choose image source:",
    [
        "📷 Take Photo",
        "🖼️ Upload Image"
    ],
    horizontal=True
)

image_source = None

# ------------------------------------------------------------
# CAMERA INPUT
# ------------------------------------------------------------

if input_method == "📷 Take Photo":

    st.write(
        "Take a clear photo of the plant leaf."
    )

    camera_photo = st.camera_input(
        "Capture Plant Image"
    )

    if camera_photo is not None:
        image_source = camera_photo

# ------------------------------------------------------------
# FILE UPLOAD
# ------------------------------------------------------------

else:

    uploaded_file = st.file_uploader(
        "Upload a plant image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ]
    )

    if uploaded_file is not None:
        image_source = uploaded_file


# ============================================================
# MAIN APPLICATION
# ============================================================

if image_source is not None:

    image = Image.open(image_source).convert("RGB")

    st.subheader("📷 Uploaded Image")

    st.image(
        image,
        use_container_width=True
    )

    # --------------------------------------------------------
    # PREDICT BUTTON
    # --------------------------------------------------------

    if st.button(
        "🔍 Analyze Plant",
        use_container_width=True
    ):

        with st.spinner(
            "Analyzing plant leaf..."
        ):

            result = predict_image(image)

        # ----------------------------------------------------
        # DISPLAY FINAL RESULT
        # ----------------------------------------------------

        display_final_result(result)

        display_top5(
            result["top5"]
        )

        display_prescription(
            result
        )

        # ----------------------------------------------------
        # DEBUG / MODEL INFORMATION
        # ----------------------------------------------------

        with st.expander(
            "🧠 Model analysis details"
        ):

            st.write(
                f"Model: `{MODEL_NAME}`"
            )

            st.write(
                f"Top-1: `{result['confidence']:.2f}%`"
            )

            st.write(
                f"Top-2: `{result['second_confidence']:.2f}%`"
            )

            st.write(
                f"Margin: `{result['margin']:.2f}%`"
            )

            st.write(
                f"Confidence level: `{result['analysis']['level']}`"
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "⚠️ AI predictions are for decision support and should be "
    "verified before applying agricultural treatments."
)