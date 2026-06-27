"""
Brain Tumor MRI Classification — Streamlit Web App
Upload an MRI image and get a real-time tumor type prediction
using the best-performing model (MobileNet, fine-tuned).
"""

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import time

# ──────────────────────────────────────────────
# PAGE CONFIGURATION
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────
MODEL_PATH = os.path.join("D:\AI&ML\Brain Tumor\models", "mobilenet_finetuned.h5")
IMG_SIZE   = (224, 224)
CLASSES    = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

CLASS_INFO = {
    'glioma': {
        'display_name': 'Glioma',
        'description': 'A tumor that originates in the glial cells of the brain or spine. '
                        'Gliomas can vary widely in aggressiveness.',
        'color': '#FF6B6B'
    },
    'meningioma': {
        'display_name': 'Meningioma',
        'description': 'A tumor that forms in the meninges, the membranes surrounding '
                        'the brain and spinal cord. Often slow-growing and benign.',
        'color': '#4D96FF'
    },
    'no_tumor': {
        'display_name': 'No Tumor',
        'description': 'No tumor detected in this MRI scan. The brain tissue '
                        'appears within normal patterns.',
        'color': '#6BCB77'
    },
    'pituitary': {
        'display_name': 'Pituitary Tumor',
        'description': 'A tumor that develops in the pituitary gland, which can '
                        'affect hormone regulation throughout the body.',
        'color': '#FFD93D'
    },
}

# ──────────────────────────────────────────────
# MODEL LOADING (cached so it loads only once)
# ──────────────────────────────────────────────
@st.cache_resource
def load_classification_model():
    """Load the trained MobileNet model. Cached so it's only loaded once per session."""
    model = load_model(MODEL_PATH)
    return model


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess a PIL image for MobileNet input:
    - Convert to RGB
    - Resize to 224x224
    - Normalize to [0, 1]
    - Add batch dimension
    """
    image = image.convert('RGB')
    image = image.resize(IMG_SIZE)
    img_array = np.array(image).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict(model, image: Image.Image):
    """Run prediction and return class probabilities."""
    processed = preprocess_image(image)
    predictions = model.predict(processed, verbose=0)[0]
    return predictions


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 About This App")
    st.markdown("""
    This application uses a **deep learning model** (MobileNet,
    fine-tuned via transfer learning) to classify brain MRI scans
    into one of four categories:

    - **Glioma**
    - **Meningioma**
    - **No Tumor**
    - **Pituitary Tumor**
    """)

    st.divider()

    st.subheader("📊 Model Performance")
    col1, col2 = st.columns(2)
    col1.metric("Test Accuracy", "91.46%")
    col2.metric("F1-Score", "91.36%")

    st.divider()

    st.subheader("⚠️ Disclaimer")
    st.warning(
        "This tool is built for **educational and demonstration purposes only**. "
        "It is **not** a certified medical diagnostic tool. Always consult a "
        "qualified radiologist or medical professional for an actual diagnosis."
    )

# ──────────────────────────────────────────────
# MAIN PAGE HEADER
# ──────────────────────────────────────────────
st.title("🧠 Brain Tumor MRI Classification")
st.markdown(
    "Upload a brain MRI scan below and the model will predict the **tumor type** "
    "along with a **confidence score** for each possible category."
)
st.divider()

# ──────────────────────────────────────────────
# LOAD MODEL
# ──────────────────────────────────────────────
try:
    with st.spinner("Loading model..."):
        model = load_classification_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(
        f"❌ Could not load the model from `{MODEL_PATH}`. "
        f"Please make sure the model file exists in the `models/` folder.\n\n"
        f"Error details: {e}"
    )

# ──────────────────────────────────────────────
# MAIN LAYOUT — Upload & Results
# ──────────────────────────────────────────────
if model_loaded:
    col_upload, col_result = st.columns([1, 1.3], gap="large")

    with col_upload:
        st.subheader("📤 Upload MRI Image")
        uploaded_file = st.file_uploader(
            "Choose a brain MRI image (JPG, JPEG, or PNG)",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a single MRI scan image for classification"
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded MRI Scan", use_container_width=True)

    with col_result:
        st.subheader("🔍 Prediction Results")

        if uploaded_file is None:
            st.info("👈 Upload an MRI image to see the prediction results here.")
        else:
            with st.spinner("Analyzing image..."):
                start_time = time.time()
                predictions = predict(model, image)
                inference_time = (time.time() - start_time) * 1000

            predicted_idx   = int(np.argmax(predictions))
            predicted_class = CLASSES[predicted_idx]
            confidence      = float(predictions[predicted_idx])
            class_meta      = CLASS_INFO[predicted_class]

            # ── Main Prediction Display ──
            if predicted_class == 'no_tumor':
                st.success(f"### ✅ Prediction: {class_meta['display_name']}")
            else:
                st.warning(f"### ⚠️ Prediction: {class_meta['display_name']}")

            st.metric("Confidence Score", f"{confidence:.2%}")
            st.caption(f"Inference time: {inference_time:.1f} ms")

            st.markdown(f"**About this result:** {class_meta['description']}")

            st.divider()

            # ── Confidence Breakdown for All Classes ──
            st.subheader("📊 Confidence Breakdown — All Classes")

            # Sort classes by confidence descending for display
            sorted_indices = np.argsort(predictions)[::-1]

            for idx in sorted_indices:
                cls_name = CLASSES[idx]
                cls_meta = CLASS_INFO[cls_name]
                prob     = float(predictions[idx])

                bar_col, pct_col = st.columns([5, 1])
                with bar_col:
                    st.markdown(f"**{cls_meta['display_name']}**")
                    st.progress(prob)
                with pct_col:
                    st.markdown(f"<br>**{prob:.1%}**", unsafe_allow_html=True)

            # ── Confidence Warning for Low-Confidence Predictions ──
            if confidence < 0.60:
                st.divider()
                st.warning(
                    "⚠️ **Low confidence prediction.** The model is not highly "
                    "certain about this result. Consider uploading a clearer "
                    "image or consulting a medical professional."
                )

# ──────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────
st.divider()
st.caption(
    "Built with TensorFlow/Keras & Streamlit | "
    "Model: MobileNet (Transfer Learning) | "
    "For educational purposes only"
)