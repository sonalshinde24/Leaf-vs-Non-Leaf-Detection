import os
import gdown
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Leaf vs Non-Leaf Detection",
    page_icon="🌿",
    layout="centered"
)

# --------------------------------------------------
# SIDEBAR CONTENT
# --------------------------------------------------
with st.sidebar:
    st.markdown("## 🌿 Leaf vs Non-Leaf Detection")

    st.markdown("---")

    st.markdown("### 📌 Problem Statement")
    st.write(
        "In many agricultural and computer vision applications, it is important "
        "to automatically determine whether an image contains a leaf or not. "
        "Manual identification is time-consuming and error-prone."
    )

    st.markdown("### 💡 Solution")
    st.write(
        "This application uses a **Deep Learning-based Convolutional Neural Network (CNN)** "
        "to classify images into **Leaf** or **Non-Leaf** categories with a confidence score."
    )

    st.markdown("### 🧭 How to Use")
    st.markdown(
        """
        1. Upload an image in JPG or PNG format  
        2. The AI model analyzes the image  
        3. View the prediction and confidence score  
        """
    )

    st.markdown("### 🛠️ Tech Stack")
    st.markdown(
        """
        - **Python**
        - **TensorFlow / Keras**
        - **Convolutional Neural Networks (CNN)**
        - **Streamlit**
        - **NumPy & PIL**
        """
    )

    st.markdown("---")
    st.markdown(
        "<small>Developed as an AI-based image classification system</small>",
        unsafe_allow_html=True
    )


# --------------------------------------------------
# DARK UI STYLING
# --------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top, #111827, #020617);
    color: #e5e7eb;
}

/* Main container */
.main .block-container {
    max-width: 720px;
    padding-top: 1.5rem;
    padding-bottom: 1.5rem;
}

/* Title */
h1 {
    text-align: center;
    font-size: 34px;
    font-weight: 700;
    color: #f9fafb;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 16px;
    color: #9ca3af;
    margin-bottom: 1.8rem;
}

/* Card style */
.card {
    background: rgba(17, 24, 39, 0.75);
    backdrop-filter: blur(10px);
    padding: 1.4rem;
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 12px 40px rgba(0,0,0,0.6);
    margin-bottom: 1.4rem;
}

/* Image box */
.image-box img {
    max-height: 280px;
    object-fit: contain;
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.08);
}

/* Prediction */
.prediction {
    font-size: 22px;
    font-weight: 600;
    margin-top: 1rem;
    color: #22c55e;
}

/* Confidence */
.confidence {
    font-size: 14px;
    color: #d1d5db;
}

/* Progress bar */
.stProgress > div > div {
    background: linear-gradient(90deg, #22c55e, #16a34a);
}

/* Footer */
.footer {
    text-align: center;
    font-size: 13px;
    color: #6b7280;
    margin-top: 2.5rem;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
MODEL_PATH = "leaf_vs_non_leaf_model.keras"
MODEL_ID = "1YXocLE0aXa0c_BWMD_HiL6ehWxMVTpCZ"

@st.cache_resource
def load_leaf_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            gdown.download(
                id=MODEL_ID,
                output=MODEL_PATH,
                quiet=False
            )
    return load_model(MODEL_PATH)
    
model = load_leaf_model()

IMG_SIZE = 224

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
def predict_image(img, model):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    p = model.predict(arr, verbose=0)[0][0]

    if p > 0.5:
        return "Non-Leaf", round(p * 100, 2)
    else:
        return "Leaf", round((1 - p) * 100, 2)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown("<h1>🌿 Leaf vs Non-Leaf Detection</h1>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>AI-powered image classifier with deep learning</div>",
    unsafe_allow_html=True
)

# --------------------------------------------------
# UPLOAD CARD
# --------------------------------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Upload an image (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)
st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# RESULT CARD
# --------------------------------------------------
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    label, confidence = predict_image(img, model)

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.markdown("<div class='image-box'>", unsafe_allow_html=True)
    st.image(img, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(f"<div class='prediction'>Prediction: {label}</div>", unsafe_allow_html=True)
    st.progress(int(confidence))
    st.markdown(
        f"<div class='confidence'>Confidence: <b>{confidence}%</b></div>",
        unsafe_allow_html=True
    )

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown(
    """
    <div class="footer">
        Built with Deep Learning & Streamlit<br>
        Leaf vs Non-Leaf Classification System
    </div>
    """,
    unsafe_allow_html=True
)




