import io
import os
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

import gdown


@st.cache_resource
def load_disease_model(model_path: str = "tomato_disease_new_model.keras"):
    """
    Load and cache the tomato disease classification model.
    """
    ensure_disease_model_present(model_path)
    return load_model(model_path)


# Global model handle for convenience
model = load_disease_model()


# Class names (must match training order)
CLASS_NAMES: List[str] = [
    "Early_blight",
    "Healthy",
    "Late_blight",
    "Leaf Miner",
    "Magnesium Deficiency",
    "Nitrogen Deficiency",
    "Pottassium Deficiency",
    "Spotted Wilt Virus",
]


def _preprocess_image(
    image: Image.Image,
    target_size: Tuple[int, int] = (380, 380),
) -> np.ndarray:
    """
    Resize to 380x380 RGB and apply EfficientNet preprocessing.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize(target_size)
    array = np.array(image).astype("float32")
    array = np.expand_dims(array, axis=0)
    array = preprocess_input(array)
    return array


def predict_disease_from_pil(image: Image.Image) -> Dict:
    """
    Run disease prediction on a PIL image and return a structured result.
    """
    input_array = _preprocess_image(image)
    preds = model.predict(input_array, verbose=0)[0]

    if preds.ndim != 1:
        preds = preds.squeeze()

    best_idx = int(np.argmax(preds))
    best_conf = float(preds[best_idx])
    best_label = CLASS_NAMES[best_idx] if best_idx < len(CLASS_NAMES) else f"class_{best_idx}"

    top_indices = preds.argsort()[-3:][::-1]
    top3: List[Tuple[str, float]] = []
    for idx in top_indices:
        label = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"class_{int(idx)}"
        top3.append((label, float(preds[idx])))

    return {
        "predicted_label": best_label,
        "confidence": best_conf,
        "top3": top3,
    }


def analyze_uploaded_image(file) -> Dict:
    """
    Convenience wrapper for Streamlit file_uploader output.
    Compatible with Streamlit's UploadedFile and generic file-like objects.
    """
    if hasattr(file, "getvalue"):
        image_bytes = file.getvalue()
    else:
        try:
            file.seek(0)
        except Exception:
            pass
        image_bytes = file.read()

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return predict_disease_from_pil(image)


def _get_secret(name: str):
    """
    Read config from Streamlit secrets (preferred) or environment variables (fallback).
    """
    try:
        if name in st.secrets:
            return st.secrets.get(name)
    except Exception:
        pass
    return os.getenv(name)


def ensure_disease_model_present(model_path: str):
    """
    Ensure the Keras model file exists locally.
    If missing (e.g., on Streamlit Cloud), download it from Google Drive using MODEL_FILE_ID.
    """
    if os.path.exists(model_path):
        return

    file_id = _get_secret("MODEL_FILE_ID")
    if not file_id:
        st.error(
            "Disease model file is missing.\n\n"
            "Set `MODEL_FILE_ID` in Streamlit Cloud Secrets to auto-download the model."
        )
        st.stop()

    # Download from Google Drive to the expected filename
    try:
        url = f"https://drive.google.com/uc?id={file_id}"
        with st.spinner("Downloading disease model (first-time setup)..."):
            gdown.download(url, model_path, quiet=False)
    except Exception as e:
        st.error(f"Failed to download disease model from Google Drive: {e}")
        st.stop()

    if not os.path.exists(model_path):
        st.error("Model download did not create the expected file. Please re-check Drive permissions/link.")
        st.stop()

