"""
Utility functions for Streamlit application.

Handles:
- Model loading
- Preprocessor loading
"""

from pathlib import Path
import joblib
import streamlit as st


# ===============================
# PATH CONFIGURATION
# ===============================

# Project Root Directory
BASE_DIR = Path(__file__).resolve().parent.parent


# Model paths
MODEL_PATH = (
    BASE_DIR
    / "models"
    / "gradient_boosting_model.pkl"
)


PREPROCESSOR_PATH = (
    BASE_DIR
    / "models"
    / "preprocessor.pkl"
)


# ===============================
# LOAD MODEL
# ===============================

@st.cache_resource
def load_model():
    """
    Load trained Gradient Boosting model.

    cache_resource prevents reloading
    the model after every Streamlit refresh.
    """

    if not MODEL_PATH.exists():
        st.error(
            f"Model file not found:\n{MODEL_PATH}"
        )
        st.stop()

    model = joblib.load(
        MODEL_PATH
    )

    return model



# ===============================
# LOAD PREPROCESSOR
# ===============================

@st.cache_resource
def load_preprocessor():
    """
    Load preprocessing pipeline.
    """

    if not PREPROCESSOR_PATH.exists():
        st.error(
            f"Preprocessor file not found:\n{PREPROCESSOR_PATH}"
        )
        st.stop()


    preprocessor = joblib.load(
        PREPROCESSOR_PATH
    )

    return preprocessor