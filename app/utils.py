"""
Utility functions for the Streamlit application.
"""

from pathlib import Path
import joblib

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "gradient_boosting_model.pkl"
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.pkl"


def load_model():
    """Load trained model."""
    return joblib.load(MODEL_PATH)


def load_preprocessor():
    """Load preprocessor."""
    return joblib.load(PREPROCESSOR_PATH)