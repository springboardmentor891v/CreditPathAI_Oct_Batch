# streamlit_app/utils.py

import streamlit as st  # <-- Add this import
import pandas as pd
import joblib
import os
from typing import Dict, Any, Optional

# --- Model Loading with Caching ---
@st.cache_resource  # <-- Key Streamlit decorator!
def load_model(model_name: str):
    """
    Load a trained model pipeline from disk.
    Uses @st.cache_resource to prevent reloading the model from disk on every app rerun.
    """
    # ... (rest of your function is perfect, no changes needed inside)
    try:
        filename = f"{model_name.lower().replace(' ', '_')}_pipeline.joblib"
        model_path = os.path.join(os.path.dirname(__file__), 'models', filename)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = joblib.load(model_path)
        return model
    except Exception as e:
        raise Exception(f"Error loading model {model_name}: {str(e)}")

# --- Data Loading with Caching ---
@st.cache_data  # <-- Use for data that can be hashed (like a DataFrame)
def get_model_performance() -> Optional[pd.DataFrame]:
    """
    Load model performance summary if available.
    Uses @st.cache_data to prevent rereading the CSV file on every app rerun.
    """
    # ... (rest of your function is perfect, no changes needed inside)
    try:
        performance_path = os.path.join(os.path.dirname(__file__), 'models', 'model_performance_summary.csv')
        if os.path.exists(performance_path):
            return pd.read_csv(performance_path)
        return None
    except Exception:
        return None

# --- Your other functions are great, no changes needed for them yet ---

def predict_loan_default(model, input_data: Dict[str, Any]) -> Dict[str, Any]:
    # ... (This function is perfect as is)
    try:
        df = pd.DataFrame([input_data])
        prediction = model.predict(df)[0]
        prediction_proba = model.predict_proba(df)[0]
        result = {
            'prediction': int(prediction),
            'prediction_label': 'Default' if prediction == 1 else 'No Default',
            'probability_no_default': float(prediction_proba[0]),
            'probability_default': float(prediction_proba[1]),
            'confidence': float(max(prediction_proba))
        }
        return result
    except Exception as e:
        raise Exception(f"Error making prediction: {str(e)}")


def get_available_models() -> list:
    # ... (This function is perfect as is)
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    if not os.path.exists(models_dir):
        return []
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib')]
    model_names = [f.replace('_pipeline.joblib', '').replace('_', ' ').title() for f in model_files]
    return sorted(model_names) # Added sorted() for a consistent order