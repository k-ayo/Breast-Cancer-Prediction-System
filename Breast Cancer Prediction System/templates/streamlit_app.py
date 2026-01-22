# streamlit_app.py
import streamlit as st
import joblib
import numpy as np
import os

st.set_page_config(page_title="Breast Cancer Predictor", layout="centered")

st.title("Breast Cancer Prediction System")
st.write("⚠️ This application is for educational purposes only. Not a medical diagnostic tool.")

MODEL_PATH = "model/breast_cancer_model.pkl"

# Check model existence
if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Please ensure 'model/breast_cancer_model.pkl' exists.")
    st.stop()

# Load model
model = joblib.load(MODEL_PATH)

FEATURES = [
    "radius_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "concavity_mean"
]

st.sidebar.header("Input Tumor Features")

inputs = []
for feature in FEATURES:
    value = st.sidebar.number_input(
        label=feature.replace("_", " ").title(),
        min_value=0.0,
        value=1.0,
        format="%.4f"
    )
    inputs.append(value)

if st.sidebar.button("Predict"):
    X = np.array(inputs).reshape(1, -1)

    try:
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]

        label = "Malignant" if prediction == 1 else "Benign"

        st.success(f"### Prediction Result: **{label}**")
        st.write(
            f"**Probability** → Benign: {probabilities[0]:.4f}, "
            f"Malignant: {probabilities[1]:.4f}"
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown("---")
st.caption(
    "Ensure the trained model file exists at "
    "`model/breast_cancer_model.pkl` before deployment."
)
