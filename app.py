# app.py
import streamlit as st
import pandas as pd
import traceback

# --- IMPORT your model logic here ---
# from loan_defaulter import predict_default
# or from main import some_predict_function

st.set_page_config(page_title="CreditPathAI – Loan Default Predictor", layout="centered")
st.title("CreditPathAI — Loan Default Predictor")

st.write("""
Enter the details below to get a loan-default prediction based on your trained model.
""")

st.sidebar.header("Input Parameters")

# --- Numeric inputs (all consistent type: int)
age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
monthly_income = st.sidebar.number_input("Monthly Income", min_value=0, value=20000, step=500)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0, value=50000, step=1000)
loan_term_months = st.sidebar.number_input("Loan Term (months)", min_value=1, max_value=360, value=60)

# --- Categorical input
employment_type = st.sidebar.selectbox("Employment Type", ["Salaried", "Self-employed", "Unemployed"])

# Compile input into dictionary
input_data = {
    "age": age,
    "monthly_income": monthly_income,
    "loan_amount": loan_amount,
    "loan_term_months": loan_term_months,
    "employment_type": employment_type
}

st.sidebar.subheader("Input preview")
st.sidebar.json(input_data)

if st.button("Predict Default Risk"):
    try:
        # --- Prediction logic here ---
        # Replace the placeholder with your actual model function
        # Example if your function is `predict_default(df)`:
        # X = pd.DataFrame([input_data])
        # result = predict_default(X)

        # Placeholder result for now
        result = {"default_risk": 0.23, "will_default": False}

        st.subheader("Prediction Result")
        st.write(result)

    except Exception as e:
        st.error("Error occurred during prediction")
        st.text(traceback.format_exc())
