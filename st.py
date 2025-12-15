# st_app.py
import streamlit as st
import pandas as pd
import pickle
import os

# -------------------------------
# Load saved models and columns
# -------------------------------
@st.cache_resource
def load_models_and_columns():
    model_files = [
        "logistic_regression.pkl",
        "decision_tree.pkl",
        "random_forest.pkl",
        "naive_bayes.pkl",
        "linear_svm.pkl",
        "xgboost.pkl"
    ]
    
    models = {}
    for f in model_files:
        with open(f, "rb") as file:
            model_name = f.replace(".pkl", "")
            models[model_name] = pickle.load(file)
    
    with open("feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)
    with open("num_cols.pkl", "rb") as f:
        num_cols = pickle.load(f)
    with open("cat_cols.pkl", "rb") as f:
        cat_cols = pickle.load(f)
    
    return models, feature_columns, num_cols, cat_cols


models, feature_columns, num_cols, cat_cols = load_models_and_columns()

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("Loan Recovery Prediction App")
st.write(
    "Automating and optimizing the loan recovery lifecycle by modelling "
    "repayment behaviour using diverse data."
)

st.sidebar.header("Borrower Information")

# ---- Inputs ----
age = st.sidebar.number_input("Borrower Age", 18, 90, 30)
income = st.sidebar.number_input("Annual Income", 0, value=50000)
loan_amount = st.sidebar.number_input("Loan Amount", 0, value=10000)
loan_term = st.sidebar.selectbox("Loan Term", ["Short", "Medium", "Long"])
employment = st.sidebar.selectbox(
    "Employment Type", ["Salaried", "Self-Employed", "Unemployed"]
)
credit_score = st.sidebar.number_input("Credit Score", 300, 900, 700)

# ---- Model selection ----
selected_model_name = st.sidebar.selectbox("Choose Model", list(models.keys()))
model = models[selected_model_name]

# -------------------------------
# Create input DataFrame
# -------------------------------
input_data = pd.DataFrame({
    "Age": [age],
    "Income": [income],
    "LoanAmount": [loan_amount],
    "LoanTerm": [loan_term],
    "EmploymentType": [employment],
    "CreditScore": [credit_score],
})

# Fill missing columns
for col in feature_columns:
    if col not in input_data.columns:
        input_data[col] = 0 if col in num_cols else "Unknown"

# Ensure correct column order
input_data = input_data[feature_columns]

# -------------------------------
# Predict button
# -------------------------------
if st.button("Predict Repayment Behavior"):
    try:
        prediction = model.predict(input_data)[0]

        if hasattr(model.named_steps["model"], "predict_proba"):
            prob = model.predict_proba(input_data)[0][1]
            prob_text = f"{prob:.2f}"
        else:
            prob_text = "N/A"

        if prediction == 1:
            st.error(f"The borrower is likely to default (Probability: {prob_text})")
        else:
            st.success(f"The borrower is likely to repay (Probability: {prob_text})")

    except Exception as e:
        st.error(f"Prediction error: {e}")

# -------------------------------
# Optional: Show model metrics
# -------------------------------
if st.checkbox("Show Model Metrics"):
    if os.path.exists("final_model_metrics.csv"):
        st.dataframe(pd.read_csv("final_model_metrics.csv"))
    else:
        st.warning("final_model_metrics.csv not found.")
