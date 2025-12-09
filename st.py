import streamlit as st
import pandas as pd
import pickle
from sklearn.pipeline import Pipeline

# -------------------------------------------------
# Load your saved model (Pipeline with preprocessing)
# -------------------------------------------------
@st.cache_resource
def load_model():
    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# -------------------------------------------------
# Streamlit UI
# -------------------------------------------------
st.title("Loan Recovery Prediction App")
st.write("Automating and optimizing the loan recovery lifecycle by modelling repayment behaviour using diverse data.")

st.sidebar.header("Borrower Information")

# Example input fields (modify based on your dataset)
age = st.sidebar.number_input("Borrower Age", min_value=18, max_value=90, value=30)
income = st.sidebar.number_input("Annual Income", min_value=0, value=50000)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0, value=10000)
term = st.sidebar.selectbox("Loan Term", ["Short", "Medium", "Long"])
employment = st.sidebar.selectbox("Employment Type", ["Salaried", "Self-Employed", "Unemployed"])
credit_score = st.sidebar.number_input("Credit Score", min_value=300, max_value=900, value=700)

# Convert to dataframe (must match training features)
input_data = pd.DataFrame({
    "age": [age],
    "income": [income],
    "loan_amount": [loan_amount],
    "term": [term],
    "employment": [employment],
    "credit_score": [credit_score],
})

# -------------------------------------------------
# Prediction
# -------------------------------------------------
if st.button("Predict Repayment Behavior"):

    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if pred == 1:
        st.success(f"The borrower is **likely to repay**. (Probability: {prob:.2f})")
    else:
        st.error(f"The borrower is **likely to default**. (Probability: {prob:.2f})")

# -------------------------------------------------
# Show Model Performance 
# -------------------------------------------------
if st.checkbox("Show Model Metrics"):
    st.subheader("Model Performance Metrics")
    metrics = pd.read_csv("final_model_metrics.csv") 
    st.dataframe(metrics)
