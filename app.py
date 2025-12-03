import streamlit as st
import pandas as pd
import pickle

# -----------------------------------
# Load the saved best model
# -----------------------------------
@st.cache_resource
def load_model():
    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# -----------------------------------
# Streamlit UI
# -----------------------------------
st.title("Loan Default Prediction App")
st.write("Predict whether a borrower will default or repay using machine learning.")

st.sidebar.header("Borrower Information")

# These MUST match your dataset features
age = st.sidebar.number_input("Age", min_value=18, max_value=90, value=30)
income = st.sidebar.number_input("Annual Income", min_value=0, value=50000)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=1000, value=15000)
open_credit = st.sidebar.number_input("Number of Open Credit Lines", min_value=0, max_value=30, value=5)
credit_score = st.sidebar.number_input("Credit Score", min_value=300, max_value=900, value=720)
loan_purpose = st.sidebar.selectbox("Loan Purpose", ["debt_consolidation", "home_improvement", "credit_card", "other"])

# Create input dataframe
input_data = pd.DataFrame({
    "age": [age],
    "income": [income],
    "loan_amount": [loan_amount],
    "open_credit": [open_credit],
    "Credit_Score": [credit_score],
    "loan_purpose": [loan_purpose]
})

# -----------------------------------
# Prediction
# -----------------------------------
if st.button("Predict Default Risk"):

    pred = model.predict(input_data)[0]

    # If model supports probability
    try:
        prob = model.predict_proba(input_data)[0][1]
    except:
        prob = None

    if pred == 1:
        st.error(f"❌ Borrower is LIKELY to DEFAULT. Prob: {prob:.2f}" if prob else "❌ Borrower is LIKELY to DEFAULT.")
    else:
        st.success(f"✔ Borrower is LIKELY to REPAY. Prob: {prob:.2f}" if prob else "✔ Borrower is LIKELY to REPAY.")

# -----------------------------------
# Show model table
# -----------------------------------
if st.checkbox("Show Model Metrics"):
    metrics = pd.read_csv("final_model_metrics.csv")
    st.dataframe(metrics)


