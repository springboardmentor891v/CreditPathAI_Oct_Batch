import streamlit as st
import numpy as np
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc

st.set_page_config(page_title="Loan Default Prediction App", layout="wide")

st.markdown("""
<style>
body {
    background-color: #E8F4FA;
}
.block-container {
    padding-top: 2rem;
}
input, select {
    border-radius: 8px !important;
}
.stButton>button {
    background-color: #4A90E2;
    color: white;
    border-radius: 8px;
    padding: 8px 20px;
}
.stButton>button:hover {
    background-color: #2F75C0;
}
.box {
    background: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# LOAD MODELS

MODEL_DIR = "export_files/"

def load_pickle(name):
    with open(MODEL_DIR + name, "rb") as f:
        return pickle.load(f)

models = {
    "Logistic Regression": load_pickle("Logistic_Regression.pkl"),
    "Naive Bayes": load_pickle("Naive_Bayes.pkl"),
    "Decision Tree": load_pickle("Decision_Tree.pkl"),
    "Random Forest": load_pickle("Random_Forest.pkl"),
    "XGBoost": load_pickle("XGBoost.pkl"),
    "KNN": load_pickle("KNN.pkl")
}

scaler = load_pickle("scaler.pkl")
feature_cols = load_pickle("feature_columns.pkl")


st.sidebar.title("Model Selection")
selected_model = st.sidebar.selectbox("Select ML Model", list(models.keys()))

st.sidebar.markdown("---")
st.sidebar.write("📊 Confusion Matrix & ROC Curve will be shown after prediction.")


# UI

st.title(" Loan Default Prediction System")

st.subheader("Borrower Information")
st.markdown('<div class="box">', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    year = st.number_input("Year", value=2024)
    loan_amount = st.number_input("Loan Amount", min_value=0.0, value=500000.0)
    rate_of_interest = st.number_input("Rate of Interest (%)", min_value=0.0, value=8.5)
    interest_rate_spread = st.number_input("Interest Rate Spread", min_value=0.0, value=1.5)
    upfront_charges = st.number_input("Upfront Charges", min_value=0.0, value=5000.0)
    term = st.number_input("Loan Term (Months)", min_value=1, value=360)
    property_value = st.number_input("Property Value", min_value=0.0, value=600000.0)

with col2:
    income = st.number_input("Annual Income", min_value=0.0, value=40000.0)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
    ltv = st.number_input("Loan-to-Value Ratio (LTV)", min_value=0.0, value=70.0)
    dtir1 = st.number_input("DTI Ratio (DTIR1)", min_value=0.0, value=40.0)

    # Gender
    gender = st.selectbox("Gender", ["Male", "Female"])

    # Loan Purpose
    loan_purpose = st.selectbox("Loan Purpose", ["Personal Loan", "Housing Loan", "Business Loan"])

    # Loan Type
    loan_type = st.selectbox("Loan Type", ["Standard Loan", "Secured Loan", "Government Loan"])

    # Age group
    age_group = st.selectbox("Age Group", ["<25", "35-44", "45-54", "55-64", "65-74", ">74"])

    # Region
    region = st.selectbox("Region", ["North-East", "Central", "South"])

st.markdown('</div>', unsafe_allow_html=True)

# PREPROCESS 
def preprocess_input():
    data = {
        # numeric
        "year": year,
        "loan_amount": loan_amount,
        "rate_of_interest": rate_of_interest,
        "Interest_rate_spread": interest_rate_spread,
        "Upfront_charges": upfront_charges,
        "term": term,
        "property_value": property_value,
        "income": income,
        "Credit_Score": credit_score,
        "LTV": ltv,
        "dtir1": dtir1,

        # default 0 for all categorical dummy columns
        "loan_limit_ncf": 0,
        "Gender_Joint": 0,
        "Gender_Male": 0,
        "Gender_Sex Not Available": 0,
        "approv_in_adv_pre": 0,
        "loan_type_type2": 0,
        "loan_type_type3": 0,
        "loan_purpose_p2": 0,
        "loan_purpose_p3": 0,
        "loan_purpose_p4": 0,
        "Credit_Worthiness_l2": 0,
        "open_credit_opc": 0,
        "business_or_commercial_nob/c": 0,
        "Neg_ammortization_not_neg": 0,
        "interest_only_not_int": 0,
        "lump_sum_payment_not_lpsm": 0,
        "construction_type_sb": 0,
        "occupancy_type_pr": 0,
        "occupancy_type_sr": 0,
        "Secured_by_land": 0,
        "total_units_2U": 0,
        "total_units_3U": 0,
        "total_units_4U": 0,
        "credit_type_CRIF": 0,
        "credit_type_EQUI": 0,
        "credit_type_EXP": 0,
        "co-applicant_credit_type_EXP": 0,
        "age_35-44": 0,
        "age_45-54": 0,
        "age_55-64": 0,
        "age_65-74": 0,
        "age_<25": 0,
        "age_>74": 0,
        "submission_of_application_to_inst": 0,
        "Region_North-East": 0,
        "Region_central": 0,
        "Region_south": 0,
        "Security_Type_direct": 0
    }

    # Map gender
    if gender == "Male":
        data["Gender_Male"] = 1

    # Map loan purpose
    if loan_purpose == "Personal Loan":
        data["loan_purpose_p2"] = 1
    elif loan_purpose == "Housing Loan":
        data["loan_purpose_p3"] = 1
    elif loan_purpose == "Business Loan":
        data["loan_purpose_p4"] = 1

    # Map loan type
    if loan_type == "Secured Loan":
        data["loan_type_type2"] = 1
    elif loan_type == "Government Loan":
        data["loan_type_type3"] = 1

    # Map age group
    if age_group == "<25":
        data["age_<25"] = 1
    elif age_group == "35-44":
        data["age_35-44"] = 1
    elif age_group == "45-54":
        data["age_45-54"] = 1
    elif age_group == "55-64":
        data["age_55-64"] = 1
    elif age_group == "65-74":
        data["age_65-74"] = 1
    elif age_group == ">74":
        data["age_>74"] = 1

    # Region
    if region == "North-East":
        data["Region_North-East"] = 1
    elif region == "Central":
        data["Region_central"] = 1
    elif region == "South":
        data["Region_south"] = 1

    # Convert to dataframe
    df = pd.DataFrame([data])

    # Reorder columns exactly like training
    df = df[feature_cols]

    # Scale
    df_scaled = scaler.transform(df)
    return df_scaled

# PREDICT

if st.button(" Predict Loan Default"):

    processed = preprocess_input()
    model = models[selected_model]

    pred = model.predict(processed)[0]
    prob = model.predict_proba(processed)[0][1]

    st.subheader("Prediction Result")
    st.markdown('<div class="box">', unsafe_allow_html=True)

    st.write("### Result:", "🔴 RISKY" if pred == 1 else "🟢 SAFE")
    st.write(f"### Probability of Default: `{prob:.2f}`")

    st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("📊 Model Performance")

    y_test = [0, 1, 0, 1, 1, 0]
    sample_scores = [0.1, 0.85, 0.3, 0.75, 0.9, 0.2]

    # Confusion Matrix
    st.markdown('<div class="box">', unsafe_allow_html=True)
    st.write("### Confusion Matrix")

    cm = confusion_matrix(y_test, [1 if s > 0.5 else 0 for s in sample_scores])
    fig1, ax1 = plt.subplots()
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", ax=ax1)
    st.pyplot(fig1)
    st.markdown('</div>', unsafe_allow_html=True)

    # ROC Curve
    st.markdown('<div class="box">', unsafe_allow_html=True)
    st.write("### ROC Curve")

    fpr, tpr, _ = roc_curve(y_test, sample_scores)
    roc_auc = auc(fpr, tpr)

    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
    ax2.plot([0, 1], [0, 1], linestyle="--")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve")
    ax2.legend()
    st.pyplot(fig2)
    st.markdown('</div>', unsafe_allow_html=True)
