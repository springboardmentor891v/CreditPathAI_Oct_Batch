import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from preprocess import Preprocessor
from models import ModelLoader

# -----------------------------------------------------
# 1. CONFIGURATION & CACHING
# -----------------------------------------------------
st.set_page_config(
    page_title="CreditPath AI | Loan Analysis",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_system():
    try:
        pre = Preprocessor()
        pre.load()
        loader = ModelLoader()
        loader.load_all()
        return pre, loader
    except Exception as e:
        return None, str(e)

pre, loader = load_system()

if isinstance(loader, str):
    st.error(f"System Error: {loader}")
    st.stop()

# -----------------------------------------------------
# 2. INTERNAL DATASET (20 ROWS)
# -----------------------------------------------------
# (Standard 20-row dataset logic)
mock_database = {
    1:  { "loan_limit": "cf", "Gender": "Male", "approv_in_adv": "nopre", "loan_type": "type1", "loan_purpose": "p1", "Credit_Worthiness": "l1", "open_credit": "nopc", "business_or_commercial": "nob/c", "Neg_ammortization": "not_neg", "interest_only": "not_int", "lump_sum_payment": "not_lpsm", "construction_type": "sb", "occupancy_type": "pr", "Secured_by": "home", "total_units": "1U", "credit_type": "CIB", "co-applicant_credit_type": "CIB", "age": "35-44", "submission_of_application": "to_inst", "Region": "North", "Security_Type": "direct", "loan_amount": 150000.0, "rate_of_interest": 3.0, "Interest_rate_spread": 0.1, "Upfront_charges": 1000.0, "term": 360.0, "property_value": 400000.0, "income": 12000.0, "Credit_Score": 850.0, "LTV": 37.5, "dtir1": 15.0 },
    2:  { "loan_limit": "cf", "Gender": "Female", "approv_in_adv": "nopre", "loan_type": "type1", "loan_purpose": "p3", "Credit_Worthiness": "l1", "open_credit": "nopc", "business_or_commercial": "nob/c", "Neg_ammortization": "not_neg", "interest_only": "not_int", "lump_sum_payment": "not_lpsm", "construction_type": "sb", "occupancy_type": "pr", "Secured_by": "home", "total_units": "1U", "credit_type": "CRIF", "co-applicant_credit_type": "EXP", "age": "45-54", "submission_of_application": "not_inst", "Region": "central", "Security_Type": "direct", "loan_amount": 250000.0, "rate_of_interest": 4.5, "Interest_rate_spread": 0.5, "Upfront_charges": 1500.0, "term": 360.0, "property_value": 350000.0, "income": 6000.0, "Credit_Score": 720.0, "LTV": 71.0, "dtir1": 38.0 },
    3:  { "loan_limit": "cf", "Gender": "Male", "approv_in_adv": "nopre", "loan_type": "type1", "loan_purpose": "p4", "Credit_Worthiness": "l1", "open_credit": "nopc", "business_or_commercial": "nob/c", "Neg_ammortization": "not_neg", "interest_only": "not_int", "lump_sum_payment": "not_lpsm", "construction_type": "sb", "occupancy_type": "pr", "Secured_by": "home", "total_units": "1U", "credit_type": "EXP", "co-applicant_credit_type": "CIB", "age": "55-64", "submission_of_application": "to_inst", "Region": "North-East", "Security_Type": "direct", "loan_amount": 400000.0, "rate_of_interest": 4.0, "Interest_rate_spread": 0.3, "Upfront_charges": 2000.0, "term": 360.0, "property_value": 500000.0, "income": 8000.0, "Credit_Score": 750.0, "LTV": 80.0, "dtir1": 42.0 },
    4:  { "loan_limit": "cf", "Gender": "Joint", "approv_in_adv": "nopre", "loan_type": "type2", "loan_purpose": "p1", "Credit_Worthiness": "l1", "open_credit": "nopc", "business_or_commercial": "b/c", "Neg_ammortization": "not_neg", "interest_only": "not_int", "lump_sum_payment": "not_lpsm", "construction_type": "sb", "occupancy_type": "pr", "Secured_by": "home", "total_units": "1U", "credit_type": "CIB", "co-applicant_credit_type": "CIB", "age": "25-34", "submission_of_application": "not_inst", "Region": "south", "Security_Type": "direct", "loan_amount": 300000.0, "rate_of_interest": 3.8, "Interest_rate_spread": 0.2, "Upfront_charges": 1200.0, "term": 180.0, "property_value": 450000.0, "income": 9000.0, "Credit_Score": 780.0, "LTV": 66.0, "dtir1": 30.0 },
    5:  { "loan_limit": "ncf", "Gender": "Sex Not Available", "approv_in_adv": "nopre", "loan_type": "type1", "loan_purpose": "p3", "Credit_Worthiness": "l1", "open_credit": "nopc", "business_or_commercial": "nob/c", "Neg_ammortization": "not_neg", "interest_only": "not_int", "lump_sum_payment": "not_lpsm", "construction_type": "mh", "occupancy_type": "sr", "Secured_by": "land", "total_units": "1U", "credit_type": "CRIF", "co-applicant_credit_type": "EXP", "age": "65-74", "submission_of_application": "to_inst", "Region": "central", "Security_Type": "Indriect", "loan_amount": 100000.0, "rate_of_interest": 5.0, "Interest_rate_spread": 0.6, "Upfront_charges": 500.0, "term": 240.0, "property_value": 150000.0, "income": 4000.0, "Credit_Score": 690.0, "LTV": 66.0, "dtir1": 25.0 },
    6:  { "loan_limit": "cf", "Gender": "Male", "approv_in_adv": "nopre", "loan_type": "type1", "loan_purpose": "p1", "Credit_Worthiness": "l1", "open_credit": "nopc", "business_or_commercial": "nob/c", "Neg_ammortization": "not_neg", "interest_only": "not_int", "lump_sum_payment": "not_lpsm", "construction_type": "sb", "occupancy_type": "pr", "Secured_by": "home", "total_units": "1U", "credit_type": "CIB", "co-applicant_credit_type": "CIB", "age": "35-44", "submission_of_application": "to_inst", "Region": "North", "Security_Type": "direct", "loan_amount": 220000.0, "rate_of_interest": 3.2, "Interest_rate_spread": 0.15, "Upfront_charges": 1100.0, "term": 360.0, "property_value": 350000.0, "income": 7500.0, "Credit_Score": 740.0, "LTV": 62.0, "dtir1": 36.0 },
    7:  { "loan_limit": "cf", "Gender": "Female", "approv_in_adv": "nopre", "loan_type": "type2", "loan_purpose": "p3", "Credit_Worthiness": "l1", "open_credit": "nopc", "business_or_commercial": "b/c", "Neg_ammortization": "not_neg", "interest_only": "not_int", "lump_sum_payment": "not_lpsm", "construction_type": "sb", "occupancy_type": "pr", "Secured_by": "home", "total_units": "1U", "credit_type": "EXP", "co-applicant_credit_type": "EXP", "age": "45-54", "submission_of_application": "not_inst", "Region": "south", "Security_Type": "direct", "loan_amount": 350000.0, "rate_of_interest": 4.2, "Interest_rate_spread": 0.4, "Upfront_charges": 1800.0, "term": 360.0, "property_value": 420000.0, "income": 6500.0, "Credit_Score": 710.0, "LTV": 83.0, "dtir1": 40.0 },
    8:  { "loan_limit": "cf", "Gender": "Male", "approv_in_adv": "pre", "loan_type": "type1", "loan_purpose": "p1", "Credit_Worthiness": "l2", "open_credit": "opc", "business_or_commercial": "nob/c", "Neg_ammortization": "neg_amm", "interest_only": "int_only", "lump_sum_payment": "not_lpsm", "construction_type": "sb", "occupancy_type": "ir", "Secured_by": "home", "total_units": "2U", "credit_type": "EQUI", "co-applicant_credit_type": "EXP", "age": "<25", "submission_of_application": "to_inst", "Region": "North-East", "Security_Type": "direct", "loan_amount": 550000.0, "rate_of_interest": 6.5, "Interest_rate_spread": 1.5, "Upfront_charges": 0.0, "term": 360.0, "property_value": 600000.0, "income": 3000.0, "Credit_Score": 600.0, "LTV": 91.0, "dtir1": 50.0 },
    9:  { "loan_limit": "cf", "Gender": "Female", "approv_in_adv": "pre", "loan_type": "type3", "loan_purpose": "p4", "Credit_Worthiness": "l2", "open_credit": "opc", "business_or_commercial": "nob/c", "Neg_ammortization": "neg_amm", "interest_only": "int_only", "lump_sum_payment": "lpsm", "construction_type": "sb", "occupancy_type": "sr", "Secured_by": "home", "total_units": "3U", "credit_type": "EXP", "co-applicant_credit_type": "EXP", "age": ">74", "submission_of_application": "not_inst", "Region": "central", "Security_Type": "direct", "loan_amount": 120000.0, "rate_of_interest": 7.0, "Interest_rate_spread": 2.0, "Upfront_charges": 500.0, "term": 180.0, "property_value": 130000.0, "income": 1500.0, "Credit_Score": 550.0, "LTV": 92.0, "dtir1": 55.0 },
    10: { "loan_limit": "ncf", "Gender": "Joint", "approv_in_adv": "pre", "loan_type": "type2", "loan_purpose": "p3", "Credit_Worthiness": "l2", "open_credit": "opc", "business_or_commercial": "b/c", "Neg_ammortization": "neg_amm", "interest_only": "int_only", "lump_sum_payment": "not_lpsm", "construction_type": "mh", "occupancy_type": "ir", "Secured_by": "land", "total_units": "4U", "credit_type": "EQUI", "co-applicant_credit_type": "EXP", "age": "35-44", "submission_of_application": "to_inst", "Region": "south", "Security_Type": "Indriect", "loan_amount": 600000.0, "rate_of_interest": 8.0, "Interest_rate_spread": 2.5, "Upfront_charges": 0.0, "term": 360.0, "property_value": 610000.0, "income": 2000.0, "Credit_Score": 500.0, "LTV": 98.0, "dtir1": 60.0 },
    11: { "loan_limit": "cf", "Gender": "Male", "approv_in_adv": "pre", "loan_type": "type2", "loan_purpose": "p3", "Credit_Worthiness": "l2", "open_credit": "opc", "business_or_commercial": "b/c", "Neg_ammortization": "neg_amm", "interest_only": "int_only", "lump_sum_payment": "not_lpsm", "construction_type": "mh", "occupancy_type": "ir", "Secured_by": "land", "total_units": "4U", "credit_type": "EXP", "co-applicant_credit_type": "EXP", "age": "<25", "submission_of_application": "to_inst", "Region": "south", "Security_Type": "Indriect", "loan_amount": 600000.0, "rate_of_interest": 8.5, "Interest_rate_spread": 3.0, "Upfront_charges": 0.0, "term": 360.0, "property_value": 610000.0, "income": 1000.0, "Credit_Score": 500.0, "LTV": 98.0, "dtir1": 60.0 },
    12: { "loan_limit": "ncf", "Gender": "Joint", "approv_in_adv": "pre", "loan_type": "type3", "loan_purpose": "p4", "Credit_Worthiness": "l2", "open_credit": "opc", "business_or_commercial": "b/c", "Neg_ammortization": "neg_amm", "interest_only": "int_only", "lump_sum_payment": "lpsm", "construction_type": "sb", "occupancy_type": "sr", "Secured_by": "home", "total_units": "4U", "credit_type": "EQUI", "co-applicant_credit_type": "EXP", "age": ">74", "submission_of_application": "to_inst", "Region": "North-East", "Security_Type": "direct", "loan_amount": 100000.0, "rate_of_interest": 9.0, "Interest_rate_spread": 5.0, "Upfront_charges": 5000.0, "term": 180.0, "property_value": 105000.0, "income": 500.0, "Credit_Score": 400.0, "LTV": 95.0, "dtir1": 80.0 },
    13: { "loan_limit": "cf", "Gender": "Female", "approv_in_adv": "nopre", "loan_type": "type1", "loan_purpose": "p2", "Credit_Worthiness": "l1", "open_credit": "nopc", "business_or_commercial": "nob/c", "Neg_ammortization": "not_neg", "interest_only": "not_int", "lump_sum_payment": "not_lpsm", "construction_type": "sb", "occupancy_type": "pr", "Secured_by": "home", "total_units": "1U", "credit_type": "CIB", "co-applicant_credit_type": "CIB", "age": "35-44", "submission_of_application": "not_inst", "Region": "North", "Security_Type": "direct", "loan_amount": 280000.0, "rate_of_interest": 3.9, "Interest_rate_spread": 0.25, "Upfront_charges": 1300.0, "term": 360.0, "property_value": 350000.0, "income": 6800.0, "Credit_Score": 730.0, "LTV": 80.0, "dtir1": 39.0 },
    14: { "loan_limit": "cf", "Gender": "Male", "approv_in_adv": "pre", "loan_type": "type1", "loan_purpose": "p3", "Credit_Worthiness": "l2", "open_credit": "opc", "business_or_commercial": "nob/c", "Neg_ammortization": "neg_amm", "interest_only": "int_only", "lump_sum_payment": "lpsm", "construction_type": "sb", "occupancy_type": "ir", "Secured_by": "home", "total_units": "2U", "credit_type": "EXP", "co-applicant_credit_type": "EXP", "age": "55-64", "submission_of_application": "to_inst", "Region": "central", "Security_Type": "direct", "loan_amount": 180000.0, "rate_of_interest": 6.0, "Interest_rate_spread": 1.0, "Upfront_charges": 0.0, "term": 300.0, "property_value": 200000.0, "income": 3500.0, "Credit_Score": 620.0, "LTV": 90.0, "dtir1": 48.0 },
    15: { "loan_limit": "cf", "Gender": "Joint", "approv_in_adv": "nopre", "loan_type": "type1", "loan_purpose": "p1", "Credit_Worthiness": "l1", "open_credit": "nopc", "business_or_commercial": "nob/c", "Neg_ammortization": "not_neg", "interest_only": "not_int", "lump_sum_payment": "not_lpsm", "construction_type": "sb", "occupancy_type": "pr", "Secured_by": "home", "total_units": "1U", "credit_type": "CIB", "co-applicant_credit_type": "CIB", "age": "25-34", "submission_of_application": "to_inst", "Region": "North", "Security_Type": "direct", "loan_amount": 210000.0, "rate_of_interest": 3.1, "Interest_rate_spread": 0.1, "Upfront_charges": 900.0, "term": 360.0, "property_value": 300000.0, "income": 7200.0, "Credit_Score": 760.0, "LTV": 70.0, "dtir1": 33.0 },
    16: { "loan_limit": "cf", "Gender": "Male", "approv_in_adv": "pre", "loan_type": "type2", "loan_purpose": "p4", "Credit_Worthiness": "l2", "open_credit": "opc", "business_or_commercial": "b/c", "Neg_ammortization": "neg_amm", "interest_only": "int_only", "lump_sum_payment": "not_lpsm", "construction_type": "sb", "occupancy_type": "sr", "Secured_by": "home", "total_units": "1U", "credit_type": "CRIF", "co-applicant_credit_type": "EXP", "age": "45-54", "submission_of_application": "not_inst", "Region": "south", "Security_Type": "direct", "loan_amount": 320000.0, "rate_of_interest": 7.8, "Interest_rate_spread": 2.2, "Upfront_charges": 0.0, "term": 360.0, "property_value": 330000.0, "income": 4100.0, "Credit_Score": 580.0, "LTV": 96.0, "dtir1": 52.0 },
    17: { "loan_limit": "cf", "Gender": "Female", "approv_in_adv": "nopre", "loan_type": "type1", "loan_purpose": "p1", "Credit_Worthiness": "l1", "open_credit": "nopc", "business_or_commercial": "nob/c", "Neg_ammortization": "not_neg", "interest_only": "not_int", "lump_sum_payment": "not_lpsm", "construction_type": "sb", "occupancy_type": "pr", "Secured_by": "home", "total_units": "1U", "credit_type": "CIB", "co-applicant_credit_type": "CIB", "age": "35-44", "submission_of_application": "to_inst", "Region": "North-East", "Security_Type": "direct", "loan_amount": 260000.0, "rate_of_interest": 3.7, "Interest_rate_spread": 0.2, "Upfront_charges": 1400.0, "term": 360.0, "property_value": 380000.0, "income": 6900.0, "Credit_Score": 740.0, "LTV": 68.0, "dtir1": 37.0 },
    18: { "loan_limit": "ncf", "Gender": "Joint", "approv_in_adv": "pre", "loan_type": "type3", "loan_purpose": "p3", "Credit_Worthiness": "l2", "open_credit": "opc", "business_or_commercial": "b/c", "Neg_ammortization": "neg_amm", "interest_only": "int_only", "lump_sum_payment": "lpsm", "construction_type": "mh", "occupancy_type": "ir", "Secured_by": "land", "total_units": "4U", "credit_type": "EXP", "co-applicant_credit_type": "EXP", "age": ">74", "submission_of_application": "to_inst", "Region": "south", "Security_Type": "Indriect", "loan_amount": 150000.0, "rate_of_interest": 8.8, "Interest_rate_spread": 4.5, "Upfront_charges": 4000.0, "term": 240.0, "property_value": 160000.0, "income": 1800.0, "Credit_Score": 480.0, "LTV": 93.0, "dtir1": 65.0 },
    19: { "loan_limit": "cf", "Gender": "Male", "approv_in_adv": "nopre", "loan_type": "type1", "loan_purpose": "p1", "Credit_Worthiness": "l1", "open_credit": "nopc", "business_or_commercial": "nob/c", "Neg_ammortization": "not_neg", "interest_only": "not_int", "lump_sum_payment": "not_lpsm", "construction_type": "sb", "occupancy_type": "pr", "Secured_by": "home", "total_units": "1U", "credit_type": "CIB", "co-applicant_credit_type": "CIB", "age": "35-44", "submission_of_application": "to_inst", "Region": "North", "Security_Type": "direct", "loan_amount": 190000.0, "rate_of_interest": 3.3, "Interest_rate_spread": 0.15, "Upfront_charges": 1000.0, "term": 360.0, "property_value": 250000.0, "income": 5500.0, "Credit_Score": 710.0, "LTV": 76.0, "dtir1": 34.0 },
    20: { "loan_limit": "cf", "Gender": "Female", "approv_in_adv": "pre", "loan_type": "type2", "loan_purpose": "p3", "Credit_Worthiness": "l2", "open_credit": "opc", "business_or_commercial": "b/c", "Neg_ammortization": "neg_amm", "interest_only": "int_only", "lump_sum_payment": "not_lpsm", "construction_type": "sb", "occupancy_type": "ir", "Secured_by": "home", "total_units": "2U", "credit_type": "EXP", "co-applicant_credit_type": "EXP", "age": "<25", "submission_of_application": "not_inst", "Region": "central", "Security_Type": "Indriect", "loan_amount": 420000.0, "rate_of_interest": 7.2, "Interest_rate_spread": 1.8, "Upfront_charges": 0.0, "term": 360.0, "property_value": 430000.0, "income": 3100.0, "Credit_Score": 590.0, "LTV": 97.0, "dtir1": 58.0 }
}

# -----------------------------------------------------
# 3. SIDEBAR
# -----------------------------------------------------
with st.sidebar:
    st.header("🎛️ Control Panel")
    
    st.subheader("Data Auto-Fill")
    app_id = st.number_input("Applicant ID (1-20)", min_value=1, max_value=20, value=1, step=1)
    
    if st.button("🔍 Auto-fill from ID", type="secondary", use_container_width=True):
        if app_id in mock_database:
            st.session_state["form_data"] = mock_database[app_id]
            st.session_state["auto_filled"] = True
            st.success(f"Loaded Applicant #{app_id}")
            st.rerun()
        else:
            st.warning("ID not found.")

    st.divider()
    st.subheader("Ensemble Settings")
    risk_threshold = st.slider("Strictness Level", 0, 100, 50, help="If Average Risk Score > this %, loan is rejected.") / 100.0

# -----------------------------------------------------
# 4. INPUT FORM
# -----------------------------------------------------
st.title("💳 CreditPath AI Risk Analyzer")
st.write("Enter applicant details below or use the Sidebar to load a dataset record.")

if "form_data" not in st.session_state:
    st.session_state["form_data"] = {}

with st.expander("📝 Applicant Information Form", expanded=True):
    dd_vals = {
        'loan_limit': ['cf', 'ncf'], 'Gender': ['Female', 'Male', 'Joint', 'Sex Not Available'], 'approv_in_adv': ['nopre', 'pre'], 'loan_type': ['type1', 'type2', 'type3'], 'loan_purpose': ['p1', 'p2', 'p3', 'p4'], 'Credit_Worthiness': ['l1', 'l2'], 'open_credit': ['nopc', 'opc'], 'business_or_commercial': ['b/c', 'nob/c'], 'Neg_ammortization': ['not_neg', 'neg_amm'], 'interest_only': ['not_int', 'int_only'], 'lump_sum_payment': ['not_lpsm', 'lpsm'], 'construction_type': ['sb', 'mh'], 'occupancy_type': ['pr', 'sr', 'ir'], 'Secured_by': ['home', 'land'], 'total_units': ['1U', '2U', '3U', '4U'], 'credit_type': ['EXP', 'EQUI', 'CRIF', 'CIB'], 'co-applicant_credit_type': ['CIB', 'EXP'], 'age': ['<25','25-34','35-44','45-54','55-64','65-74','>74'], 'submission_of_application': ['to_inst', 'not_inst'], 'Region': ['south', 'North', 'central', 'North-East'], 'Security_Type': ['direct', 'Indriect']
    }

    user_input = {}
    c1, c2, c3, c4 = st.columns(4)
    cols = [c1, c2, c3, c4]
    
    for i, col in enumerate(pre.categorical_cols):
        with cols[i % 4]:
            default = st.session_state["form_data"].get(col, dd_vals[col][0])
            try: idx = dd_vals[col].index(default)
            except: idx = 0
            user_input[col] = st.selectbox(col, dd_vals[col], index=idx)

    st.markdown("---")
    for i, col in enumerate(pre.numeric_cols):
        with cols[i % 4]:
            default = st.session_state["form_data"].get(col, 0.0)
            user_input[col] = st.number_input(col, value=float(default))

# -----------------------------------------------------
# 5. PREDICTION BUTTONS
# -----------------------------------------------------
st.divider()
st.subheader("🚀 Prediction Controls")

col_all, col_single = st.columns(2)

with col_all:
    st.markdown("#### Option A: Ensemble Analysis")
    if st.button("🚀 Predict with All Models", type="primary", use_container_width=True):
        st.session_state["mode"] = "ALL"
        st.session_state["run"] = True

with col_single:
    st.markdown("#### Option B: Specific Model")
    model_list = list(loader.models.keys())
    selected_model = st.selectbox("Select Model:", model_list, label_visibility="collapsed")
    
    if st.button(f"🎯 Predict with {selected_model}", use_container_width=True):
        st.session_state["mode"] = "SINGLE"
        st.session_state["selected_model"] = selected_model
        st.session_state["run"] = True

# -----------------------------------------------------
# 6. RESULTS LOGIC
# -----------------------------------------------------
if st.session_state.get("run", False):
    
    X_processed = pre.transform(user_input)
    st.divider()
    
    # ----------------------------------------------------------
    # 6A. PREPROCESSING VISUALIZATION (Key Risk Intensity)
    # ----------------------------------------------------------
    with st.expander("🔍 Risk Factor Intensity Map (Scaled Data)", expanded=True):
        st.markdown("This chart highlights the **intensity of risk factors** seen by the AI (0=Low, 1=High). A longer bar indicates a higher relative value in the dataset.")
        
        # We filter X_processed to just the critical numeric columns to make the chart readable
        critical_cols = ["Credit_Score", "LTV", "dtir1", "income", "loan_amount"]
        
        # Safely extract if columns exist
        plot_data = {}
        for c in critical_cols:
            if c in X_processed.columns:
                plot_data[c] = X_processed[c].values[0]
        
        if plot_data:
            df_plot = pd.DataFrame(list(plot_data.items()), columns=["Feature", "Intensity"])
            
            chart_intensity = alt.Chart(df_plot).mark_bar().encode(
                x=alt.X('Intensity', scale=alt.Scale(domain=[0, 1])),
                y=alt.Y('Feature', sort='-x'),
                color=alt.Color('Intensity', scale=alt.Scale(scheme='reds')),
                tooltip=['Feature', 'Intensity']
            ).properties(height=200)
            
            st.altair_chart(chart_intensity, use_container_width=True)
        else:
            st.write("No critical features found in processed data.")


    # ----------------------------------------------------------
    # 6B. PREDICTION & GUARDRAILS
    # ----------------------------------------------------------
    st.subheader("📊 Analysis Results")
    
    if st.session_state["mode"] == "ALL":
        model_results = []
        total_risk_prob = 0.0
        
        risk_votes = 0
        safe_votes = 0
        
        for name in loader.models.keys():
            try:
                _, prob_risk = loader.predict(name, X_processed)
                total_risk_prob += prob_risk
                
                status = "Risk" if prob_risk > 0.5 else "Safe"
                if status == "Risk": risk_votes += 1
                else: safe_votes += 1
                
                model_results.append({"Model": name, "Verdict": status})
            except: pass
        
        # Calculate Base Score
        avg_risk_score = total_risk_prob / len(loader.models)
        
        # --- POLICY GUARDRAILS (The "Still Approved" Fix) ---
        policy_violations = []
        # Rule 1: Credit Score < 600 is Auto-Reject
        if user_input.get("Credit_Score", 0) < 600:
            policy_violations.append("Critical: Credit Score too low (< 600)")
        # Rule 2: LTV > 95% is Auto-Reject
        if user_input.get("LTV", 0) > 95:
            policy_violations.append("Critical: LTV Ratio too high (> 95%)")
            
        # If Violations -> Force Rejection
        if policy_violations:
            is_rejected = True
            avg_risk_score = max(avg_risk_score, 0.95) # Force high risk meter
            rejection_reason = " | ".join(policy_violations)
        else:
            is_rejected = avg_risk_score >= risk_threshold
            rejection_reason = "Ensemble Consensus"

        # DISPLAY
        c_verdict, c_meter = st.columns([1, 2])
        with c_verdict:
            if is_rejected:
                st.error(f"# REJECTED\n### High Risk Profile")
                if policy_violations:
                    st.caption(f"⛔ {rejection_reason}")
            else:
                st.success(f"# APPROVED\n### Safe Profile")

        with c_meter:
            st.write("**Risk Meter**")
            st.progress(avg_risk_score)
            c_leg1, c_leg2, c_leg3 = st.columns(3)
            c_leg1.caption("Safe Zone")
            c_leg2.caption("⚠️ Caution")
            c_leg3.caption("Risk Zone")

        st.write("")
        st.markdown("#### Model Consensus")
        
        c_chart, c_grid = st.columns([1, 1])
        
        with c_chart:
            # Donut Chart
            vote_data = pd.DataFrame({
                'Category': ['Safe Votes', 'Risk Votes'],
                'Count': [safe_votes, risk_votes]
            })
            
            base = alt.Chart(vote_data).encode(theta=alt.Theta("Count", stack=True))
            pie = base.mark_arc(outerRadius=100, innerRadius=50).encode(
                color=alt.Color("Category", scale=alt.Scale(domain=['Safe Votes', 'Risk Votes'], range=['#4caf50', '#f44336'])),
                tooltip=["Category", "Count"]
            )
            text = base.mark_text(radius=120).encode(
                text="Count", 
                order=alt.Order("Category"), 
                color=alt.value("black")
            )
            st.altair_chart(pie + text, use_container_width=True)
            
        with c_grid:
            st.markdown("##### Individual Model Verdicts")
            # Create a scrolling list or simple grid
            cols = st.columns(3)
            for i, res in enumerate(model_results):
                with cols[i % 3]:
                    if res['Verdict'] == "Risk":
                        st.error(f"🔴 {res['Model']}")
                    else:
                        st.success(f"🟢 {res['Model']}")

    elif st.session_state["mode"] == "SINGLE":
        target_model = st.session_state["selected_model"]
        try:
            _, prob = loader.predict(target_model, X_processed)
            status = "Risk" if prob > 0.5 else "Safe"
            
            c1, c2 = st.columns([1, 2])
            with c1:
                st.metric("Model Used", target_model)
            with c2:
                if status == "Risk":
                    st.error(f"## Verdict: RISK")
                else:
                    st.success(f"## Verdict: SAFE")
            
            st.progress(prob)
        except Exception as e:
            st.error(f"Error running {target_model}: {e}")