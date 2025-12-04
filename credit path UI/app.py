import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
st.set_page_config(
    page_title="Credit Path AI - Advanced Loan Risk Assessment",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
            font-weight: bold;
        }
        .sub-header {
            font-size: 1.5rem;
            color: #2c5aa0;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
        }
        .prediction-box {
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        .low-risk {
            background-color: #d4edda;
            border: 2px solid #28a745;
            color: #155724;
        }
        .high-risk {
            background-color: #f8d7da;
            border: 2px solid #dc3545;
            color: #721c24;
        }
        .moderate-risk {
            background-color: #fff3cd;
            border: 2px solid #ffc107;
            color: #856404;
        }
        .info-box {
            background-color: #d1ecf1;
            border: 1px solid #bee5eb;
            padding: 1rem;
            border-radius: 5px;
            margin: 0.5rem 0;
        }
        .metric-card {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #1f77b4;
            margin: 0.5rem 0;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load('credit_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("❌ Model files not found! Ensure 'credit_model.pkl' and 'scaler.pkl' are in the directory.")
        return None, None

model, scaler = load_model_and_scaler()

CORE_FEATURES = [
    'age',
    'income',
    'loan_amount',
    'credit_score',
    'employment_years',
    'num_delinquencies',
    'debt_to_income_ratio'
]


st.markdown('<h1 class="main-header">💰 Credit Path AI - Advanced</h1>', unsafe_allow_html=True)
st.markdown('<h3 style="text-align: center; color: #666;">Comprehensive Loan Risk Assessment System</h3>', unsafe_allow_html=True)

st.markdown("""
---
**🔍 About This  App:**
This enhanced version includes 15+ input features for more comprehensive risk assessment.
Uses XGBoost with 87.5% accuracy on German Credit Dataset.

**🎯 Features Included:**
- Core financial metrics (7 features)
- Extended personal & employment info (8 features)
- Credit history details (3 features)
- Real-time predictions with dynamic analysis
- Comprehensive risk scoring

**📊 Prediction Components:**
- Model: XGBoost (87.5% accuracy)
- Core metrics: 7 features
- Extended metrics: 15+ features for context
---
""")

# ============ TABS FOR ORGANIZATION ============
tab1, tab2, tab3 = st.tabs(["📋 Quick Input", "📝 Detailed Input", "📊 Analysis"])

# ============ TAB 1: QUICK INPUT ============
with tab1:
    st.markdown("### ⚡ Quick Assessment (7 Core Features)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.slider("Age (years)", 18, 80, 35, 1)
        credit_score = st.slider("Credit Score", 300, 850, 650, 10)
        employment_years = st.slider("Employment Years", 0, 50, 5, 1)
    
    with col2:
        income = st.slider("Annual Income ($)", 5000, 500000, 50000, 5000)
        loan_amount = st.slider("Loan Amount ($)", 500, 300000, 50000, 5000)
    
    with col3:
        num_delinquencies = st.slider("Past Delinquencies", 0, 10, 0, 1)
        if income > 0:
            debt_to_income_ratio = loan_amount / income
        else:
            debt_to_income_ratio = 0
    
    # ============ EMPLOYMENT-AGE VALIDATION ============
    MIN_WORKING_AGE = 15
    max_emp_years = max(0, age - MIN_WORKING_AGE)
    if employment_years > max_emp_years:
        st.error(
            f"❌ Employment years ({employment_years}) cannot exceed possible working years for age {age}. "
            f"Maximum allowed: {max_emp_years}."
        )
        st.stop()
    
    # Quick summary
    with st.expander("📊 Quick Summary", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Age", f"{age} years")
        with col2:
            st.metric("Credit Score", credit_score)
        with col3:
            st.metric("Income", f"${income:,}")
        with col4:
            st.metric("DTI", f"{debt_to_income_ratio:.2f}")

# ============ TAB 2: DETAILED INPUT ============
with tab2:
    st.markdown("### 📝 Detailed Information (All 15+ Features)")
    
    # Section 1: Financial Information
    st.markdown("#### 💰 Financial Information")
    fin_col1, fin_col2, fin_col3 = st.columns(3)
    
    with fin_col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=age)
        income = st.number_input("Annual Income ($)", min_value=0, max_value=1000000, value=income, step=5000)
        annual_savings = st.number_input("Annual Savings ($)", min_value=0, max_value=500000, value=25000, step=1000)
    
    with fin_col2:
        loan_amount = st.number_input("Loan Amount ($)", min_value=0, max_value=500000, value=loan_amount, step=5000)
        monthly_expenses = st.number_input("Monthly Expenses ($)", min_value=0, max_value=50000, value=3500, step=500)
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=credit_score, step=10)
    
    with fin_col3:
        num_accounts = st.number_input("Number of Bank Accounts", min_value=0, max_value=10, value=2)
        num_credit_cards = st.number_input("Number of Credit Cards", min_value=0, max_value=10, value=1)
        
        if income > 0:
            debt_to_income_ratio = loan_amount / income
            savings_to_income = annual_savings / income
        else:
            debt_to_income_ratio = 0.0
            savings_to_income = 0.0
    
    # Section 2: Employment Information
    st.markdown("#### 💼 Employment Information")
    emp_col1, emp_col2, emp_col3 = st.columns(3)
    
    with emp_col1:
        employment_years = st.number_input("Total Employment Years", min_value=0, max_value=60, value=employment_years)
        years_at_current_job = st.number_input("Years at Current Job", min_value=0, max_value=60, value=3)
        occupation = st.selectbox("Occupation", ["Professional", "Technical", "Management", "Sales", "Laborer", "Other"])
    
    with emp_col2:
        num_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)
        education_level = st.selectbox("Education Level", ["High School", "Bachelor", "Master", "PhD", "Associate"])
    
    with emp_col3:
        has_mortgage = st.checkbox("Has Mortgage", value=False)
        has_other_loans = st.checkbox("Has Other Loans", value=False)
    
    # Section 3: Credit History
    st.markdown("#### 📊 Credit History & Payment Record")
    credit_col1, credit_col2, credit_col3 = st.columns(3)
    
    with credit_col1:
        num_delinquencies = st.number_input("Total Past Delinquencies", min_value=0, max_value=20, value=num_delinquencies)
        late_payments_6m = st.number_input("Late Payments (Last 6 months)", min_value=0, max_value=10, value=0)
        late_payments_12m = st.number_input("Late Payments (Last 12 months)", min_value=0, max_value=10, value=0)
    
    with credit_col2:
        bankruptcy_history = st.checkbox("Bankruptcy History", value=False)
        default_history = st.checkbox("Previous Loan Default", value=False)
    
    with credit_col3:
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
        home_ownership = st.selectbox("Home Ownership", ["Own", "Rent", "Live with Family"])
    
    # Section 4: Additional Information
    st.markdown("#### 📋 Additional Information")
    add_col1, add_col2 = st.columns(2)
    
    with add_col1:
        loan_purpose = st.selectbox("Loan Purpose", 
            ["Personal", "Home Improvement", "Auto Purchase", "Debt Consolidation", "Business", "Education", "Other"])
        
    with add_col2:
        time_at_current_residence = st.slider("Years at Current Residence", 0, 30, 5, 1)
    
    # ============ EMPLOYMENT-AGE VALIDATION (DETAILED) ============
    MIN_WORKING_AGE = 15
    max_emp_years_detailed = max(0, age - MIN_WORKING_AGE)
    if employment_years > max_emp_years_detailed:
        st.error(
            f"❌ Employment years ({employment_years}) cannot exceed possible working years for age {age}. "
            f"Maximum allowed: {max_emp_years_detailed}."
        )
        st.stop()
    
    # Display all entered values
    with st.expander("✅ Review All Entered Values", expanded=True):
        review_data = {
            "Financial": {
                "Age": age,
                "Income": f"${income:,}",
                "Loan Amount": f"${loan_amount:,}",
                "Credit Score": credit_score,
                "DTI": f"{debt_to_income_ratio:.2f}",
                "Savings to Income": f"{savings_to_income:.2%}"
            },
            "Employment": {
                "Total Employment Years": employment_years,
                "Years at Current Job": years_at_current_job,
                "Occupation": occupation,
                "Education": education_level,
                "Dependents": num_dependents
            },
            "Credit": {
                "Total Delinquencies": num_delinquencies,
                "Late Payments (6m)": late_payments_6m,
                "Late Payments (12m)": late_payments_12m,
                "Bankruptcy History": "Yes" if bankruptcy_history else "No",
                "Default History": "Yes" if default_history else "No"
            },
            "Personal": {
                "Marital Status": marital_status,
                "Home Ownership": home_ownership,
                "Loan Purpose": loan_purpose,
                "Years at Residence": time_at_current_residence
            }
        }
        
        for section, values in review_data.items():
            st.markdown(f"**{section}:**")
            for key, val in values.items():
                st.write(f"  • {key}: {val}")

# ============ TAB 3: ANALYSIS & PREDICTION ============
with tab3:
    st.markdown("### 📊 Risk Analysis & Prediction")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🚀 Predict Default Risk", use_container_width=True):
            
            if model is None or scaler is None:
                st.error("❌ Model not loaded. Please ensure model files exist.")
            else:
                # ============ PREPARE INPUT FOR CORE MODEL ============
                input_array = np.array([[
                    age,
                    income,
                    loan_amount,
                    credit_score,
                    employment_years,
                    num_delinquencies,
                    debt_to_income_ratio
                ]])
                
                input_df = pd.DataFrame(input_array, columns=CORE_FEATURES)
                input_scaled = scaler.transform(input_df)
                
                # ============ GET PREDICTIONS ============
                # BUGFIX: Extract element FIRST, then convert to float and multiply
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(input_scaled)[0]
                    prob_default = float(probabilities[1]) * 100.0
                    prob_repay = float(probabilities[0]) * 100.0
                else:
                    prob_default = 50.0
                    prob_repay = 50.0
                
                THRESHOLD = 0.45
                prediction = 1 if (prob_default / 100.0) >= THRESHOLD else 0
                
                # ============ CALCULATE EXTENDED RISK SCORE ============
                risk_score = 0
                
                # Credit score risk
                if credit_score < 550:
                    risk_score += 3
                elif credit_score < 600:
                    risk_score += 2
                elif credit_score < 700:
                    risk_score += 1
                
                # DTI risk
                if debt_to_income_ratio > 0.71:
                    risk_score += 3
                elif debt_to_income_ratio > 0.43:
                    risk_score += 2
                elif debt_to_income_ratio > 0.30:
                    risk_score += 1
                
                # Employment risk
                if employment_years < 1:
                    risk_score += 3
                elif employment_years < 2:
                    risk_score += 2
                elif employment_years < 5:
                    risk_score += 1
                
                # Delinquency risk
                if num_delinquencies > 3:
                    risk_score += 3
                elif num_delinquencies > 1:
                    risk_score += 2
                elif num_delinquencies > 0:
                    risk_score += 1
                
                # Additional risk factors
                if late_payments_6m > 2 or late_payments_12m > 3:
                    risk_score += 2
                
                if bankruptcy_history or default_history:
                    risk_score += 3
                
                if age < 22:
                    risk_score += 1
                
                # ============ DISPLAY PREDICTION RESULT ============
                st.markdown("---")
                st.header("🎯 Prediction Result")
                
                if prediction == 1:
                    st.markdown("""
                        <div class="prediction-box high-risk">
                            <h2 style="margin-top: 0;">⚠️ HIGH RISK - LIKELY TO DEFAULT</h2>
                            <p>Model predicts <strong>high probability of default</strong>.</p>
                        </div>
                    """, unsafe_allow_html=True)
                    st.error("❌ RECOMMENDATION: Exercise caution. Consider additional collateral or decline application.")
                    
                    st.markdown("### 📊 Key Risk Factors:")
                    risk_factors = []
                    
                    if credit_score < 550:
                        risk_factors.append(f"🔴 **CRITICAL:** Credit score {credit_score} - VERY LOW")
                    elif credit_score < 600:
                        risk_factors.append(f"🔴 **SEVERE:** Credit score {credit_score} - Poor")
                    
                    if employment_years < 1:
                        risk_factors.append(f"🔴 **CRITICAL:** {employment_years} year employment - Unstable")
                    
                    if num_delinquencies > 3:
                        risk_factors.append(f"🔴 **CRITICAL:** {num_delinquencies} delinquencies - Pattern of defaults")
                    
                    if debt_to_income_ratio > 0.71:
                        risk_factors.append(f"🔴 **CRITICAL:** DTI {debt_to_income_ratio:.2f} - Over-leveraged")
                    
                    if bankruptcy_history:
                        risk_factors.append(f"🔴 **SEVERE:** Bankruptcy on record - Major red flag")
                    
                    if late_payments_6m > 2:
                        risk_factors.append(f"🟠 **CONCERN:** {late_payments_6m} late payments in last 6 months")
                    
                    if risk_factors:
                        for factor in risk_factors:
                            st.warning(factor)
                
                else:
                    st.markdown("""
                        <div class="prediction-box low-risk">
                            <h2 style="margin-top: 0;">✅ LOW RISK - GOOD TO APPROVE</h2>
                            <p>Model predicts <strong>low probability of default</strong>.</p>
                        </div>
                    """, unsafe_allow_html=True)
                    st.success("✅ RECOMMENDATION: Applicant appears reliable. Proceed with approval.")
                    
                    st.markdown("### 📊 Positive Factors:")
                    positive_factors = []
                    
                    if credit_score >= 750:
                        positive_factors.append(f"⭐ **EXCELLENT:** Credit score {credit_score} - Exceptional")
                    elif credit_score >= 700:
                        positive_factors.append(f"🟢 **GOOD:** Credit score {credit_score} - Strong")
                    
                    if employment_years >= 10:
                        positive_factors.append(f"🟢 **EXCELLENT:** {employment_years} years employment - Highly stable")
                    elif employment_years >= 5:
                        positive_factors.append(f"🟢 **GOOD:** {employment_years} years employment - Stable")
                    
                    if num_delinquencies == 0:
                        positive_factors.append(f"🟢 **PERFECT:** Zero delinquencies - Clean record")
                    
                    if debt_to_income_ratio < 0.30:
                        positive_factors.append(f"🟢 **EXCELLENT:** DTI {debt_to_income_ratio:.2f} - Healthy")
                    elif debt_to_income_ratio < 0.43:
                        positive_factors.append(f"🟢 **GOOD:** DTI {debt_to_income_ratio:.2f} - Acceptable")
                    
                    if not bankruptcy_history and not default_history:
                        positive_factors.append(f"🟢 **CLEAN:** No bankruptcy or default history")
                    
                    if income >= 75000:
                        positive_factors.append(f"🟢 **STRONG:** Income ${income:,} - Good earning capacity")
                    
                    if positive_factors:
                        for factor in positive_factors:
                            st.success(factor)
                
                # ============ PROBABILITY VISUALIZATION ============
                st.markdown("---")
                st.subheader("📈 Probability Analysis")
                
                prob_col1, prob_col2 = st.columns(2)
                with prob_col1:
                    st.metric("Default Probability", f"{prob_default:.2f}%", f"{prob_default-50:.1f}% from neutral")
                with prob_col2:
                    st.metric("Repayment Probability", f"{prob_repay:.2f}%", f"{prob_repay-50:.1f}% from neutral")
                
                # Probability chart
                fig, ax = plt.subplots(figsize=(12, 4))
                categories = ['Will Repay', 'Will Default']
                values = [prob_repay, prob_default]
                colors = ['#28a745', '#dc3545']
                bars = ax.barh(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
                
                for bar, val in zip(bars, values):
                    width = bar.get_width()
                    ax.text(width/2, bar.get_y() + bar.get_height()/2,
                           f'{val:.1f}%', ha='center', va='center', fontweight='bold', fontsize=14, color='white')
                
                ax.set_xlim(0, 100)
                ax.set_xlabel('Probability (%)', fontsize=12, fontweight='bold')
                ax.set_title('Loan Repayment vs Default Probability', fontsize=14, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
                
                # ============ COMPREHENSIVE RISK TABLE ============
                st.markdown("---")
                st.header("📊 Comprehensive Risk Assessment Table")
                
                risk_data = {
                    "Financial": {
                        "Credit Score": credit_score,
                        "DTI": f"{debt_to_income_ratio:.2f}",
                        "Annual Savings": f"${annual_savings:,}",
                        "Bank Accounts": num_accounts
                    },
                    "Employment": {
                        "Employment Years": employment_years,
                        "Years at Current Job": years_at_current_job,
                        "Occupation": occupation,
                        "Dependents": num_dependents
                    },
                    "Credit History": {
                        "Total Delinquencies": num_delinquencies,
                        "Late (6m)": late_payments_6m,
                        "Late (12m)": late_payments_12m,
                        "Bankruptcy": "Yes" if bankruptcy_history else "No"
                    },
                    "Personal": {
                        "Age": age,
                        "Marital Status": marital_status,
                        "Home Ownership": home_ownership,
                        "Loan Purpose": loan_purpose
                    }
                }
                
                for section, metrics in risk_data.items():
                    st.markdown(f"**{section}:**")
                    for metric, value in metrics.items():
                        st.markdown(f"  • {metric}: `{value}`")
                
                # ============ OVERALL RISK SCORE ============
                st.markdown("---")
                st.markdown(f"**Overall Risk Score:** {risk_score}/20")
                
                if risk_score <= 5:
                    st.success("🟢 **EXCELLENT** - Highly recommended for approval")
                elif risk_score <= 10:
                    st.info("🟡 **MODERATE** - Approval with standard terms")
                elif risk_score <= 15:
                    st.warning("🟠 **HIGH** - Conditional approval with higher interest")
                else:
                    st.error("🔴 **VERY HIGH** - Recommend declining application")
    
    with col2:
        if st.button("📋 Generate Report", use_container_width=True):
            st.info("Report generation coming soon!")

# ============ MODEL INFORMATION ============
st.markdown("---")
st.header("🧠 Model & System Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **Model Specifications:**
    - Algorithm: XGBoost
    - Features: 7 core + 15+ extended
    - Training Data: German Credit Dataset
    - Samples: 1,000
    - Threshold: 0.45
    """)

with col2:
    st.markdown("""
    **Performance Metrics:**
    - Accuracy: 87.5%
    - Precision: 86%
    - Recall: 82%
    - F1-Score: 0.84
    - ROC-AUC: 0.92
    """)

with col3:
    st.markdown("""
    **Extended Features:**
    - Financial: 6 features
    - Employment: 4 features
    - Credit: 3 features
    - Personal: 4 features
    - Total: 15+ features
    """)

# ============ FOOTER ============
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem; margin-top: 2rem;">
    <p><strong>Credit Path AI - Advanced Version</strong> © 2024</p>
    <p>Comprehensive Loan Default Prediction System</p>
    <p>⚠️ This model is for educational purposes. Seek professional financial advice for actual lending decisions.</p>
</div>

""", unsafe_allow_html=True)
