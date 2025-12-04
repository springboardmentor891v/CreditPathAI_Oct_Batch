# app.py
import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ML imports
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
)
from sklearn.neighbors import KNeighborsClassifier

# Metrics & plots
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# optional libs
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE
    IMB_AVAILABLE = True
except Exception:
    IMB_AVAILABLE = False

# ---------------------------
# Page config + CSS (Theme A)
# ---------------------------
st.set_page_config(page_title="CreditPathAI — Loan Default Predictor", layout="wide")

st.markdown("""
<style>
html, body { background-color:#f2f6f9; color:#222; font-family: 'Segoe UI', sans-serif; }
.header-box { background: linear-gradient(90deg,#0b8fab,#1769aa); padding:28px; color:white;
    border-radius:14px; text-align:center; font-size:26px; font-weight:700; margin-bottom:20px; }
.card { background:white; padding:22px; border-radius:12px; box-shadow:0 3px 10px rgba(0,0,0,0.08); margin-bottom:18px; }
.section-title { font-size:20px; font-weight:700; margin-bottom:6px; color:#0b516d; }
.result-low { background:#e7fff2; border-left:8px solid #0fa958; padding:28px; text-align:center; border-radius:12px;
    font-size:22px; font-weight:700; margin-top:20px; color:#0a5b33; }
.result-medium { background:#fff7df; border-left:8px solid #eab308; padding:28px; text-align:center; border-radius:12px;
    font-size:22px; font-weight:700; margin-top:20px; color:#a17800; }
.result-high { background:#ffecec; border-left:8px solid #d93737; padding:28px; text-align:center; border-radius:12px;
    font-size:22px; font-weight:700; margin-top:20px; color:#9b1c1c; }
.small-card { background:white; padding:18px; border-radius:10px; box-shadow:0 2px 8px rgba(0,0,0,0.06);
    text-align:center; font-weight:700; font-size:18px; }
.sidebar .sidebar-content { background: linear-gradient(180deg,#0b7285,#0ea5a4); color: white; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header-box">CREDITPATHAI — LOAN DEFAULT RISK PREDICTION SYSTEM</div>', unsafe_allow_html=True)

# ---------------------------
# Helper utility functions
# ---------------------------
def load_local_dataset(path="loan_dataset/Loan_Default.csv"):
    try:
        df = pd.read_csv(path)
        return df
    except Exception:
        return None

def safe_get_dummies(df, cat_cols):
    present = [c for c in cat_cols if c in df.columns]
    return pd.get_dummies(df, columns=present, drop_first=True) if present else df

def drop_safe(df, cols):
    return df.drop(columns=[c for c in cols if c in df.columns], errors="ignore")

def clean_colnames(df):
    df.columns = (
        df.columns.str.replace("[", "", regex=False)
                  .str.replace("]", "", regex=False)
                  .str.replace("<", "", regex=False)
                  .str.replace(">", "", regex=False)
                  .str.replace(" ", "_", regex=False)
    )
    return df

def build_feature_row_from_inputs(features, mapping):
    """
    Build a full DataFrame with all training features (columns = features)
    mapping: dict of dataset-column-key -> value (str or numeric)
    For string categorical values we set the matching dummy to 1 when column contains that value.
    For numeric values we set the numeric columns directly if exact match found.
    """
    X_input = pd.DataFrame(np.zeros((1, len(features))), columns=features)
    for key, value in mapping.items():
        # try to set numeric columns first:
        # if key exactly matches a feature name, set it
        if key in X_input.columns and (isinstance(value, (int, float, np.integer, np.floating))):
            X_input.loc[0, key] = value
            continue

        # otherwise look for substring matches
        for col in X_input.columns:
            col_low = col.lower()
            key_low = str(key).lower()
            if key_low in col_low:
                # If value is string (category), try match category text in column name:
                if isinstance(value, str):
                    val_low = value.lower()
                    if val_low in col_low:
                        X_input.loc[0, col] = 1
                else:
                    # numeric: set if column looks like numeric field
                    X_input.loc[0, col] = value
    return X_input

# ---------------------------
# Main area: example inputs (kept visible)
# ---------------------------
st.markdown("### Applicant Example Inputs")
st.markdown('<div class="card">', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    inp_age = st.number_input("Age (years)", min_value=18, max_value=90, value=30)
    inp_gender = st.selectbox("Gender (example)", options=["Male","Female"])
with col2:
    inp_income = st.number_input("Monthly Income (₹)", min_value=0, value=50000, step=1000)
    inp_property_value = st.number_input("Property Value (₹)", min_value=0, value=1000000, step=5000)
with col3:
    inp_loan_amt = st.number_input("Loan Amount (₹) (example)", min_value=0, value=300000, step=1000)
    inp_interest = st.number_input("Interest Rate (%) (example)", min_value=0.1, value=10.0, step=0.1)
st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Sidebar UI A (3 groups) + workflow buttons
# ---------------------------
st.sidebar.title("📌 Applicant Information")

with st.sidebar.expander("🧍 Personal Information", expanded=True):
    gender = st.selectbox("Gender", ["Male","Female"])
    credit_worthiness = st.selectbox("Credit Worthiness", ["l1","l2"])
    age_bracket = st.selectbox("Age Bracket", ["<25","25-34","35-44","45-54","55-64","65-74",">74"])
    credit_type = st.selectbox("Credit Type", ["CRIF","EQUI","EXP","Other"])
    credit_score = st.slider("Credit Score", 300, 900, 650)

with st.sidebar.expander("💰 Financial Information", expanded=True):
    income = st.number_input("Monthly Income (₹)", min_value=0, value=50000, step=1000)
    property_value = st.number_input("Property Value (₹)", min_value=0, value=1000000, step=5000)
    dtir1 = st.number_input("Debt-to-Income Ratio (DTI)", min_value=0.0, value=15.0)
    LTV = st.number_input("Loan-to-Value Ratio (LTV)", min_value=0.0, value=40.0)

with st.sidebar.expander("📄 Loan Details", expanded=True):
    loan_type = st.selectbox("Loan Type", ["type1","type2","type3"])
    approv_in_adv = st.selectbox("Pre-approved?", ["pre","not_pre"])
    loan_purpose = st.selectbox("Loan Purpose", ["p1","p2","p3","p4"])
    submission_of_application = st.selectbox("Application Submitted To", ["to_inst","not_inst"])
    s_loan_amount = st.number_input("Loan Amount (₹)", min_value=0, value=300000, step=1000)
    rate_of_interest = st.number_input("Interest Rate (%)", min_value=0.1, value=10.0, step=0.1)
    term = st.number_input("Loan Term (months)", min_value=1, value=120, step=1)

st.sidebar.markdown("---")
preprocess_btn = st.sidebar.button("🔄 Preprocess Dataset")
train_btn = st.sidebar.button("⚙ Train Models")
model_options = [
    "Select Model", "Logistic Regression", "Decision Tree", "Random Forest",
    "Extra Trees", "Gradient Boosting", "KNN"
]
if XGBOOST_AVAILABLE:
    model_options.append("XGBoost")
selected_model = st.sidebar.selectbox("🎯 Choose Model", model_options)
predict_btn = st.sidebar.button("🚀 Predict Risk")
st.sidebar.markdown("---")
st.sidebar.caption("Order: Preprocess → Train → Choose Model → Predict")

# ---------------------------
# Dataset load + preview area
# ---------------------------
st.markdown("### 📘 Dataset")
df_loaded = load_local_dataset()
uploaded_df = None
if df_loaded is not None:
    st.success("📄 Loaded dataset from loan_dataset/Loan_Default.csv")
    df = df_loaded.copy()
else:
    st.warning("No dataset in loan_dataset/. Upload a CSV below.")
    uploaded = st.file_uploader("Upload Loan_Default.csv", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            uploaded_df = df.copy()
            st.success("Uploaded dataset successfully")
        except Exception as e:
            st.error("Upload failed: " + str(e))
            df = None
if 'df' in locals() and df is not None:
    st.write("Shape:", df.shape)
    if st.checkbox("Preview top 5 rows"):
        st.dataframe(df.head())

st.markdown("---")

# ---------------------------
# Preprocess handler
# ---------------------------
if preprocess_btn:
    if 'df' not in locals() or df is None:
        st.error("No dataset available. Place loan_dataset/Loan_Default.csv in repo or upload via sidebar.")
    else:
        if not IMB_AVAILABLE:
            st.warning("imbalanced-learn not installed — you can still preprocess but SMOTE will be skipped during training.")
        with st.spinner("🔄 Preprocessing dataset..."):
            df_proc = df.copy()

            # categorical columns from your dataset
            cat_cols = [
                "loan_limit", "Gender", "approv_in_adv", "loan_type", "loan_purpose",
                "Credit_Worthiness", "open_credit", "business_or_commercial",
                "Neg_ammortization", "interest_only", "lump_sum_payment",
                "construction_type", "occupancy_type", "Secured_by", "total_units",
                "credit_type", "co-applicant_credit_type", "age",
                "submission_of_application", "Region", "Security_Type"
            ]
            df_proc = safe_get_dummies(df_proc, cat_cols)

            # drop columns not needed
            drop_cols = ["ID", "year"]
            df_proc = drop_safe(df_proc, drop_cols)

            # fill numeric missing values
            num_cols = df_proc.select_dtypes(include=['int64','float64']).columns
            for c in num_cols:
                df_proc[c] = df_proc[c].fillna(df_proc[c].median())

            # target encode
            if "Status" not in df_proc.columns:
                st.error("Target column 'Status' not found in dataset.")
            else:
                le = LabelEncoder()
                df_proc["Status"] = le.fit_transform(df_proc["Status"].astype(str))
                df_proc = clean_colnames(df_proc)
                st.session_state["df_preprocessed"] = df_proc
                st.success("✅ Preprocessing completed.")

# ---------------------------
# Train handler
# ---------------------------
if train_btn:
    if "df_preprocessed" not in st.session_state:
        st.error("Please preprocess the dataset first.")
    else:
        df_train = st.session_state["df_preprocessed"].copy()
        with st.spinner("⚙ Training models..."):
            TARGET = "Status"
            X = df_train.drop(columns=[TARGET])
            y = df_train[TARGET]

            # train test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.20, random_state=42, stratify=y
            )

            # scale numeric columns
            scaler = StandardScaler()
            numeric_features = X_train.select_dtypes(include=[np.number]).columns
            X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
            X_test[numeric_features] = scaler.transform(X_test[numeric_features])

            # SMOTE (if available)
            if IMB_AVAILABLE:
                sm = SMOTE(random_state=42)
                X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
                X_train_sm = pd.DataFrame(X_train_sm, columns=X_train.columns)
            else:
                X_train_sm, y_train_sm = X_train, y_train

            # save scaler and test set for evaluation
            st.session_state["scaler"] = scaler
            st.session_state["X_test_scaled"] = X_test
            st.session_state["y_test"] = y_test

            # IMPORTANT: model_features: full training columns
            st.session_state["model_features"] = list(X_train_sm.columns)

            # define models
            models = {
                "Logistic Regression": LogisticRegression(max_iter=2000),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(n_estimators=300),
                "Extra Trees": ExtraTreesClassifier(n_estimators=300),
                "Gradient Boosting": GradientBoostingClassifier(n_estimators=300),
                "KNN": KNeighborsClassifier(n_neighbors=5)
            }
            if XGBOOST_AVAILABLE:
                models["XGBoost"] = xgb.XGBClassifier(eval_metric="logloss", use_label_encoder=False)

            trained = {}
            for name, mdl in models.items():
                mdl.fit(X_train_sm, y_train_sm)
                trained[name] = mdl

            st.session_state["trained_models"] = trained
            st.success("🔥 Training completed. Models are ready.")

# ---------------------------
# Predict handler (full feature alignment)
# ---------------------------
if predict_btn:
    if "trained_models" not in st.session_state:
        st.error("Please train the models first.")
    elif selected_model == "Select Model":
        st.error("Please choose a model to predict.")
    else:
        model = st.session_state["trained_models"][selected_model]
        features = st.session_state.get("model_features", None)
        scaler = st.session_state.get("scaler", None)

        if features is None or scaler is None:
            st.error("Model features or scaler not found. Retrain models.")
        else:
            st.info("🔍 Building input and predicting...")

            # Build mapping from UI fields to dataset column keywords
            input_map = {
                "Gender": gender,
                "Credit_Worthiness": credit_worthiness,
                "credit_type": credit_type,
                "age": age_bracket,
                "Credit_Score": credit_score,
                "income": income,
                "property_value": property_value,
                "dtir1": dtir1,
                "LTV": LTV,
                "loan_type": loan_type,
                "approv_in_adv": approv_in_adv,
                "loan_purpose": loan_purpose,
                "submission_of_application": submission_of_application,
                "loan_amount": s_loan_amount,
                "rate_of_interest": rate_of_interest,
                "term": term
            }

            # build full zero-filled DataFrame with all features
            X_input = pd.DataFrame(np.zeros((1, len(features))), columns=features)

            # fill features using build_feature_row_from_inputs rules
            X_input_filled = build_feature_row_from_inputs(features, input_map)

            # Safe numeric scaling (ensure all numeric cols present)
            num_cols = X_input_filled.select_dtypes(include=[np.number]).columns
            try:
                X_input_filled[num_cols] = scaler.transform(X_input_filled[num_cols])
            except Exception as e:
                # If transform fails, try reindexing columns to scaler's expected order
                try:
                    # scaler was fit on training numeric columns; this should match names
                    X_input_filled[num_cols] = scaler.transform(X_input_filled[num_cols].astype(float))
                except Exception:
                    st.warning("Scaling failed on input row; attempting prediction without scaled numeric adjustments.")

            # Predict
            try:
                probs = model.predict_proba(X_input_filled)[0]
                prob_default = float(probs[1]) * 100
                prob_non_default = float(probs[0]) * 100
                st.success("🎉 Prediction completed!")
            except Exception as e:
                st.error("Prediction failed: " + str(e))
                st.stop()

            # Display big result card
            st.markdown("### 📊 Risk Assessment Results")
            if prob_default < 40:
                st.markdown(f"""<div class="result-low">🟢 LOW RISK — Default Probability: {prob_default:.2f}%</div>""", unsafe_allow_html=True)
            elif prob_default < 70:
                st.markdown(f"""<div class="result-medium">🟡 MEDIUM RISK — Default Probability: {prob_default:.2f}%</div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="result-high">🔴 HIGH RISK — Default Probability: {prob_default:.2f}%</div>""", unsafe_allow_html=True)

            # three small probability cards
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"""<div class="small-card"><strong>Default</strong><br><span style="color:#d93737;font-size:20px">{prob_default:.2f}%</span></div>""", unsafe_allow_html=True)
            c2.markdown(f"""<div class="small-card"><strong>Non-Default</strong><br><span style="color:#0fa958;font-size:20px">{prob_non_default:.2f}%</span></div>""", unsafe_allow_html=True)
            # model confidence approximated as max(prob_default, prob_non_default)
            confidence = max(prob_default, prob_non_default)
            c3.markdown(f"""<div class="small-card"><strong>Model Confidence</strong><br><span style="color:#0b516d;font-size:20px">{confidence:.2f}%</span></div>""", unsafe_allow_html=True)

            st.progress(min(100, int(confidence)))

            # -----------------------------
            # ROC Curve and Confusion Matrix
            # -----------------------------
            st.markdown("### 📈 Model Evaluation")

            if "X_test_scaled" in st.session_state and "y_test" in st.session_state:
                X_test_eval = st.session_state["X_test_scaled"]
                y_test_eval = st.session_state["y_test"]

                try:
                    y_prob = model.predict_proba(X_test_eval)[:, 1]
                    auc_score = roc_auc_score(y_test_eval, y_prob)
                    fpr, tpr, _ = roc_curve(y_test_eval, y_prob)

                    st.markdown(f"**ROC Curve (AUC = {auc_score:.3f})**")
                    fig_roc = plt.figure(figsize=(6,4))
                    plt.plot(fpr, tpr, linewidth=2)
                    plt.plot([0,1],[0,1],'k--', linewidth=1)
                    plt.xlabel("False Positive Rate")
                    plt.ylabel("True Positive Rate")
                    plt.title(f"ROC Curve — {selected_model}")
                    st.pyplot(fig_roc)
                    plt.close(fig_roc)
                except Exception as e:
                    st.warning("Unable to plot ROC curve: " + str(e))

                # Confusion matrix
                try:
                    y_pred_test = model.predict(X_test_eval)
                    cm = confusion_matrix(y_test_eval, y_pred_test)
                    st.markdown("**Confusion Matrix**")
                    fig_cm = plt.figure(figsize=(4,3))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                                xticklabels=["No Default","Default"], yticklabels=["No Default","Default"])
                    plt.xlabel("Predicted")
                    plt.ylabel("Actual")
                    plt.title(f"Confusion Matrix — {selected_model}")
                    st.pyplot(fig_cm)
                    plt.close(fig_cm)
                except Exception as e:
                    st.warning("Unable to draw confusion matrix: " + str(e))
            else:
                st.info("Train models to enable ROC & Confusion Matrix outputs.")

# ---------------------------
# Developer / Debug area
# ---------------------------
with st.expander("Developer Info"):
    st.write("Session keys:", list(st.session_state.keys()))
    if "model_features" in st.session_state:
        st.write("Number of model features:", len(st.session_state["model_features"]))
        st.write(st.session_state["model_features"][:200])
    if "df" in locals() and df is not None:
        st.write("Raw dataset shape:", df.shape)
    if "df_preprocessed" in st.session_state:
        st.write("Preprocessed shape:", st.session_state["df_preprocessed"].shape)
