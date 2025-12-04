# app.py
import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier)
from sklearn.neighbors import KNeighborsClassifier

# optional xgboost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

# imbalanced-learn (SMOTE)
try:
    from imblearn.over_sampling import SMOTE
    IMB_AVAILABLE = True
except Exception:
    IMB_AVAILABLE = False

# -------------------------
# Helper UI & Utility funcs
# -------------------------
def clear_session():
    for k in ["df", "df_preprocessed", "scaler", "trained_models", "X_columns", "model_features"]:
        if k in st.session_state:
            del st.session_state[k]

def load_dataset_from_path(path="loan_dataset/Loan_Default.csv"):
    try:
        df_local = pd.read_csv(path)
        return df_local
    except Exception as e:
        return None

def safe_get_dummies(df, cat_cols):
    present = [c for c in cat_cols if c in df.columns]
    if present:
        return pd.get_dummies(df, columns=present, drop_first=True)
    return df

def drop_safe(df, cols):
    return df.drop(columns=[c for c in cols if c in df.columns], errors="ignore")

def clean_colnames(df):
    df.columns = (df.columns
                  .str.replace("[", "", regex=False)
                  .str.replace("]", "", regex=False)
                  .str.replace("<", "", regex=False)
                  .str.replace(">", "", regex=False)
                  .str.replace(" ", "_", regex=False))
    return df

# -------------------------
# App UI
# -------------------------
st.set_page_config(page_title="CreditPathAI — Loan Default Predictor", layout="wide")
st.title("🏦 CreditPathAI — Loan Default Risk Prediction and Recovery Recommendations")
st.markdown(
    "Follow steps: **(1)** Provide applicant details, **(2)** Load dataset (from repo or upload), "
    "**(3)** Preprocess, **(4)** Train/Prepare models, **(5)** Select model & Predict."
)
st.write("---")

# ==========================
# Main applicant input area
# ==========================
st.subheader("👤 Applicant Details (use these values to test predictions)")

col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age (Years)", min_value=18, max_value=90, value=30)
    annual_income = st.number_input("Annual Income (₹)", min_value=0, step=1000, value=300000)
with col2:
    loan_amount = st.number_input("Loan Amount (₹)", min_value=0, step=1000, value=200000)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
with col3:
    interest_rate = st.slider("Interest Rate (%)", min_value=0.1, max_value=40.0, value=10.0)
    tenure = st.slider("Loan Tenure (Years)", min_value=1, max_value=60, value=10)

st.write("---")

# ==========================
# Sidebar workflow
# ==========================
st.sidebar.title("⚙ Workflow — Data ▶ Preprocess ▶ Models ▶ Predict")

# 1) Try to load dataset from repo path first (no upload)
dataset_path = "datasets/Loan_Default.csv"
df_local = load_dataset_from_path(dataset_path)
if df_local is not None:
    st.sidebar.success(f"📄 Loaded dataset from `{dataset_path}` (shape: {df_local.shape})")
    st.session_state["df"] = df_local
else:
    st.sidebar.info("No local dataset found at `datasets/Loan_Default.csv`.")
    uploaded = st.sidebar.file_uploader("Or upload a smaller CSV (if your dataset is huge, place it in datasets/)", type=["csv"])
    if uploaded is not None:
        try:
            st.session_state["df"] = pd.read_csv(uploaded)
            st.sidebar.success(f"Uploaded dataset (shape: {st.session_state['df'].shape})")
        except Exception as e:
            st.sidebar.error("Could not read uploaded CSV. Error: " + str(e))

# show quick dataset preview if available
if "df" in st.session_state:
    if st.sidebar.checkbox("Preview dataset (first 5 rows)"):
        st.subheader("📋 Dataset Preview")
        st.dataframe(st.session_state["df"].head())

# show warning if imblearn not available
if not IMB_AVAILABLE:
    st.sidebar.error("`imbalanced-learn` (SMOTE) not installed. Run: pip install imbalanced-learn in this environment.")
    st.sidebar.markdown("---")

# Model selection (always visible but will be enabled after training)
# --- Correct Workflow Order ---

# 1️⃣ Preprocess dataset
preprocess_btn = st.sidebar.button("🔄 Preprocess Dataset")

# 2️⃣ Train models
train_btn = st.sidebar.button("⚙ Train Models")

# 3️⃣ Choose model
model_options = ["Select Model", "Logistic Regression", "Decision Tree", "Random Forest",
                 "Extra Trees", "Gradient Boosting", "KNN"]
if XGBOOST_AVAILABLE:
    model_options.append("XGBoost")

selected_model = st.sidebar.selectbox("🤖 Choose Model", model_options)

# 4️⃣ Predict
predict_btn = st.sidebar.button("🔍 Predict Default Risk")

# ==========================
# Preprocessing logic
# ==========================
if preprocess_btn:
    if "df" not in st.session_state:
        st.error("No dataset available. Upload or place `Loan_Default.csv` into `datasets/`.")
    elif not IMB_AVAILABLE:
        st.error("SMOTE not available. Install `imbalanced-learn` to continue.")
    else:
        with st.spinner("Preprocessing dataset..."):
            df = st.session_state["df"].copy()
            # --- categorical columns (from your notebook) ---
            cat_cols = [
                "loan_limit","Gender","approv_in_adv","loan_type","loan_purpose",
                "Credit_Worthiness","open_credit","business_or_commercial",
                "Neg_ammortization","interest_only","lump_sum_payment",
                "construction_type","occupancy_type","Secured_by","total_units",
                "credit_type","co-applicant_credit_type","age",
                "submission_of_application","Region","Security_Type"
            ]
            df = safe_get_dummies(df, cat_cols)

            drop_cols = [
                'ID','loan_limit_ncf','approv_in_adv_pre','loan_type_type2','loan_type_type3',
                'loan_purpose_p2','loan_purpose_p3','loan_purpose_p4','Credit_Worthiness_l2',
                'open_credit_opc','business_or_commercial_nob/c','Neg_ammortization_not_neg',
                'interest_only_not_int','lump_sum_payment_not_lpsm','construction_type_sb',
                'occupancy_type_pr','occupancy_type_sr','Secured_by_land','total_units_2U',
                'total_units_3U','total_units_4U','credit_type_CRIF','credit_type_EQUI',
                'credit_type_EXP','co-applicant_credit_type_EXP','age_35-44','age_45-54',
                'age_55-64','age_65-74','age_<25','age_>74','submission_of_application_to_inst',
                'Region_North-East','Region_central','Region_south','Security_Type_direct'
            ]
            df = drop_safe(df, drop_cols)

            # ensure TARGET exists
            TARGET = "Status"
            if TARGET not in df.columns:
                st.error(f"Target column `{TARGET}` not found in dataset. Please verify.")
            else:
                # fill numeric missing with median
                num_cols = df.select_dtypes(include=["int64","float64"]).columns
                for c in num_cols:
                    df[c] = df[c].fillna(df[c].median())

                # label encode target
                le = LabelEncoder()
                df[TARGET] = le.fit_transform(df[TARGET].astype(str))

                # Save preprocessed df in session
                st.session_state["df_preprocessed"] = clean_colnames(df)
                st.success("✅ Preprocessing completed. You can now Train models.")

# ==========================
# Model training logic
# ==========================
if train_btn:
    if "df_preprocessed" not in st.session_state:
        st.error("Please preprocess the dataset first.")
    else:
        df = st.session_state["df_preprocessed"].copy()
        with st.spinner("Preparing data, scaling and SMOTE..."):
            TARGET = "Status"
            X = df.drop(columns=[TARGET])
            y = df[TARGET]

            # train/test split (we only need to fit scaler and SMOTE on train)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

            scaler = StandardScaler()
            numeric_features = X_train.select_dtypes(include=[np.number]).columns
            X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
            X_test[numeric_features] = scaler.transform(X_test[numeric_features])

            # SMOTE (on training)
            if not IMB_AVAILABLE:
                st.error("imbalanced-learn (SMOTE) not installed — cannot continue training.")
            else:
                sm = SMOTE(random_state=42)
                X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
                # keep cleaned column names
                X_train_sm = pd.DataFrame(X_train_sm, columns=X_train.columns)
                X_test = pd.DataFrame(X_test, columns=X_test.columns)

                # Save scaler and columns to session for predictions
                st.session_state["scaler"] = scaler
                st.session_state["X_columns"] = list(X_train_sm.columns)

                # Define models
                models = {
                    "Logistic Regression": LogisticRegression(max_iter=2000),
                    "Decision Tree": DecisionTreeClassifier(),
                    "Random Forest": RandomForestClassifier(n_estimators=200),
                    "Extra Trees": ExtraTreesClassifier(n_estimators=200),
                    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200),
                    "KNN": KNeighborsClassifier(n_neighbors=5),
                }
                if XGBOOST_AVAILABLE:
                    models["XGBoost"] = xgb.XGBClassifier(eval_metric="logloss", use_label_encoder=False)

                trained = {}
                for name, mdl in models.items():
                    with st.spinner(f"Training {name} ..."):
                        mdl.fit(X_train_sm, y_train_sm)
                        trained[name] = mdl

                st.session_state["trained_models"] = trained
                st.session_state["model_features"] = st.session_state["X_columns"]
                st.success("🔥 All models trained and ready. Choose a model and press Predict.")

# ==========================
# Prediction logic
# ==========================
if predict_btn:
    # checks
    if "trained_models" not in st.session_state:
        st.error("Models not trained yet. Preprocess -> Train first.")
    else:
        trained = st.session_state["trained_models"]
        if selected_model == "Select Model":
            st.error("Please select a model from the sidebar.")
        else:
            model = trained.get(selected_model)
            if model is None:
                st.error(f"Model `{selected_model}` not found (maybe training didn't include it).")
            else:
                # Build input vector aligned to trained columns
                features = st.session_state.get("model_features", None)
                if not features:
                    st.error("No feature list found from training. Re-run training.")
                else:
                    # create zero DataFrame with model features
                    X_input = pd.DataFrame(np.zeros((1, len(features))), columns=features)

                    # map applicant inputs to likely column names if present
                    mapping = {
                        "age": age,
                        "Annual_Income": annual_income,
                        "Loan_Amt": loan_amount,
                        "Credit": credit_score,
                        "Interest": interest_rate,
                        "Tenure": tenure
                    }
                    # attempt to set values for exact matches
                    for k, v in mapping.items():
                        if k in X_input.columns:
                            X_input.loc[0, k] = v
                        else:
                            # try lowercase/variants
                            for col in X_input.columns:
                                if col.lower() == k.lower() or k.lower() in col.lower():
                                    X_input.loc[0, col] = v
                                    break

                    # scale numeric cols using stored scaler
                    scaler = st.session_state.get("scaler", None)
                    if scaler is not None:
                        num_cols = X_input.select_dtypes(include=[np.number]).columns.intersection(st.session_state["X_columns"])
                        try:
                            X_input[num_cols] = scaler.transform(X_input[num_cols])
                        except Exception:
                            # if scaler expects more columns, try filling missing with zeros and transform
                            pass

                    # predict
                    try:
                        prob = model.predict_proba(X_input)[0][1]
                        pred = model.predict(X_input)[0]
                    except Exception as e:
                        st.error("Prediction failed. The app could not align user inputs to training features. Error: " + str(e))
                        st.info("Suggestion: Re-check model features or display `model_features` for debugging.")
                        st.write("Model features count:", len(st.session_state.get("model_features", [])))
                        st.stop()

                    # display results
                    st.subheader("📈 Prediction Result")
                    pct = prob * 100
                    if pct >= 70:
                        st.error(f"🔴 HIGH RISK — Probability of default: {pct:.2f}%")
                        st.write("Recommendation: Immediate follow-up, legal notice, restructure/settlement options.")
                    elif pct >= 40:
                        st.warning(f"🟡 MEDIUM RISK — Probability of default: {pct:.2f}%")
                        st.write("Recommendation: Frequent reminders, EMI rescheduling, close monitoring.")
                    else:
                        st.success(f"🟢 LOW RISK — Probability of default: {pct:.2f}%")
                        st.write("Recommendation: Approve or monitor normally; consider promotions/discounts.")

                    # show probability bar
                    st.progress(min(100, int(pct)))

# ==========================
# Developer debug / show trained features
# ==========================
with st.expander("Debug / Model Feature Info (for developer)"):
    if "model_features" in st.session_state:
        st.write(f"Trained model features ({len(st.session_state['model_features'])}):")
        st.write(st.session_state["model_features"][:150])
    else:
        st.write("Model features not available yet. Preprocess & Train first.")
