import os
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from imblearn.over_sampling import SMOTE

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except:
    XGB_AVAILABLE = False

st.set_page_config(page_title="Loan Default Prediction", layout="wide")
st.title("Loan Default Prediction — ML Studio")

DATA_PATHS = [
    "Loan_Default.csv",
    "data/Loan_Default.csv",
    "Loan_default.csv",
    "data/Loan_default.csv"
]

def get_data_path():
    for p in DATA_PATHS:
        if os.path.exists(p):
            return p
    return None

def basic_clean(df):
    df = df.drop_duplicates().reset_index(drop=True)
    for c in df.columns:
        if df[c].isnull().any():
            if pd.api.types.is_numeric_dtype(df[c]):
                df[c] = df[c].fillna(df[c].median())
            else:
                df[c] = df[c].fillna(df[c].mode().iat[0])
    return df

def compute_metrics(y_true, y_pred, y_proba=None):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = (cm.ravel() if cm.size == 4 else (np.nan,)*4)
    specificity = tn / (tn + fp) if not np.isnan(tn) else np.nan
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Sensitivity": recall_score(y_true, y_pred),
        "Specificity": specificity,
        "Precision": precision_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba) if y_proba is not None else np.nan,
        "ConfusionMatrix": cm
    }

def make_models():
    models = {
        "Logistic Regression": LogisticRegression(max_iter=500),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42),
        "KNN": KNeighborsClassifier()
    }
    if XGB_AVAILABLE:
        models["XGBoost"] = XGBClassifier(eval_metric='logloss')
    return models

with st.sidebar:
    st.header("Actions")
    st.write("1. Preprocess → 2. Train → 3. Compare → 4. Predict")

    btn_preprocess = st.button("Preprocess Data")

    selected_models = st.multiselect(
        "Select models to train:",
        options=list(make_models().keys()),
        default=["Logistic Regression", "Random Forest"]
    )

    btn_train = st.button("Train Models")
    btn_predict = st.button("Quick Prediction")

DATA_PATH = get_data_path()
if DATA_PATH is None:
    st.error("Dataset not found. Place Loan_Default.csv in project folder.")
    st.stop()

df = pd.read_csv(DATA_PATH)
st.success(f"Dataset loaded from: {DATA_PATH}")

with st.expander("Preview Dataset"):
    st.dataframe(df.head())

if btn_preprocess:
    with st.spinner("Processing dataset..."):
        df_clean = basic_clean(df.copy())

        if "Loan_Default" not in df_clean.columns:
            df_clean["risk_score"] = 0
            df_clean["risk_score"] += (df_clean["LTV"] > 80).astype(int)
            if "Interest_rate_spread" in df_clean.columns:
                df_clean["risk_score"] += (df_clean["Interest_rate_spread"] > 0.5).astype(int)
            if "Upfront_charges" in df_clean.columns:
                df_clean["risk_score"] += (df_clean["Upfront_charges"] > 0.6).astype(int)
            if "approv_in_adv" in df_clean.columns:
                df_clean["risk_score"] += (df_clean["approv_in_adv"] == "N").astype(int)
            if "Neg_ammortization" in df_clean.columns:
                df_clean["risk_score"] += (df_clean["Neg_ammortization"] == "Y").astype(int)
            df_clean["Loan_Default"] = (df_clean["risk_score"] >= 2).astype(int)
            df_clean.drop(columns=["risk_score"], inplace=True)

        y = df_clean["Loan_Default"]
        X = df_clean.drop(columns=["Loan_Default"])

        num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

        if len(cat_cols) > 0:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            X_cat = pd.DataFrame(
                encoder.fit_transform(X[cat_cols]),
                columns=encoder.get_feature_names_out(cat_cols),
                index=X.index
            )
            X = pd.concat([X[num_cols].reset_index(drop=True),
                           X_cat.reset_index(drop=True)], axis=1)
        else:
            encoder = None

        scaler = StandardScaler()
        if len(num_cols) > 0:
            X[num_cols] = scaler.fit_transform(X[num_cols])

        X = X.fillna(0)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        sm = SMOTE(random_state=42)
        X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)

        st.session_state["preproc"] = {
            "encoder": encoder,
            "scaler": scaler,
            "num_cols": num_cols,
            "cat_cols": cat_cols,
            "feature_names": X.columns.tolist(),
            "X_train_bal": X_train_bal,
            "y_train_bal": y_train_bal,
            "X_test": X_test,
            "y_test": y_test
        }

        st.success("Preprocessing complete.")

if btn_train:
    if "preproc" not in st.session_state:
        st.error("Preprocess the data first.")
        st.stop()

    with st.spinner("Training models..."):
        pre = st.session_state["preproc"]

        Xtr = pre["X_train_bal"]
        ytr = pre["y_train_bal"]
        Xte = pre["X_test"]
        yte = pre["y_test"]

        trained = {}
        rows = []
        confs = {}
        models = make_models()

        for name in selected_models:
            model = models[name]
            model.fit(Xtr, ytr)

            y_pred = model.predict(Xte)
            y_proba = model.predict_proba(Xte)[:, 1] if hasattr(model, "predict_proba") else None

            metrics = compute_metrics(yte, y_pred, y_proba)
            confs[name] = metrics["ConfusionMatrix"]

            rows.append({
                "Model": name,
                "Accuracy": round(metrics["Accuracy"]*100, 3),
                "Sensitivity": round(metrics["Sensitivity"]*100, 3),
                "Specificity": round(metrics["Specificity"]*100, 3),
                "Precision": round(metrics["Precision"]*100, 3),
                "F1": round(metrics["F1"]*100, 3),
                "AUC": round(metrics["AUC"], 4)
            })

            trained[name] = model

        metrics_df = pd.DataFrame(rows).set_index("Model")

        metrics_df_style = metrics_df.style.highlight_max(color="#c6efce", axis=0)

        st.session_state["trained_models"] = trained
        st.session_state["metrics_df"] = metrics_df
        st.session_state["styled_df"] = metrics_df_style
        st.session_state["conf_mats"] = confs

        joblib.dump({"trained_models": trained,
                     "metrics_df": metrics_df,
                     "conf_mats": confs}, "models.joblib")

        st.success("Models trained and saved.")

if "styled_df" in st.session_state:
    st.subheader("Model Performance Table")
    st.write(st.session_state["styled_df"])

if btn_predict:
    if "preproc" not in st.session_state:
        st.warning("Preprocess the data first.")
        st.stop()

    pre = st.session_state["preproc"]

    st.subheader("Quick Prediction")

    idx = np.random.randint(0, pre["X_test"].shape[0])
    sample = pre["X_test"].iloc[[idx]]

    st.write("Random Test Input:")
    st.dataframe(sample)

    best_model = st.session_state["metrics_df"]["AUC"].idxmax()
    model = st.session_state["trained_models"][best_model]

    pred = model.predict(sample)[0]
    proba = model.predict_proba(sample)[0][1]

    st.info(f"Best Model: {best_model}")
    st.success(f"Prediction: {'DEFAULT' if pred==1 else 'NON-DEFAULT'}")
    st.info(f"Probability of Default: {proba:.3f}")