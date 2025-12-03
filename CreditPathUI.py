# streamlit_app.py

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    roc_auc_score, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from imblearn.over_sampling import SMOTE
import joblib

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

# ------------------- Page & Theme -------------------
if "busy" not in st.session_state:
    st.session_state["busy"] = False


st.set_page_config(
    page_title="Loan Default ML Studio",
    layout="wide",
)

st.markdown("""
<style>

    /* MAIN APP BACKGROUND */
    .stApp {
        background-color: #0d1117 !important;
        color: #e6e6e6 !important;
    }

    /* HEADERS */
    h1, h2, h3, h4, h5, h6 {
        color: #f0f6fc !important;
        font-weight: 700 !important;
    }

    /* GENERAL TEXT */
    p, span, div, label {
        color: #e6e6e6 !important;
    }

    /* SIDEBAR */
    [data-testid="stSidebar"] {
        background-color: #111827 !important;
        color: #e6e6e6 !important;
    }

    [data-testid="stSidebar"] * {
        color: #e6e6e6 !important;
    }

    /* METRIC CARDS */
    .stMetric, .stMetricLabel, .stMetricValue {
        color: #e6e6e6 !important;
    }

    /* DATAFRAME TABLE */
    table {
        color: #e6e6e6 !important;
        background-color: #161b22 !important;
    }

    /* CODE BLOCKS */
    pre, code {
        background-color: #161b22 !important;
        color: #e6e6e6 !important;
        border-radius: 6px;
        padding: 6px;
    }

    /* INPUT BOXES / SELECT / BUTTON BEAUTIFY */
    .stTextInput > div, .stSelectbox, .stFileUploader {
        background-color: #161b22 !important;
        color: #e6e6e6 !important;
        border-radius: 6px;
    }

    /* BUTTON STYLES */
    button[kind="primary"] {
        background-color: #1f6feb !important;
        color: white !important;
        border-radius: 8px !important;
        border: 1px solid #1f6feb !important;
    }

    /* EXPANDER */
    .streamlit-expanderHeader {
        background-color: #161b22 !important;
        color: #e6e6e6 !important;
    }

</style>
""", unsafe_allow_html=True)



# ------------------- Helpers -------------------

DATA_PATHS = ["data/Loan_Default.csv", "Loan_Default.csv"]

def load_dataset():
    for p in DATA_PATHS:
        if os.path.exists(p):
            df = pd.read_csv(p)
            return df, p
    st.error("Loan_Default.csv not found. Place it in ./data/ or current folder.")
    return None, None

def basic_clean(data: pd.DataFrame):
    # Drop duplicates
    data = data.drop_duplicates().reset_index(drop=True)

    # Simple missing handling
    for col in data.columns:
        if data[col].isnull().any():
            if pd.api.types.is_numeric_dtype(data[col]):
                data[col] = data[col].fillna(data[col].median())
            else:
                data[col] = data[col].fillna(data[col].mode()[0])
    return data

def preprocess(data: pd.DataFrame, target="Status"):
    # Schema (as in your notebook)
    schema = {
        'loan_amount': 'float',
        'rate_of_interest': 'float',
        'term': 'category',
        'Credit_Score': 'float',
        'income': 'float',
        'loan_purpose': 'category',
        'Credit_Worthiness': 'category',
        'Gender': 'category',
        'Status': 'category',
        'age': 'float',
        'LTV': 'float',
        'dtir1': 'float'
    }

    for col, dtype in schema.items():
        if col in data.columns:
            try:
                if dtype == "category":
                    data[col] = data[col].astype("category")
                else:
                    data[col] = pd.to_numeric(data[col], errors="coerce")
            except Exception:
                pass

    # Encode target if not numeric
    if not pd.api.types.is_numeric_dtype(data[target]):
        le = LabelEncoder()
        data[target] = le.fit_transform(data[target])
    else:
        le = None

    y = data[target]
    X = data.drop(columns=[target])

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_cat = pd.DataFrame(
        encoder.fit_transform(X[cat_cols]),
        columns=encoder.get_feature_names_out(cat_cols),
        index=X.index,
    )

    X_num = X[num_cols].reset_index(drop=True)
    X_final = pd.concat([X_num, X_cat], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Scale numeric cols
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    # Replace NaN
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    # SMOTE
    sm = SMOTE(random_state=42)
    X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)

    # Variance Threshold
    vt = VarianceThreshold(threshold=1e-4)
    X_train_vt = pd.DataFrame(vt.fit_transform(X_train_bal))
    X_test_vt = pd.DataFrame(vt.transform(X_test))

    # SelectKBest
    selector = SelectKBest(mutual_info_classif, k=min(40, X_train_vt.shape[1]))
    selector.fit(X_train_vt, y_train_bal)
    Xtr_sel = selector.transform(X_train_vt)
    Xte_sel = selector.transform(X_test_vt)

    artifacts = {
        "encoder": encoder,
        "scaler": scaler,
        "selector": selector,
        "vt": vt,
        "label_encoder": le,
        "feature_names": list(X_final.columns),
        "num_cols": num_cols,
        "cat_cols": cat_cols,
    }

    return Xtr_sel, Xte_sel, y_train_bal, y_test, artifacts


def get_metrics(y_true, y_pred, y_proba=None):
    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    else:
        specificity = np.nan

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Sensitivity": recall_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Specificity": specificity,
        "F1": f1_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba) if y_proba is not None else np.nan,
        "ConfusionMatrix": cm,
    }

def make_model_dict():
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
        "KNN": KNeighborsClassifier(5),
        "MLP (Neural Net)": MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", max_iter=500),
    }
    if XGB_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss"
        )
    return models

def plot_confusion(cm, title):
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    st.pyplot(fig)

# ------------------- Session Restore -------------------

# If we saved artifacts / models, load them (to survive refresh)
if "preprocessed" not in st.session_state and os.path.exists("preprocessed.pkl"):
    st.session_state["preprocessed"] = joblib.load("preprocessed.pkl")

if "trained" not in st.session_state and os.path.exists("trained_models.pkl"):
    saved = joblib.load("trained_models.pkl")
    st.session_state["trained"] = saved["trained"]
    st.session_state["metrics_df"] = saved["metrics_df"]
    st.session_state["conf_mats"] = saved["conf_mats"]
    st.session_state["trained_models"] = saved["trained_models"]

# ------------------- Sidebar -------------------

with st.sidebar:

    uploaded_file = st.file_uploader("📤 Upload your CSV dataset (optional)", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state["df"] = df
    
    st.markdown("**Steps:**")
    st.markdown("1. Preprocess data\n2. Select models\n3. Train\n4. Predict & compare")

    st.markdown("---")

    selected_models = st.multiselect(
    "Select Models",
    options=list(make_model_dict().keys()),
    default=[]
)

    btn_preprocess = st.button("🔄 Preprocess Data", disabled=st.session_state["busy"])
    btn_train     = st.button("⚙️ Train Models", disabled=st.session_state["busy"])
    btn_predict = st.button("📊 Predict & Compare", disabled=not st.session_state.get("trained", False))


# ------------------- Main Layout -------------------

st.title("📈 Loan Default Risk Prediction")
st.caption("Simple ML dashboard for predicting loan default risk using various models.")
st.markdown("""
<div style="
    background: linear-gradient(90deg, #0ea5e9, #2563eb);
    padding: 16px;
    border-radius: 10px;
    color: white;
    margin-bottom: 18px;
">
<h3>📌 Project: AI-Based Loan Default Prediction & Recovery Recommendation System</h3>
<p>
To design and develop a machine learning–driven platform that predicts borrower loan default risk and recommends personalized recovery actions.
The system uses open-source technologies to ensure cost-effectiveness, scalability, and reliability — empowering financial institutions and collection agents with actionable insights, improved delinquency recovery efficiency, and enhanced borrower optimization.
</p>
</div>
""", unsafe_allow_html=True)


# ---------- Load data and show basic info ----------

# ------------------- LOAD DATA -------------------
data = None
used_path = None

if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file)
        used_path = f"(User Uploaded) {uploaded_file.name}"
        st.success(f"📥 Using uploaded dataset: {uploaded_file.name}")
    except Exception:
        st.error("❌ Invalid CSV format. Please upload a valid dataset.")
else:
    data, used_path = load_dataset()
    st.info("📦 Using default dataset (Loan_Default.csv)")

if data is not None:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", len(data))
    with col2:
        st.metric("Total Columns", data.shape[1])
    with col3:
        if "Status" in data.columns:
            st.metric("Default Rate", f"{100 * data['Status'].value_counts(normalize=True).max():.1f}%")

    with st.expander("Peek at data", expanded=False):
        st.write(f"Loaded from: `{used_path}`")
        st.dataframe(data.head())

required_target = "Status"
if required_target not in data.columns:
    st.error(f"❗ Dataset must contain target column `{required_target}`")
    st.stop()

# ---------- Preprocessing step ----------

# ------------------- PREPROCESS STEP -------------------
if btn_preprocess and not st.session_state["busy"]:
    st.session_state["busy"] = True
    try:
        if data is None:
            st.error("⚠️ Dataset not found.")
        else:
            with st.spinner("🔄 Preprocessing data..."):
                data_clean = basic_clean(data)

                # user or default dataset both flow here
                Xtr_sel, Xte_sel, y_train_bal, y_test, artifacts = preprocess(data_clean)

                st.session_state["preprocessed"] = {
                    "Xtr_sel": Xtr_sel,
                    "Xte_sel": Xte_sel,
                    "y_train_bal": y_train_bal,
                    "y_test": y_test,
                    "artifacts": artifacts,
                    "dataset_name": used_path,
                }

                # ---------- Baseline Model to measure preprocessing ----------
                baseline = LogisticRegression(max_iter=1500)
                baseline.fit(Xtr_sel, y_train_bal)
                pred = baseline.predict(Xte_sel)

                st.session_state["preprocess_acc"] = accuracy_score(y_test, pred)
                st.session_state["preprocess_f1"] = f1_score(y_test, pred)

            st.success("✨ Preprocessing completed!")
    finally:
        st.session_state["busy"] = False
    if "preprocess_acc" in st.session_state:
        st.markdown("### ⚙️ Preprocessing Performance (Baseline Model)")
        st.metric("Accuracy", round(st.session_state["preprocess_acc"], 3))
        st.metric("F1 Score", round(st.session_state["preprocess_f1"], 3))

# Show preview of preprocessing results

    pp = st.session_state["preprocessed"]

    st.markdown("### 🧠 Preprocessing Summary")

    col1, col2, col3 = st.columns(3)
    col1.metric("Train Rows", len(pp["y_train_bal"]))
    col2.metric("Test Rows", len(pp["y_test"]))
    col3.metric("Features", pp["Xtr_sel"].shape[1])

    st.caption(f"📦 Dataset Source: **{pp['dataset_name']}**")
    st.markdown(f"📂 **Current Dataset:** `{used_path}`")


# ---------- Training step ----------

# ------------------- TRAINING STEP -------------------
if btn_train and not st.session_state["busy"]:

    if "preprocessed" not in st.session_state:
        st.error("⚠️ Please preprocess the data first.")
    elif len(selected_models) == 0:
        st.error("⚠️ Please select at least one model.")
    else:
        st.session_state["busy"] = True
        try:
            pp = st.session_state["preprocessed"]
            Xtr_sel = pp["Xtr_sel"]
            Xte_sel = pp["Xte_sel"]
            y_train_bal = pp["y_train_bal"]
            y_test = pp["y_test"]

            all_models = make_model_dict()
            trained_models = {}
            conf_mats = {}

            train_rows = []
            test_rows = []

            with st.spinner("⚙️ Training models..."):
                for name in selected_models:
                    model = all_models[name]
                    model.fit(Xtr_sel, y_train_bal)

                    # ---------------- TRAIN METRICS ----------------
                    y_pred_tr = model.predict(Xtr_sel)
                    y_proba_tr = model.predict_proba(Xtr_sel)[:,1] if hasattr(model,'predict_proba') else None
                    mtr = get_metrics(y_train_bal, y_pred_tr, y_proba_tr)

                    train_rows.append({
                        "Model": name,
                        "Accuracy": mtr["Accuracy"],
                        "Precision": mtr["Precision"],
                        "Sensitivity": mtr["Sensitivity"],
                        "Specificity": mtr["Specificity"],
                        "F1": mtr["F1"],
                        "AUC": mtr["AUC"],
                    })

                    # ---------------- TEST METRICS ----------------
                    y_pred_te = model.predict(Xte_sel)
                    y_proba_te = model.predict_proba(Xte_sel)[:,1] if hasattr(model,'predict_proba') else None
                    mte = get_metrics(y_test, y_pred_te, y_proba_te)

                    test_rows.append({
                        "Model": name,
                        "Accuracy": mte["Accuracy"],
                        "Precision": mte["Precision"],
                        "Sensitivity": mte["Sensitivity"],
                        "Specificity": mte["Specificity"],
                        "F1": mte["F1"],
                        "AUC": mte["AUC"],
                    })

                    conf_mats[name] = mte["ConfusionMatrix"]
                    trained_models[name] = model

            # SAVE TO SESSION 👇 MOST IMPORTANT
            st.session_state["train_table"] = pd.DataFrame(train_rows)
            st.session_state["test_table"] = pd.DataFrame(test_rows)
            st.session_state["trained_models"] = trained_models
            st.session_state["conf_mats"] = conf_mats
            st.session_state["trained"] = True

            st.success("🎯 Training Complete!")

        finally:
            st.session_state["busy"] = False




        st.markdown("### 📚 Training Results Overview")

        metrics_df = st.session_state["metrics_df"]
        st.dataframe(metrics_df.round(3), use_container_width=True)

        best_row = metrics_df.sort_values("Accuracy", ascending=False).iloc[0]

        st.markdown(f"""
🎯 **Best Model so far:**  
💡 **{best_row['Model']}**  
🏆 **Accuracy:** `{best_row['Accuracy']:.3f}`
        """)


# ---------- Prediction / Comparison step ----------
# ------------------- PREDICT & COMPARE -------------------
if btn_predict:

    if "train_table" not in st.session_state or "test_table" not in st.session_state:
        st.error("⚠️ Train models first.")
    else:
        st.subheader("📊 Model Performance Overview")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🏋️ Training Metrics")
            st.dataframe(st.session_state["train_table"].round(3), use_container_width=True)

        with col2:
            st.markdown("### 🧪 Testing Metrics")
            st.dataframe(st.session_state["test_table"].round(3), use_container_width=True)

# ====================================
#  Overfit / Underfit Risk Analysis
# ====================================

        df_compare = st.session_state["train_table"].merge(
    st.session_state["test_table"],
    on="Model",
    suffixes=("_Train", "_Test")
        )

        df_compare["Accuracy_Gap"] = df_compare["Accuracy_Train"] - df_compare["Accuracy_Test"]
        df_compare["F1_Gap"] = df_compare["F1_Train"] - df_compare["F1_Test"]

        st.markdown("### ⚠️ Model Generalization Report")
        st.dataframe(df_compare.round(3), use_container_width=True)




        # ------------------- Smaller Bar Plot -------------------
    metrics_long = df_compare.melt(id_vars="Model", var_name="Metric", value_name="Score")

    fig, ax = plt.subplots(figsize=(8, 4))  # smaller
    sns.barplot(data=metrics_long, x="Model", y="Score", hue="Metric", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Metrics Comparison")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("### Confusion Matrices")

    conf_mats = st.session_state["conf_mats"]
    names = list(conf_mats.keys())

    for i in range(0, len(names), 2):
            row = names[i: i + 2]            
            cm1 = conf_mats[row[0]]
            cm2 = conf_mats[row[1]] if len(row) > 1 else None
            name1 = row[0]
            name2 = row[1] if len(row) > 1 else None

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**{name1}**")
                fig, ax = plt.subplots(figsize=(4,3))
                sns.heatmap(cm1, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                ax.set_title(name1)
                plt.tight_layout()
                st.pyplot(fig)

            with col2:
                if cm2 is not None:
                    st.markdown(f"**{name2}**")
                    fig, ax = plt.subplots(figsize=(4,3))
                    sns.heatmap(cm2, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("Actual")
                    ax.set_title(name2)
                    plt.tight_layout()
                    st.pyplot(fig)

            st.markdown("---")
        

        # ------------------- Quick Prediction -------------------
    st.subheader("Quick Prediction Demo (Random Test Row)")
    st.markdown("---")
    st.subheader("📌 Model Evaluation Summary")

    best_row = st.session_state["metrics_df"].sort_values("Accuracy", ascending=False).iloc[0]
    st.markdown(f"""
### 🏅 Best Performing Model

- **Model:** `{best_row["Model"]}`
- **Accuracy:** `{best_row["Accuracy"]:.3f}`
- **F1 Score:** `{best_row["F1"]:.3f}`
- **Sensitivity (Recall):** `{best_row["Sensitivity"]:.3f}`
- **Precision:** `{best_row["Precision"]:.3f}`
- **Specificity:** `{best_row["Specificity"]:.3f}`
- **AUC:** `{best_row["AUC"]:.3f}`
""")


    if "preprocessed" in st.session_state and "trained_models" in st.session_state:
            Xte_sel = st.session_state["preprocessed"]["Xte_sel"]
            y_test = st.session_state["preprocessed"]["y_test"]
            idx = np.random.randint(0, Xte_sel.shape[0])

            col_a, col_b = st.columns(2)
            with col_a:
                st.write(f"Random test index: {idx}")
                st.write(f"True label: {y_test.iloc[idx]}")

            with col_b:
                preds = {}
                for name, model in st.session_state["trained_models"].items():
                    pred = model.predict(Xte_sel[idx: idx + 1])[0]
                    preds[name] = int(pred)
                st.write("Model predictions:")
                st.json(preds)

        # ----------------------------------------------------------
        #             ROC + Precision Recall Comparison
        # ----------------------------------------------------------
    st.markdown("---")
    st.subheader("📈 ROC & Precision-Recall Comparison")

    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

    models = st.session_state["trained_models"]

        # ---------- ROC ----------
    fig3, ax3 = plt.subplots(figsize=(6,4))
    for name, model in models.items():
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(Xte_sel)[:, 1]
            elif hasattr(model, "decision_function"):
                y_score = model.decision_function(Xte_sel)
            else:
                continue
            fpr, tpr, _ = roc_curve(y_test, y_score)
            roc_auc = auc(fpr, tpr)
            ax3.plot(fpr, tpr, lw=2, label=f"{name} (AUC={roc_auc:.3f})")

    ax3.plot([0, 1], [0, 1], "k--", lw=1)
    ax3.set_title("ROC Curve")
    ax3.set_xlabel("False Positive Rate")
    ax3.set_ylabel("True Positive Rate")
    ax3.legend(loc="lower right")
    plt.tight_layout()
        # ---------- Precision–Recall ----------
    fig4, ax4 = plt.subplots(figsize=(6,4))
    for name, model in models.items():
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(Xte_sel)[:, 1]
            elif hasattr(model, "decision_function"):
                y_score = model.decision_function(Xte_sel)
            else:
                continue
            precision, recall, _ = precision_recall_curve(y_test, y_score)
            ap = average_precision_score(y_test, y_score)
            ax4.plot(recall, precision, lw=2, label=f"{name} (AP={ap:.3f})")

    ax4.set_title("Precision–Recall Curve")
    ax4.set_xlabel("Recall")
    ax4.set_ylabel("Precision")
    ax4.legend(loc="lower left")
    plt.tight_layout()
    colROC, colPR = st.columns(2)
    with colROC:
        st.pyplot(fig3)
    with colPR:
        st.pyplot(fig4)
# ---------- Save best model (for later use) ----------

if "metrics_df" in st.session_state and "trained_models" in st.session_state:
    best_row = st.session_state["metrics_df"].sort_values("Accuracy", ascending=False).iloc[0]
    best_model_name = best_row["Model"]
    best_model = st.session_state["trained_models"][best_model_name]

    if "preprocessed" in st.session_state:
        payload = {
            "encoder": st.session_state["preprocessed"]["artifacts"]["encoder"],
            "scaler": st.session_state["preprocessed"]["artifacts"]["scaler"],
            "selector": st.session_state["preprocessed"]["artifacts"]["selector"],
            "model": best_model,
            "feature_names": st.session_state["preprocessed"]["artifacts"]["feature_names"],
            "label_encoder": st.session_state["preprocessed"]["artifacts"]["label_encoder"],
        }
        joblib.dump(payload, "PurviModel.joblib")

        st.info(f"Best model right now: **{best_model_name}** (saved as PurviModel.joblib)")

    st.markdown("""<style>
.block-container {
    padding-top: 1.3rem;
    padding-left: 2.2rem;
    padding-right: 2.2rem;
}

.dataframe tbody tr:hover {
    background-color: #1e293b !important;
}
</style>

""", unsafe_allow_html=True)