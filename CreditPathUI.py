# streamlit_app.py
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve, auc
)
import shap
# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# optional xgboost
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

# optional SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

st.set_page_config(page_title="CreditPathAI — Demo (Mock Data)", layout="wide")

# ---------------------------
# Helpers & small mock dataset
# ---------------------------

def build_mock_dataset():
    """
    Build a small 20-row mock dataset using 7 core numeric features used in UI.
    Target: 1 = default, 0 = no default.
    This dataset is purposely small for demo and instant training.
    """
    rng = np.random.RandomState(42)
    n = 200  # use 200 rows for slightly better metrics but still fast
    age = rng.randint(20, 75, size=n)
    income = rng.randint(1000, 150000, size=n)
    loan_amount = rng.randint(5000, 500000, size=n)
    credit_score = rng.randint(300, 850, size=n)
    employment_years = rng.randint(0, 40, size=n)
    num_delinquencies = rng.poisson(0.8, size=n)
    debt_to_income_ratio = np.round(loan_amount / (income + 1), 2)

    # create a simple rule-based target (for demo)
    risk = (
        (credit_score < 600).astype(int) * 2 +
        (debt_to_income_ratio > 0.5).astype(int) * 2 +
        (num_delinquencies > 1).astype(int) * 1 +
        (employment_years < 1).astype(int) * 1
    )
    # convert to binary target (>=2 risk points -> default)
    target = (risk >= 2).astype(int)

    df = pd.DataFrame({
        "age": age,
        "income": income,
        "loan_amount": loan_amount,
        "credit_score": credit_score,
        "employment_years": employment_years,
        "num_delinquencies": num_delinquencies,
        "debt_to_income_ratio": debt_to_income_ratio,
        "target": target
    })
    return df

@st.cache_data
def get_demo_data():
    return build_mock_dataset()

def get_model_by_name(name):
    name = name.lower()
    if "logistic" in name:
        return LogisticRegression(max_iter=1000)
    if "decision" in name:
        return DecisionTreeClassifier(random_state=42)
    if "random" in name:
        return RandomForestClassifier(n_estimators=200, random_state=42)
    if "gaussian" in name or "naive" in name:
        return GaussianNB()
    if "knn" in name or "nearest" in name:
        return KNeighborsClassifier(n_neighbors=5)
    if "xgb" in name or "xgboost" in name:
        if XGB_AVAILABLE:
            return XGBClassifier(use_label_encoder=False, eval_metric="logloss", verbosity=0)
        else:
            raise RuntimeError("XGBoost not available in this environment.")
    raise ValueError("Unknown model")

def compute_metrics(y_true, y_pred, y_prob=None):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    aucv = roc_auc_score(y_true, y_prob) if (y_prob is not None and len(np.unique(y_true))>1) else np.nan
    return {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "AUC": aucv}

def plot_confusion_small(cm, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(3.2, 2.6))  # small size
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                annot_kws={"size":10})
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("Actual", fontsize=9)
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    return fig

def plot_roc_pr(y_test, y_score, name="Model"):
    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    fig_roc, ax_roc = plt.subplots(figsize=(4,3))
    ax_roc.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    ax_roc.plot([0,1],[0,1],"k--", lw=0.8)
    ax_roc.set_title(f"ROC — {name}", fontsize=10)
    ax_roc.set_xlabel("FPR", fontsize=9)
    ax_roc.set_ylabel("TPR", fontsize=9)
    ax_roc.legend(fontsize=8)
    plt.tight_layout()

    # PR
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    ap = auc(recall, precision)
    fig_pr, ax_pr = plt.subplots(figsize=(4,3))
    ax_pr.plot(recall, precision, label=f"AP={ap:.3f}")
    ax_pr.set_title(f"Precision–Recall — {name}", fontsize=10)
    ax_pr.set_xlabel("Recall", fontsize=9)
    ax_pr.set_ylabel("Precision", fontsize=9)
    ax_pr.legend(fontsize=8)
    plt.tight_layout()

    return fig_roc, fig_pr

# ---------------------------
# Build / prepare data
# ---------------------------

df = get_demo_data()
FEATURES = ["age","income","loan_amount","credit_score","employment_years","num_delinquencies","debt_to_income_ratio"]
TARGET = "target"

# Split once and cache in session_state to reuse
if "split_done" not in st.session_state:
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    st.session_state["X_train"] = X_train
    st.session_state["X_test"] = X_test
    st.session_state["X_train_scaled"] = X_train_scaled
    st.session_state["X_test_scaled"] = X_test_scaled
    st.session_state["y_train"] = y_train
    st.session_state["y_test"] = y_test
    st.session_state["scaler"] = scaler
    st.session_state["split_done"] = True

# ---------------------------
# Sidebar controls
# ---------------------------
st.sidebar.title("Controls")
st.sidebar.markdown("**Select model to train on demo data (fast)**")

model_options = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "Gaussian Naive Bayes",
    "K-Nearest Neighbors"
]
if XGB_AVAILABLE:
    model_options.append("XGBoost")

selected_model_name = st.sidebar.selectbox("Model", model_options, index=0)

train_btn = st.sidebar.button("Train Selected Model")
clear_btn = st.sidebar.button("Clear Trained Model")

st.sidebar.markdown("---")
st.sidebar.markdown("After training, use the input form to Predict — graphs & metrics will appear.")

# ---------------------------
# Main UI layout & header
# ---------------------------
st.title("🏦 CreditPathAI")
st.markdown(
    "Design & develop a machine learning–driven platform that predicts borrower default risk and recommends recovery actions. "
)

# show some dataset summary
c1,c2,c3 = st.columns(3)
with c1:
    st.metric("Rows", len(df))
with c2:
    st.metric("Features", len(FEATURES))
with c3:
    st.metric("Default rate", f"{df[TARGET].mean()*100:.1f}%")

# ---------------------------
# Train / Clear logic
# ---------------------------
if clear_btn:
    for k in ["trained_model", "trained_name", "metrics", "y_score", "y_pred_test", "cm"]:
        if k in st.session_state:
            del st.session_state[k]
    st.success("Cleared trained model and metrics from session.")

if train_btn:
    # Train only selected model (fast)
    model = get_model_by_name(selected_model_name)
    Xtr = st.session_state["X_train_scaled"]
    ytr = st.session_state["y_train"]
    Xte = st.session_state["X_test_scaled"]
    yte = st.session_state["y_test"]

    with st.spinner(f"Training {selected_model_name} (fast)..."):
        model.fit(Xtr, ytr)

        # predict on test
        if hasattr(model, "predict_proba"):
            yprob = model.predict_proba(Xte)[:,1]
        else:
            # fallback using decision_function (scaled) or dummy
            try:
                yprob = model.decision_function(Xte)
                # scale to 0-1
                yprob = (yprob - yprob.min()) / (yprob.max()-yprob.min()+1e-9)
            except Exception:
                yprob = np.zeros(len(Xte))

        ypred = model.predict(Xte)
        cm = confusion_matrix(yte, ypred)
        metrics = compute_metrics(yte, ypred, yprob)

        # store in session
        st.session_state["trained_model"] = model
        st.session_state["trained_name"] = selected_model_name
        st.session_state["metrics"] = metrics
        st.session_state["y_score"] = yprob
        st.session_state["y_pred_test"] = ypred
        st.session_state["cm"] = cm

    st.success(f"{selected_model_name} trained and evaluated model metrics ✅")

# If trained, show compact summary at top
if "trained_model" in st.session_state:
    st.markdown("### ✅ Trained Model Summary")
    tn,fp,fn,tp = (st.session_state["cm"].ravel() if st.session_state["cm"].size == 4 else (np.nan,)*4)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Model", st.session_state["trained_name"])
    with col2:
        st.metric("Accuracy", f"{st.session_state['metrics']['Accuracy']:.3f}")
    with col3:
        st.metric("F1", f"{st.session_state['metrics']['F1']:.3f}")
    with col4:
        st.metric("AUC", f"{st.session_state['metrics']['AUC']:.3f}" if not np.isnan(st.session_state['metrics']['AUC']) else "N/A")

# ---------------------------
# Input form (center) - this is what you asked: enter values manually
# ---------------------------
st.markdown("---")
st.header("🔷 Enter Applicant Values (single sample)")
st.info("Fill these fields and click **Predict** to get model output (must train model first).")

colA, colB, colC = st.columns(3)
with colA:
    age_in = st.number_input("Age", min_value=18, max_value=90, value=35)
    income_in = st.number_input("Annual Income", min_value=0, value=50000, step=1000)
    loan_amount_in = st.number_input("Loan Amount", min_value=0, value=150000, step=1000)
with colB:
    credit_score_in = st.slider("Credit Score", 300, 850, 700)
    employment_years_in = st.number_input("Employment Years", min_value=0, max_value=50, value=5)
    num_delinq_in = st.number_input("Past Delinquencies", min_value=0, max_value=20, value=0)
with colC:
    dti_in = st.number_input("Debt-to-Income Ratio (loan / income)", min_value=0.0, value=loan_amount_in/(income_in+1), step=0.01, format="%.2f")
    st.caption("You can adjust DTI or it will be calculated from loan & income.")
    st.markdown("")  # spacing

predict_btn = st.button("🔮 Predict (using trained model)")

# ---------------------------
# Predict action
# ---------------------------
if predict_btn:
    if "trained_model" not in st.session_state:
        st.error("No trained model found. Train a model from the sidebar first.")
    else:
        # build sample and scale
        sample = pd.DataFrame([{
            "age": age_in,
            "income": income_in,
            "loan_amount": loan_amount_in,
            "credit_score": credit_score_in,
            "employment_years": employment_years_in,
            "num_delinquencies": num_delinq_in,
            "debt_to_income_ratio": dti_in
        }])

        scaler = st.session_state["scaler"]
        sample_scaled = scaler.transform(sample[FEATURES])
        model = st.session_state["trained_model"]

        # predict and proba
        pred = model.predict(sample_scaled)[0]
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(sample_scaled)[0][1])
        else:
            try:
                dec = model.decision_function(sample_scaled)[0]
                # naive scale to 0-1 (for display only)
                prob = float((dec - (-1)) / (2.0))
                prob = max(0.0, min(1.0, prob))
            except Exception:
                prob = 0.5

        # Show card
        if pred == 1:
            st.markdown(f"""
    <div style="background:#ffe6e6;padding:18px;border-left:6px solid #d32f2f;border-radius:8px;">
        <h3 style="color:#b71c1c;margin:0;">⚠️ HIGH DEFAULT RISK</h3>
        <p style="margin:4px 0 0 0; color:#222;">Predicted probability of default: <strong>{prob*100:.1f}%</strong></p>
    </div>
""", unsafe_allow_html=True)

        else:
            st.markdown(f"""
    <div style="background:#e6fff5;padding:18px;border-left:6px solid #2e7d32;border-radius:8px;">
        <h3 style="color:#1b5e20;margin:0;">✅ LOW DEFAULT RISK</h3>
        <p style="margin:4px 0 0 0; color:#222;">Predicted probability of default: <strong>{prob*100:.1f}%</strong></p>
    </div>
""", unsafe_allow_html=True)


        # show short model predictions table
        st.markdown("### 🔎 Model Predictions (selected)")
        pred_table = pd.DataFrame({
            "Model": [st.session_state["trained_name"]],
            "Predicted": [int(pred)],
            "Probability(Default)": [f"{prob:.3f}"]
        })
        st.table(pred_table)

# If trained, show evaluation plots & confusion matrix
# ---------------------------
if "trained_model" in st.session_state:
    st.markdown("---")
    st.header("📈 Model Evaluation ")

    metrics = st.session_state["metrics"]
    metrics_df = pd.DataFrame([metrics], index=[st.session_state["trained_name"]]).T.reset_index()
    metrics_df.columns = ["Metric", "Value"]
    st.table(metrics_df.style.format({"Value": "{:.3f}"}))

    # small side-by-side plots: confusion small + ROC + PR
    cm = st.session_state["cm"]
    y_test = st.session_state["y_test"]
    y_score = st.session_state["y_score"]
    y_pred_test = st.session_state["y_pred_test"]

    col1, col2 = st.columns([1,2])
    with col1:
        st.markdown("#### Confusion Matrix")
        fig_cm = plot_confusion_small(cm, title=st.session_state["trained_name"])
        st.pyplot(fig_cm)
    with col2:
        st.markdown("#### ROC & Precision-Recall")
        fig_roc, fig_pr = plot_roc_pr(y_test, y_score, st.session_state["trained_name"])
        r1, r2 = st.columns(2)
        with r1:
            st.pyplot(fig_roc)
        with r2:
            st.pyplot(fig_pr)

    # show classification breakdown
    st.markdown("#### Classification report")
    try:
        from sklearn.metrics import classification_report
        report = classification_report(y_test, y_pred_test, output_dict=True)
        report_df = pd.DataFrame(report).transpose().round(3)
        st.dataframe(report_df)
    except Exception:
        st.write("Could not compute classification report.")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.caption("CreditPathAI - Made by Purvi Porwal")
