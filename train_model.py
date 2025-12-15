# train_model.py (FINAL FIXED VERSION)
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    BaggingClassifier,
    VotingClassifier,
    StackingClassifier
)

# XGBoost
from xgboost import XGBClassifier


# ============================================================
# CONFIG
# ============================================================
DATA_FILE = "Loan_Default_100.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"{DATA_FILE} not found.")

print("Loading dataset...")
df = pd.read_csv(DATA_FILE)

if "Status" not in df.columns:
    raise ValueError("'Status' column missing in dataset")

df["Status"] = df["Status"].astype(int)

# ============================================================
# FEATURES TO USE
# ============================================================
feature_cols = [
    "loan_limit","Gender","approv_in_adv","loan_type","loan_purpose",
    "Credit_Worthiness","open_credit","business_or_commercial",
    "loan_amount","rate_of_interest","Interest_rate_spread","Upfront_charges",
    "term","Neg_ammortization","interest_only","lump_sum_payment",
    "property_value","construction_type","occupancy_type","Secured_by",
    "total_units","income","credit_type","Credit_Score",
    "co-applicant_credit_type","age","submission_of_application",
    "LTV","Region","Security_Type","dtir1"
]

# Validate
feature_cols = [c for c in feature_cols if c in df.columns]
print("Training on:", feature_cols)

X = df[feature_cols].copy()
y = df["Status"].copy()  # 1 = SAFE, 0 = RISK


# ============================================================
# LABEL ENCODERS
# ============================================================
encoders = {}

for col in X.columns:
    if X[col].dtype == "object" or X[col].dtype == "O":
        X[col] = X[col].astype(str).str.strip().fillna("NA")

        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    else:
        X[col] = X[col].fillna(X[col].median())


# ============================================================
# SCALING
# ============================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

# ============================================================
# MODEL DEFINITIONS — FIXED VERSION
# ENSURES consistent positive class = 1
# ============================================================
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    ),

    "Naive Bayes": GaussianNB(),

    "KNN": KNeighborsClassifier(),

    "Decision Tree": DecisionTreeClassifier(
        random_state=RANDOM_STATE,
        class_weight="balanced"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=400,
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample"
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        random_state=RANDOM_STATE
    ),

    "Bagging Classifier": BaggingClassifier(
        random_state=RANDOM_STATE
    ),

    # FIXED Voting — ensures probability belongs to class 1
    "Voting Classifier": VotingClassifier(
        estimators=[
            ("lr", LogisticRegression(max_iter=2000)),
            ("rf", RandomForestClassifier(n_estimators=200)),
            ("dt", DecisionTreeClassifier())
        ],
        voting="soft"
    ),

    # FIXED Stacking — avoids wrong class orders
    "Stacking Classifier": StackingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=200)),
            ("knn", KNeighborsClassifier()),
            ("dt", DecisionTreeClassifier())
        ],
        final_estimator=LogisticRegression(),
        stack_method="predict_proba"
    ),

    # FIXED XGBoost — ensures correct positive class (label=1)
    "XGBoost": XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=RANDOM_STATE,
        objective="binary:logistic",
        eval_metric="logloss"
    )
}


# ============================================================
# TRAINING LOOP
# ============================================================
print("\nTraining models...")
best_auc = -1.0
best_name = None

for name, model in models.items():
    print(f"\n----- Training {name} -----")

    # Fit
    model.fit(X_train, y_train)

    # Predict probabilities
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)

        # Always use index of class 1
        pos_idx = classes.index(1) if 1 in classes else -1
        y_prob = model.predict_proba(X_test)[:, pos_idx]

    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        y_prob = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

    else:
        y_prob = model.predict(X_test).astype(float)

    y_pred = (y_prob >= 0.5).astype(int)

    # Metrics
    acc = accuracy_score(y_test, y_pred)

    try:
        auc_val = roc_auc_score(y_test, y_prob)
    except:
        auc_val = float("nan")

    print(f"Accuracy: {acc:.4f} | AUC: {auc_val:.4f}")
    print(confusion_matrix(y_test, y_pred))

    # Save model
    fname = name.replace(" ", "_").replace("Classifier", "").strip() + ".pkl"
    joblib.dump(model, fname)
    print("Saved:", fname)

    if not np.isnan(auc_val) and auc_val > best_auc:
        best_auc = auc_val
        best_name = name


# ============================================================
# SAVE ENCODERS & SCALER
# ============================================================
joblib.dump(scaler, "scaler.pkl")
joblib.dump(encoders, "encoders.pkl")

print("\nTraining complete.")
print("Best Model:", best_name, "| AUC =", best_auc)
