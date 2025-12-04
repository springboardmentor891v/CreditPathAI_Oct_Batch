import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve, auc

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except:
    XGB_AVAILABLE = False

df = pd.read_csv("Loan_Default.csv")
print("Loaded dataset:", df.shape)

df = df.drop_duplicates().reset_index(drop=True)

for col in df.columns:
    if df[col].isna().any():
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode().iloc[0])

if "Loan_Default" not in df.columns:
    df["risk_score"] = 0
    df["risk_score"] += (df["LTV"] > 80).astype(int)
    if "Interest_rate_spread" in df.columns:
        df["risk_score"] += (df["Interest_rate_spread"] > 0.5).astype(int)
    if "Upfront_charges" in df.columns:
        df["risk_score"] += (df["Upfront_charges"] > 0.6).astype(int)
    if "approv_in_adv" in df.columns:
        df["risk_score"] += (df["approv_in_adv"] == "N").astype(int)
    if "Neg_ammortization" in df.columns:
        df["risk_score"] += (df["Neg_ammortization"] == "Y").astype(int)
    df["Loan_Default"] = (df["risk_score"] >= 2).astype(int)
    df = df.drop(columns=["risk_score"])

print("\nTarget Distribution:")
print(df["Loan_Default"].value_counts())

y = df["Loan_Default"]
X = df.drop(columns=["Loan_Default"])

num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

if len(cat_cols) > 0:
    X_cat = pd.DataFrame(
        encoder.fit_transform(X[cat_cols]),
        columns=encoder.get_feature_names_out(cat_cols),
        index=X.index
    )
else:
    X_cat = pd.DataFrame(index=X.index)

X_numeric = X[num_cols]
X = pd.concat([X_numeric, X_cat], axis=1)

X.columns = (
    X.columns
    .str.replace('[', '(', regex=False)
    .str.replace(']', ')', regex=False)
    .str.replace('<', '_lt_', regex=False)
    .str.replace('>', '_gt_', regex=False)
    .str.replace(' ', '_', regex=False)
)

num_cols = [c.replace('[','(').replace(']',')').replace('<','_lt_').replace('>','_gt_').replace(' ','_') for c in num_cols]

scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

sm = SMOTE(random_state=42)
X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)

print("\nAfter SMOTE:", y_train_bal.value_counts())

models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42),
    "KNN": KNeighborsClassifier()
}

if XGB_AVAILABLE:
    models["XGBoost"] = XGBClassifier(eval_metric='logloss')

trained_models = {}
metrics = {}
conf_mats = {}

def compute_metrics(y_true, y_pred, y_proba=None):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Sensitivity": round(recall_score(y_true, y_pred), 4),
        "Specificity": round(specificity, 4),
        "Precision": round(precision_score(y_true, y_pred), 4),
        "F1": round(f1_score(y_true, y_pred), 4),
        "AUC": round(roc_auc_score(y_true, y_proba), 4) if y_proba is not None else None
    }

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_bal, y_train_bal)
    trained_models[name] = model
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    metrics[name] = compute_metrics(y_test, y_pred, y_proba)
    conf_mats[name] = confusion_matrix(y_test, y_pred)

print("\nTraining complete.")

roc_data = {}
for name, model in trained_models.items():
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_data[name] = {
            "fpr": fpr,
            "tpr": tpr,
            "auc": auc(fpr, tpr)
        }

joblib.dump(roc_data, "roc_data.pkl")


joblib.dump(trained_models, "models.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(encoder, "encoder.pkl")
joblib.dump(X.columns.tolist(), "columns.pkl")
joblib.dump(num_cols, "numeric_cols.pkl")
joblib.dump(metrics, "metrics.pkl")
joblib.dump(conf_mats, "conf_mats.pkl")

print("\nSaved:")
print("models.pkl")
print("scaler.pkl")
print("encoder.pkl")
print("columns.pkl")
print("numeric_cols.pkl")
print("metrics.pkl")
print("conf_mats.pkl")
print("\nTraining Pipeline Completed Successfully.")