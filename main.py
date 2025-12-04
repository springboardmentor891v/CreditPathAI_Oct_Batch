# Main.py

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from imblearn.ensemble import BalancedRandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# ==========================
# 1. LOAD RAW DATASET
# ==========================
print("\n Loading dataset...")

file_path = "Loan_Default.csv" 

if not os.path.exists(file_path):
    print(f" Error: File not found at {file_path}")
    print("Please ensure Loan_Default.csv is in the same folder.")
    exit()

df = pd.read_csv(file_path)
df = df.drop(columns=["ID", "year"], errors="ignore")

X = df.drop("Status", axis=1)
y = df["Status"]

# Identify columns
categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
numeric_cols = X.select_dtypes(include=[int, float]).columns.tolist()

print("\nCATEGORICAL:", categorical_cols)
print("NUMERIC:", numeric_cols)

# ==========================
# 2. IMPUTE MISSING VALUES
# ==========================
num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")

X[numeric_cols] = num_imputer.fit_transform(X[numeric_cols])
X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])

# ==========================
# 3. LABEL ENCODE CATEGORICAL COLS
# ==========================
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# ==========================
# 4. SCALE NUMERIC FEATURES
# ==========================
scaler = MinMaxScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# ==========================
# 5. SAVE PREPROCESSOR FILES
# ==========================
os.makedirs("preprocessor", exist_ok=True)

# Save exact column order (Critical for app.py to match training)
feature_columns = X.columns.tolist()
joblib.dump(feature_columns, "preprocessor/feature_columns.pkl")

joblib.dump(label_encoders, "preprocessor/label_encoders.pkl")
joblib.dump(scaler, "preprocessor/scaler.pkl")
joblib.dump(num_imputer, "preprocessor/num_imputer.pkl")
joblib.dump(cat_imputer, "preprocessor/cat_imputer.pkl")
joblib.dump(categorical_cols, "preprocessor/categorical_cols.pkl")
joblib.dump(numeric_cols, "preprocessor/numeric_cols.pkl")

print("Preprocessor & Column Order saved.")

# ==========================
# 6. TRAIN TEST SPLIT
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================
# 7. TRAIN MODELS
# ==========================
os.makedirs("models", exist_ok=True)

def save_model(name, model):
    joblib.dump(model, f"models/{name}.pkl")
    print(f"Saved: {name}.pkl")

print("\nTraining Models...\n")

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
save_model("logistic_regression", log_reg)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
save_model("decision_tree", dt)

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
save_model("random_forest", rf)

nb = GaussianNB()
nb.fit(X_train, y_train)
save_model("naive_bayes", nb)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
save_model("knn", knn)

brf = BalancedRandomForestClassifier(n_estimators=200, random_state=42)
brf.fit(X_train, y_train)
save_model("balanced_rf", brf)

xgb = XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42)
xgb.fit(X_train, y_train)
save_model("xgboost", xgb)

lgbm = LGBMClassifier(random_state=42)
lgbm.fit(X_train, y_train)
save_model("lightgbm", lgbm)

cat = CatBoostClassifier(verbose=0, random_state=42)
cat.fit(X_train, y_train)
save_model("catboost", cat)

# --- CRITICAL FIX: SOFT VOTING ---
voting = VotingClassifier(
    estimators=[("lr", log_reg), ("dt", dt), ("rf", rf)],
    voting="soft"  
)
voting.fit(X_train, y_train)
save_model("voting_classifier", voting)
# ---------------------------------

stack = StackingClassifier(
    estimators=[("lr", log_reg), ("dt", dt), ("rf", rf)],
    final_estimator=LogisticRegression(),
    cv=5
)
stack.fit(X_train, y_train)
save_model("stacking_classifier", stack)

print("\n All Models + Preprocessor Saved Successfully!")