import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# -----------------------------
# SIMPLE MODEL FEATURES
# -----------------------------
simple_features = [
    'loan_amount', 'rate_of_interest', 'Upfront_charges', 'term',
    'property_value', 'income', 'Credit_Score', 'LTV', 'dtir1'
]

target = "Status"

# -----------------------------
# LOAD CSV
# -----------------------------
df = pd.read_csv("Loan_Default.csv")

# Keep only required columns
df = df[simple_features + [target]]

# -----------------------------
# PREPROCESSOR
# -----------------------------
# All features are numeric → no encoding needed
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", simple_features)
    ]
)

# -----------------------------
# MODEL PIPELINE
# -----------------------------
model = RandomForestClassifier(random_state=42)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

# -----------------------------
# TRAIN MODEL
# -----------------------------
X = df[simple_features]
y = df[target]

pipeline.fit(X, y)

# -----------------------------
# SAVE MODEL
# -----------------------------
joblib.dump(pipeline, "simple_model.pkl")

print("simple_model.pkl CREATED SUCCESSFULLY")
