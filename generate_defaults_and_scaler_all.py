# generate_defaults_and_scaler_all.py
import os, joblib, pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = os.path.abspath(os.path.dirname(__file__))

# list of dataset model folders to process
DATASET_MODELS = {
    "kaggle": os.path.join(ROOT, "kaggle_dataset", "models"),
    "microsoft": os.path.join(ROOT, "microsoft_dataset", "models"),
}

# Try to find a processed CSV for a dataset; returns first match or None
def find_processed_csv(dataset_name):
    base = os.path.join(ROOT, f"{dataset_name}_dataset", "data")
    if not os.path.isdir(base):
        return None
    candidates = []
    for root, dirs, files in os.walk(base):
        for f in files:
            if f.lower().endswith(".csv"):
                candidates.append(os.path.join(root, f))
    if not candidates:
        return None
    # prefer files in folders named "03_processed" or "processed"
    for c in candidates:
        p = c.replace("\\", "/").lower()
        if "/03_processed/" in p or "/processed/" in p:
            return c
    return candidates[0]

for ds_name, models_dir in DATASET_MODELS.items():
    print("\n=== Processing dataset:", ds_name, "models_dir:", models_dir)
    os.makedirs(models_dir, exist_ok=True)

    csv_path = find_processed_csv(ds_name)
    if csv_path is None:
        print("  No CSV found for dataset", ds_name, "- skipping defaults/scaler creation.")
        continue

    print("  Using CSV:", csv_path)
    df = pd.read_csv(csv_path)

    # If there is a target column, drop it for defaults/scaler
    target_candidates = ["defaultFlag", "loanStatus", "target", "y"]
    for t in target_candidates:
        if t in df.columns:
            df_features = df.drop(columns=[t])
            break
    else:
        df_features = df.copy()

    # Build defaults: median for numeric, mode for categorical
    defaults = {}
    for col in df_features.columns:
        if pd.api.types.is_numeric_dtype(df_features[col]):
            defaults[col] = df_features[col].median()
        else:
            mode_series = df_features[col].mode(dropna=True)
            defaults[col] = mode_series.iloc[0] if not mode_series.empty else (df_features[col].iloc[0] if len(df_features[col])>0 else 0)

    defaults_df = pd.DataFrame.from_dict(defaults, orient="index", columns=["default"])
    defaults_csv_path = os.path.join(models_dir, "feature_defaults.csv")
    defaults_df.to_csv(defaults_csv_path)
    print("  Saved feature_defaults.csv to:", defaults_csv_path)

    # Fit scaler on numeric columns and save the scaler WITH numeric_cols list
    numeric_cols = [c for c in df_features.columns if pd.api.types.is_numeric_dtype(df_features[c])]
    if numeric_cols:
        scaler = StandardScaler()
        scaler.fit(df_features[numeric_cols].fillna(0).values)
        scaler_container = {"scaler": scaler, "numeric_cols": numeric_cols}
        scaler_path = os.path.join(models_dir, "scaler.joblib")
        joblib.dump(scaler_container, scaler_path)
        print("  Saved scaler.joblib (dict with numeric_cols) to:", scaler_path)
    else:
        print("  No numeric columns found; skipping scaler creation.")
