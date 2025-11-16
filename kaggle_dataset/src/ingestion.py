import os
import pandas as pd

# Paths
raw_data_path = r"data/raw/Kaggle_loan_default.csv"
interim_data_path = "data/interim/loan_data_clean.csv"

# Load data
df = pd.read_csv(raw_data_path,encoding='latin1')
print("Data loaded successfully! Shape:", df.shape)

# Quick summary
print("\Columns:", list(df.columns))
print("Missing values summary:\n", df.isnull().sum()[df.isnull().sum() > 0])

# Remove duplicates
before = df.shape[0]
df.drop_duplicates(inplace=True)
after = df.shape[0]
print(f"Removed {before - after} duplicate rows.")

# Save cleaned data
os.makedirs(os.path.dirname(interim_data_path), exist_ok=True)
df.to_csv(interim_data_path, index=False)
print(f"Cleaned data saved to: {interim_data_path}")
