import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ----------------------------------------------
# Step 1: Load the dataset
# ----------------------------------------------
df = pd.read_csv('C:/Users/KAUSHIK/CreditPathAI_Oct_Batch/datasets/Loan_Default.csv')

print("\n✅ Dataset loaded successfully!")
print("📊 Shape of dataset:", df.shape)
print("🧾 Columns available:\n", df.columns.tolist())

# ----------------------------------------------
# Step 2: Separate features and target
# ----------------------------------------------
target_col = 'Status'
print(f"\n🎯 Target column selected: '{target_col}'")

X = df.drop(columns=[target_col])
y = df[target_col]

# ----------------------------------------------
# Step 3: Handle missing values
# ----------------------------------------------
print("\n🔍 Missing values before imputation:")
missing_before = X.isnull().sum()
print(missing_before[missing_before > 0])

# Numerical columns → fill with mean
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())

# Categorical columns → fill with mode
categorical_cols = X.select_dtypes(include=['object']).columns
for col in categorical_cols:
    X[col] = X[col].fillna(X[col].mode()[0])

print("\n✅ Missing values imputed successfully!")

# ----------------------------------------------
# Step 4: Encode categorical variables
# ----------------------------------------------
print("\n🧩 Encoding categorical features...")

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_df = pd.DataFrame(encoder.fit_transform(X[categorical_cols]))
encoded_df.columns = encoder.get_feature_names_out(categorical_cols)
encoded_df.index = X.index

# Combine numerical and encoded categorical columns
X_final = pd.concat([X[numeric_cols], encoded_df], axis=1)

print("\n✅ Categorical encoding completed successfully!")
print("🔢 Encoded dataset shape:", X_final.shape)

# ----------------------------------------------
# Step 5: Data summary
# ----------------------------------------------
print("\n📘 Sample of preprocessed features:")
print(X_final.head())

# ----------------------------------------------
# Optional: Split into train-test sets
# ----------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42, stratify=y
)

print("\n📊 Train/Test split completed!")
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)
