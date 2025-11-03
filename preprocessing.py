import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

#loading dataset
df = pd.read_csv('data/loan_Default.csv')

print("\n✅ Dataset loaded successfully!")
print("Shape:", df.shape)
print("Columns:\n", df.columns.tolist())


target_column = 'Status'
print(f"\n🎯 Target column selected: '{target_column}'")

# Separate features (X) and target (y)
X = df.drop(columns=[target_column])
y = df[target_column]


# Step 3: Handle Missing Values
print("\n🔍 Checking missing values before imputation:")
print(X.isnull().sum()[X.isnull().sum() > 0])

# Numerical columns → fill with mean
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
X[num_cols] = X[num_cols].fillna(X[num_cols].mean())

# Categorical columns → fill with mode
cat_cols = X.select_dtypes(include=['object']).columns
for col in cat_cols:
    X[col] = X[col].fillna(X[col].mode()[0])

print("\n✅ Missing values handled successfully!")


# Step 4: One-Hot Encoding  
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

encoded_cat = pd.DataFrame(encoder.fit_transform(X[cat_cols]))
encoded_cat.columns = encoder.get_feature_names_out(cat_cols)
encoded_cat.index = X.index

# Combine numerical + encoded categorical
X_encoded = pd.concat([X[num_cols], encoded_cat], axis=1)
print("\n✅ Categorical features encoded successfully!")
print("Encoded dataset shape:", X_encoded.shape)

# Identify categorical columns
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
print("\n🧩 Categorical columns to encode:", cat_cols)

from sklearn.preprocessing import OneHotEncoder

# Initialize encoder
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# Fit and transform categorical features
encoded_cat = pd.DataFrame(encoder.fit_transform(X[cat_cols]))

# Assign new column names
encoded_cat.columns = encoder.get_feature_names_out(cat_cols)
encoded_cat.index = X.index

# Drop old categorical columns and join encoded ones
X_encoded = pd.concat([X.drop(columns=cat_cols), encoded_cat], axis=1)

print("\n✅ All categorical features encoded successfully!")
print("🔢 Encoded dataset shape:", X_encoded.shape)

print("\n🔍 Sample of encoded features:")
print(X_encoded.head())

