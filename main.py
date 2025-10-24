import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# --------------------------------
# 🔹 Redirect Output to a Text File
# --------------------------------
sys.stdout = open("output.txt", "w", encoding="utf-8")

# Load dataset
data_path = "loan_default.csv"
loan_df = pd.read_csv(data_path)

# -------------------------------
# 🔹 Step 1: Basic Dataset Overview
# -------------------------------
print("\n--- FIRST 5 RECORDS ---")
print(loan_df.head())

print("\n--- LAST 5 RECORDS ---")
print(loan_df.tail())

print("\n--- SHAPE OF DATA (ROWS, COLUMNS) ---")
print(loan_df.shape)

print("\n--- COLUMN NAMES ---")
print(list(loan_df.columns))

print("\n--- DATA TYPES ---")
print(loan_df.dtypes)

print("\n--- STATISTICAL SUMMARY ---")
print(loan_df.describe(include='all'))

print("\n--- DATAFRAME INFO ---")
print(loan_df.info())

# --------------------------------
# 🔹 Step 2: Handling Missing Values
# --------------------------------
print("\nMissing values per column:")
print(loan_df.isnull().sum())

print("\nDuplicate rows in dataset:", loan_df.duplicated().sum())

# Function to find missing values (column-wise)
def missing_in_columns(df):
    missing_cols = df.isnull().sum()
    return missing_cols[missing_cols > 0]

# Function to find missing values (row-wise)
def missing_in_rows(df):
    missing_rows = df.isnull().sum(axis=1)
    return missing_rows[missing_rows > 0]

print("\nColumns with missing values:\n", missing_in_columns(loan_df))
print("\nRows containing missing values:\n", missing_in_rows(loan_df))

# Cleaned versions for analysis
no_missing_rows = loan_df.dropna(axis=0)
no_missing_cols = loan_df.dropna(axis=1)

print("\nAfter dropping missing rows:", no_missing_rows.shape)
print("After dropping missing columns:", no_missing_cols.shape)

# --------------------------------
# 🔹 Step 3: Column-Level Analysis
# --------------------------------
print("\nValue counts for 'age':\n", loan_df['age'].value_counts(dropna=False))
print("\nValue counts for 'loan_amount':\n", loan_df['loan_amount'].value_counts(dropna=False))
print("\nValue counts for 'loan_limit':\n", loan_df['loan_limit'].value_counts(dropna=False))

# Convert 'age' from string/object to numeric (if needed)
if loan_df['age'].dtype == 'object':
    loan_df['age'] = loan_df['age'].str.extract(r'(\d+)').astype(float)

# --------------------------------
# 🔹 Step 4: Data Visualization
# --------------------------------
# 1️⃣ Bar chart – Loan Status Distribution
plt.figure(figsize=(6,4))
loan_df['Status'].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Loan Status Distribution')
plt.xlabel('Status')
plt.ylabel('Count')
plt.savefig("loan_status_distribution.png", bbox_inches="tight")
plt.show()
plt.close()

# 2️⃣ Histogram – Loan Amount Distribution
plt.figure(figsize=(6,4))
plt.hist(loan_df['loan_amount'].dropna(), bins=25, color='lightgreen', edgecolor='black')
plt.title('Distribution of Loan Amounts')
plt.xlabel('Loan Amount')
plt.ylabel('Frequency')
plt.savefig("loan_amount_distribution.png", bbox_inches="tight")
plt.show()
plt.close()

# 3️⃣ Boxplot – Loan Amount vs Loan Status
plt.figure(figsize=(6,4))
sns.boxplot(x='Status', y='loan_amount', data=loan_df, hue='Status', palette='Set2', legend=False)
plt.title('Loan Amount by Loan Status')
plt.xlabel('Status')
plt.ylabel('Loan Amount')
plt.savefig("loan_amount_vs_status.png", bbox_inches="tight")
plt.show()
plt.close()

# --------------------------------
# 🔹 Step 5: Schema Definition
# --------------------------------
data_schema = {
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

# Apply schema to DataFrame
for column, dtype in data_schema.items():
    if column in loan_df.columns:
        try:
            loan_df[column] = loan_df[column].astype(dtype)
        except Exception as e:
            print(f"⚠️ Could not convert {column}: {e}")

# Keep only required schema columns
loan_df = loan_df[list(data_schema.keys())]

print("\n--- UPDATED DATA TYPES ---")
print(loan_df.dtypes)

print("\n--- CLEANED DATA SAMPLE ---")
print(loan_df.head())

# Close the text file and restore terminal output
sys.stdout.close()
sys.stdout = sys.__stdout__

print("✅ All output and graphs successfully saved!")
