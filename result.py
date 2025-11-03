# Exploratory Data Analysis (EDA) - Loan Default Dataset

# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('C:/Users/KAUSHIK/CreditPathAI_Oct_Batch/Loan_Default.csv')

print("\nDataset loaded successfully!")
print(f"Shape of dataset: {df.shape}")
print("Columns:", df.columns.tolist())

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary (Numerical Columns):")
print(df.describe())

print("\nFirst 5 rows:")
print(df.head())

print("\nLast 5 rows:")
print(df.tail())

print("\nMissing Values Summary:")
missing = df.isnull().sum()
print(missing[missing > 0])

plt.figure(figsize=(10,5))
sns.heatmap(df.isnull(), cbar=False, cmap='coolwarm')
plt.title("Missing Values Heatmap")
plt.show()

rows_before = df.shape[0]
df = df.drop_duplicates()
rows_after = df.shape[0]
print(f"\n Removed {rows_before - rows_after} duplicate rows.")

categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

print("\nCategorical Columns:", categorical_cols)
print("Numerical Columns:", numeric_cols)

target = 'Status'
if target in df.columns:
    print(f"\nTarget column found: '{target}'")
    plt.figure(figsize=(6,4))
    sns.countplot(x=df[target], palette='Set2')
    plt.title(f"Distribution of Target Variable: {target}")
    plt.xlabel(target)
    plt.ylabel("Count")
    plt.show()
else:
    print("\nTarget column not found! Please verify the name.")

for col in categorical_cols[:5]:  # limit to top 5 for readability
    plt.figure(figsize=(8,4))
    df[col].value_counts().head(10).plot(kind='bar', color='coral', edgecolor='black')
    plt.title(f"Bar Chart of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.xticks(rotation=30)
    plt.show()

df[numeric_cols].hist(figsize=(15,10), bins=25, color='skyblue', edgecolor='black')
plt.suptitle("Histograms of Numerical Features", fontsize=16)
plt.show()

plt.figure(figsize=(15,8))
for i, col in enumerate(numeric_cols[:6], 1):  # Display first 6
    plt.subplot(2,3,i)
    sns.boxplot(x=df[col], color='lightgreen')
    plt.title(col)
plt.tight_layout()
plt.suptitle("Boxplots of Numeric Features", y=1.02, fontsize=16)
plt.show()

plt.figure(figsize=(12,8))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=False, cmap='coolwarm', center=0)
plt.title("Correlation Heatmap (Numerical Features)")
plt.show()

sample_features = numeric_cols[:5]
if target in df.columns:
    sample_features.append(target)

sns.pairplot(df[sample_features], diag_kind='kde', palette='husl')
plt.suptitle("Pairplot of Selected Features", y=1.02)
plt.show()

if target in df.columns:
    corr_target = corr[target].dropna().sort_values(ascending=False)
    plt.figure(figsize=(8,5))
    corr_target.plot(kind='bar', color='teal', edgecolor='black')
    plt.title(f"Correlation of Numeric Features with Target '{target}'")
    plt.ylabel("Correlation Value")
    plt.show()

print("\n EDA completed successfully!")
