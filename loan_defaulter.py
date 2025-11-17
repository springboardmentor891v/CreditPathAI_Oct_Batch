import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns

df = pd.read_csv(r"E:\INTERNSHIP_INFO\Loan_default.csv\Loan_default.csv")

print(df.head())
print(f"Dataset Shape (Rows, Columns): {df.shape}")
print(df.info())
print(df.describe())
print(df.tail())

print("\nMissing Values per Column:\n", df.isnull().sum())
print("\nDuplicate Rows:", df.duplicated().sum())

print("\nDefault Value Counts:\n", df['Default'].value_counts(normalize=True))

sns.countplot(x='Default', data=df, palette='Set2')
plt.title("Loan Default Class Distribution")
plt.xlabel("Default (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()


numeric_cols = [
    'Age', 'Income', 'LoanAmount', 'CreditScore',
    'MonthsEmployed', 'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio'
]

plt.figure(figsize=(16, 12))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(3, 3, i)
    sns.histplot(df[col], kde=True, bins=30, color='skyblue')
    plt.title(f"Distribution of {col}")
    plt.tight_layout()
plt.show()

plt.figure(figsize=(16, 12))
for i, col in enumerate(numeric_cols[:6], 1):
    plt.subplot(3, 3, i)
    sns.boxplot(x='Default', y=col, data=df, palette='coolwarm')
    plt.title(f"{col} vs Default")
    plt.tight_layout()
plt.show()

categorical_cols = [
    'Education', 'EmploymentType', 'MaritalStatus',
    'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner'
]

plt.figure(figsize=(16, 14))
for i, col in enumerate(categorical_cols, 1):
    plt.subplot(4, 2, i)
    sns.countplot(x=col, data=df, hue='Default', palette='Set3')
    plt.title(f"{col} vs Default")
    plt.xticks(rotation=45)
    plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5))
sns.scatterplot(
    x='CreditScore',
    y='LoanAmount',
    hue='Default',
    data=df,
    palette='coolwarm'
)
plt.title("Credit Score vs Loan Amount (by Default Status)")
plt.show()

plt.figure(figsize=(7, 5))
sns.scatterplot(
    x='Income',
    y='DTIRatio',
    hue='Default',
    data=df,
    palette='viridis'
)
plt.title("Debt-to-Income Ratio vs Income (by Default)")
plt.show()

numeric_df = df.select_dtypes(include=['int64', 'float64'])

plt.figure(figsize=(12, 8))
corr_matrix = numeric_df.corr()
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap='coolwarm',
    fmt='.2f',
    linewidths=0.5
)
plt.title("Correlation Heatmap of Numeric Features", fontsize=14)
plt.show()


if 'Default' in corr_matrix.columns:
    plt.figure(figsize=(8, 6))
    top_corr = corr_matrix['Default'].sort_values(ascending=False)
    sns.heatmap(
        corr_matrix.loc[top_corr.index, top_corr.index],
        annot=True,
        cmap='YlGnBu',
        fmt='.2f'
    )
    plt.title("Top Correlated Features with Default")
    plt.show()
else:
    print("'Default' column not found in numeric columns.")


X = df.drop(columns=['Default', 'LoanID'])  
y = df['Default']

print("Features shape:", X.shape)
print("Target shape:", y.shape)

X = pd.get_dummies(X, drop_first=True)

print(" After encoding:")
print("Shape of X:", X.shape)
print("Columns:", X.columns[:10])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Training shape:", X_train.shape, "Testing shape:", X_test.shape)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Scaling complete")
print("X_train_scaled shape:", X_train_scaled.shape)
print("X_test_scaled shape:", X_test_scaled.shape)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
import joblib

X = df.drop(columns=['Default', 'LoanID'])
y = df['Default']
print("Features shape:", X.shape)
print("Target shape:", y.shape)

X = pd.get_dummies(X, drop_first=True)
print("After encoding, shape of X:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Training shape:", X_train.shape, "Testing shape:", X_test.shape)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Scaling complete.")

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))

print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

importances = model.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[::-1][:15]
plt.figure(figsize=(10, 8))
sns.barplot(x=importances[indices], y=feature_names[indices], palette='viridis')
plt.title("Top 15 Feature Importances")
plt.show()

joblib.dump(scaler, "scaler.joblib")
joblib.dump(model, "loan_default_rf_model.joblib")
print("Scaler and model saved.")