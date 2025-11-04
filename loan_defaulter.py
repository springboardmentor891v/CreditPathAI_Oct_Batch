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

