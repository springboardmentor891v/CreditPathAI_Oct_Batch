import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"E:\INTERNSHIP_INFO\Loan_default.csv\Loan_default.csv")
print(df.head())

print(f"Dataset Shape (Rows, Columns): {df.shape}")
print(df.info())
print(df.describe())
print(df.tail())
print(df.isnull().sum())
print(df.duplicated().sum())

print(df.columns)
print(df['Default'].value_counts(normalize=True))
sns.countplot(x='Default',data=df)
plt.title("Loan Default Class Distribution")
plt.xlabel("Default (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()


sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (8, 5)

plt.figure(figsize=(6,4))
sns.countplot(x='Default', data=df, palette='Set2')
plt.title("Target Distribution: Default vs Non-Default")
plt.show()


numeric_cols = ['Age', 'Income', 'LoanAmount', 'CreditScore',
'MonthsEmployed', 'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio']

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



'''
df.hist(['Age', 'Income', 'LoanAmount'], bins=30, figsize=(12, 8))

print(df.loc[0:2])

print(df.head())
print(df.columns)
'''
