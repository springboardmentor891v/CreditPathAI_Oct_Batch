import pandas as pd
<<<<<<< HEAD
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

=======
>>>>>>> 5156b6352e51681bbf57d64b3a257715bc713ed6
df = pd.read_csv(r"E:\INTERNSHIP_INFO\Loan_default.csv\Loan_default.csv")
print(df.head())

print(f"Dataset Shape (Rows, Columns): {df.shape}")
<<<<<<< HEAD
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
'''
df.hist(['Age', 'Income', 'LoanAmount'], bins=30, figsize=(12, 8))

print(df.loc[0:2])

print(df.head())
print(df.columns)
'''
=======
df.info() 
print(df.describe())
df.tail()
df.isnull()
print(df.loc[0:2])
>>>>>>> 5156b6352e51681bbf57d64b3a257715bc713ed6
