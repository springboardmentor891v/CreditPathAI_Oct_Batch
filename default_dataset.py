import pandas as pd

# Replace with your actual file path
df = pd.read_csv("C:/Users/KAUSHIK/CreditPathAI_Oct_Batch/Loan_Default.csv")

# Display first few rows
print(df.head())

# Check shape (rows, columns)
print(df.shape)

# Get info about columns and data types
print(df.info())

# Quick summary statistics
print(df.describe())

# Check if there are missing values
print(df.isnull().sum())
