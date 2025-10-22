import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Loan_default.csv')

print(df.head())
print(df.tail())
print(df.isnull().sum())
print(df.duplicated().sum())
print(df.shape)
print(df.columns)
print(df.dtypes)
print(df.describe())
print(df.info())
print(df.iloc[0])  # First row
print(df.iloc[-1])  # Last row
print(df.iloc[[0, 2, 4]])  # Rows 0, 2, 4
print(df.iloc[:, 1])  # All rows, column 1 (Math)
print(df.iloc[:, [0, 2]])  # All rows, columns 0 and 2 (Name & Science)
print(df.loc[0])  # First row using loc
print(df.loc[:, 'age'])  # All rows, 'age' column using loc
print(df.loc[:, ['age', 'loan_amount']])  # All rows, 'age' and 'loan_amount' columns using loc

# Analyze 'age' column
apc = df['age'].value_counts(dropna=False)  # include NaNs in counts
print(apc)

# Analyze 'loan_amount' column
lac = df['loan_amount'].value_counts(dropna=False)  # include NaNs in counts
print(lac)

# Analyze 'loan_limit' column
lpc = df['loan_limit'].value_counts(dropna=False)  # include NaNs in counts
print(lpc)

def find_missing_values(df):
    """Find and return the count of missing values in each column of the DataFrame."""
    missing_values = df.isnull().sum()
    return missing_values[missing_values > 0]

missing_values = find_missing_values(df)
print("Missing values in each column:\n", missing_values)

def find_missing_values(df):
    """Find and return the count of missing values in each rows of the DataFrame."""
    missing_values = df.isnull().sum(axis=1)
    return missing_values[missing_values > 0]
missing_values_rows = find_missing_values(df)
print("Missing values in each row:\n", missing_values_rows)

df_rows_cleaned = df.dropna(axis=0)
print("\nRows with no missing values:\n", df_rows_cleaned)

df_cols_cleaned = df.dropna(axis=1)
print("\nColumns with no missing values:\n", df_cols_cleaned)

df_rows_all_nan = df.dropna(how='all', axis=0)
print("\nRows with all values missing:\n", df_rows_all_nan)

df_cols_all_nan = df.dropna(how='all', axis=1)
print("\nColumns with all values missing:\n", df_cols_all_nan)