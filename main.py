import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:/Users/JEEVAN REDDY/OneDrive/Desktop/CreditPathAI_SpringBoard/Loan_Default.csv")

def explore_data(df):
    """Explore the DataFrame by displaying basic information and statistics."""
    print(df.info())
    print(df.describe())
    print(df.head())
    print(df.tail())
    print("DataFrame Shape:", df.shape)
    print("DataFrame Columns:", df.columns.tolist())
    
explore_data(df)

def find_missing_values(df):
    """Find and return the count of missing values in each column of the DataFrame."""
    missing_values = df.isnull().sum()
    return missing_values[missing_values > 0]

missing_values = find_missing_values(df)
print("Missing values in each column:\n", missing_values)
