import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

file_path = "Loan_Default.csv"
df = pd.read_csv(file_path)

print("Basic Info:\n")
df.info()
print("Statistical Summary:\n")
display(df.describe())
print("Shape of the DataFrame:", df.shape)
print("First 5 Rows:")
display(df.head())
print("Last 5 Rows:")
display(df.tail())

missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0].sort_values(ascending=False)

print("\nMissing Values in Each Column:\n")
print(missing_values)

plt.figure(figsize=(10, 5))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()

numeric_cols = df.select_dtypes(include=np.number).columns

if len(numeric_cols) > 0:
    df[numeric_cols].hist(figsize=(12, 10), bins=20, color='skyblue', edgecolor='black')
    plt.suptitle("Distribution of Numeric Features", fontsize=16)
    plt.show()
else:
    print("No numeric columns found in dataset.")