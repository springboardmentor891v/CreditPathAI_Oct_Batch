import pandas as pd

# Using a sample open loan dataset
url = "https://raw.githubusercontent.com/plotly/datasets/master/loan_data.csv"

# Read the dataset
df = pd.read_csv(url)

# Display top and bottom rows
print("First 5 rows of the dataset:")
print(df.head())

print("\nLast 5 rows of the dataset:")
print(df.tail())

# Show dataset shape
print("\nShape of dataset:", df.shape)

# Show info about columns
print("\nDataset Info:")
print(df.info())

# Show descriptive statistics
print("\nDescriptive Statistics:")
print(df.describe())

# Example of using sum()
print("\nSum of numeric columns:")
print(df.sum())
