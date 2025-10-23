import pandas as pd
df=pd.read_csv("data\Loan_Default.csv")
print(df.tail())        # View last 5 rows
print(df.isnull())      # Returns a DataFrame of True/False for each cell
print(df.sum)
print(df.shape)         # Get number of rows and columns
print(df.columns)       # List column names

df.iloc[0]              # Select first row by index
df.loc[0]               # Select first row by label
