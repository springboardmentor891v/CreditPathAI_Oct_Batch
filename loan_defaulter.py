import pandas as pd
df = pd.read_csv(r"E:\INTERNSHIP_INFO\Loan_default.csv\Loan_default.csv")
print(df.head())

print(f"Dataset Shape (Rows, Columns): {df.shape}")
df.info() 
print(df.describe())
df.tail()
df.isnull()
print(df.loc[0:2])
