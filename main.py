import pandas as pd

df = pd.read_csv("./data/Loan_Default.csv")
# print(df.head())
# print(df.tail(2))
# print(df.info())
print(df.describe())
# print(df.columns)
# print(df.shape)
# print(df.isnull())
# print(df.isnull().sum())
# lpc = df['loan_limit'].value_counts()
# print(lpc)

