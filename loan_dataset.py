import pandas as pd
df= pd.read_csv("C:/Users/ADITI/CreditPathAI_Oct_Batch/datasets/Loan_Default.csv")
print(df.head())
print(df.describe())
print(df.tail())
print(df.info())     