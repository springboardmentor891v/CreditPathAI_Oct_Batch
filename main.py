import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Loan_default.csv')

print(df.head())
print(df.tail())
print(df.describe())
print(df.info())