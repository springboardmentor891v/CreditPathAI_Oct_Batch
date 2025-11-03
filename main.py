import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('data/loan_default.csv')

print(df.head())  
print(df.describe()) 
print(df.shape)
print(df.columns)



