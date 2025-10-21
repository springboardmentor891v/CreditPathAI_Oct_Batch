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
    
explore_data(df)

    
