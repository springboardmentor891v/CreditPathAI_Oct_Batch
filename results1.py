#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Load Dataset
df = pd.read_csv('data/loan_Default.csv')

#exploring the dataset
print(df.info())
print(df.describe())
print(df.head())
print(df.tail())
print(df.shape)
print(df.columns.tolist())

#finding missing values
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

plt.figure(figsize=(10,5))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()

#Remove duplicates
before = df.shape[0]
df = df.drop_duplicates()
after = df.shape[0]
print(f"Removed {before - after} duplicate rows.")

# Detect Categorical and Numerical Columns
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

print("\n Categorical Columns:", cat_cols)
print("Numerical Columns:", num_cols)

target_col = 'Status'   
if target_col not in df.columns:
    print("\nTarget column not found — please update 'target_col' name!")
else:
    print(f"\nTarget Column: {target_col}")


# Distribution of Target Variable
if target_col in df.columns:
    plt.figure(figsize=(6,4))
    sns.countplot(x=df[target_col], palette='Set2')
    plt.title(f"Distribution of Target Variable: {target_col}")
    plt.show()


# Bar Plots for Categorical Columns
for col in cat_cols[:5]:  # limit to 5 for readability
    plt.figure(figsize=(8,4))
    df[col].value_counts().head(10).plot(kind='bar', color='lightcoral', edgecolor='black')
    plt.title(f"Bar Plot of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.xticks(rotation=30)
    plt.show()

# Histograms for Numerical Columns
df[num_cols].hist(figsize=(15,10), bins=30, color='skyblue', edgecolor='black')
plt.suptitle("Histograms of Numeric Features", fontsize=16)
plt.show()


plt.figure(figsize=(15,8))
for i, col in enumerate(num_cols[:6], 1):  # display first 6
    plt.subplot(2,3,i)
    sns.boxplot(x=df[col], color='lightgreen')
    plt.title(col)
plt.tight_layout()
plt.suptitle("Boxplots of Numeric Features", y=1.02, fontsize=16)
plt.show()


# Correlation Heatmap
plt.figure(figsize=(12,8))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=False, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


# Pairplot (sample columns)
sample_cols = num_cols[:5]
if target_col in df.columns:
    sample_cols.append(target_col)

sns.pairplot(df[sample_cols], diag_kind='kde', palette='husl')
plt.suptitle("Pairplot of Selected Features", y=1.02)
plt.show()


# Correlation of Features with Target
if target_col in df.columns:
    target_corr = corr[target_col].dropna().sort_values(ascending=False)
    plt.figure(figsize=(8,5))
    target_corr.plot(kind='bar', color='teal', edgecolor='black')
    plt.title(f"Correlation of Numeric Features with {target_col}")
    plt.ylabel("Correlation Value")
    plt.show()

print("\n EDA completed successfully!")
