import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('Loan_Default.csv')

print("First 5 rows:\n", df.head())
print("\nLast 5 rows:\n", df.tail())
print("\nShape of the dataset:", df.shape)
print("\nColumn Names:\n", df.columns)
print("\nCheck for null values:\n", df.isnull().sum())

# Selecting columns and rows
print("\nSelect 'Gender' column:\n", df['Gender'].head())
print("\nSelect multiple columns 'age' and 'Gender':\n", df[['age', 'Gender']].head())
print("\nSelect first row using iloc:\n", df.iloc[0])
print("\nSelect first row using loc:\n", df.loc[0])
print("\nFilter rows where 'ID' > 10:\n", df[df['ID'] > 10])

# Data Cleaning
# Convert 'age' from object to float
df['age'] = df['age'].str.extract('(\d+)').astype(float)

# Data Visualization
# 1. Bar Plot: Count of loan status
df['Status'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Loan Status Count')
plt.xlabel('Status')
plt.ylabel('Count')
plt.show()

# 2. Histogram: Distribution of loan amount
df['loan_amount'].plot(kind='hist', bins=30, color='lightgreen')
plt.title('Loan Amount Distribution')
plt.xlabel('Loan Amount')
plt.show()

# 3. Box Plot: Loan amount by status
df.boxplot(column='loan_amount', by='Status', grid=False)
plt.title('Loan Amount by Status')
plt.suptitle('')
plt.xlabel('Status')
plt.ylabel('Loan Amount')
plt.show()

# Schema Definition and Type Casting
schema = {
    'loan_amount': 'float',
    'rate_of_interest': 'float',
    'term': 'category',
    'Credit_Score': 'float',
    'income': 'float',
    'loan_purpose': 'category',
    'Credit_Worthiness': 'category',
    'Gender': 'category',
    'Status': 'category',
    'age': 'float',
    'LTV': 'float',
    'dtir1': 'float'
}

for col, dtype in schema.items():
    df[col] = df[col].astype(dtype)

# Keep only required columns
df = df[list(schema.keys())]

# Verify data types and display cleaned data
print("\nData types after conversion:\n", df.dtypes)
print("\nFirst 5 rows of cleaned DataFrame:\n", df.head())
