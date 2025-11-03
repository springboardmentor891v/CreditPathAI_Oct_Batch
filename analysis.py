import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("data/loan_default.csv")

#basic operations to understand the data
print(df.head())
print(df.tail())        # View last 5 rows
print(df.isnull())      # Returns a DataFrame of True/False for each cell
print(df.sum)
print(df.shape)         # Get number of rows and columns
print(df.columns)       # List column names


df['Gender']            # Select a single column
df[['age', 'Gender']]    # Select multiple columns
df.iloc[0]              # Select first row by index
df.loc[0]               # Select first row by label
df[df['ID'] > 10]      # Filter rows where column > 10 


#to find how many numer of null values are present in each column
print(df.isnull().sum())

#to convert the datatype of 'age' column from object to float
df['age'] = df['age'].str.extract('(\d+)').astype(float)
 
# 1. Bar plot: Count of loan status
df['Status'].value_counts().plot(kind='bar')
plt.title('Loan Status Count')
plt.xlabel('Status')
plt.ylabel('Count')
plt.show()


# 2. Histogram: Distribution of loan amount
df['loan_amount'].plot(kind='hist', bins=30)
plt.title('Loan Amount Distribution')
plt.xlabel('Loan Amount')
plt.show()

# 3. Box plot: Loan amount by loan status
df.boxplot(column='loan_amount', by='Status')
plt.title('Loan Amount by Status')
plt.suptitle('')
plt.xlabel('Status')
plt.ylabel('Loan Amount')
plt.show()


#creating schema
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

#to run the schema on dataframe
for col, dtype in schema.items():
    df[col] = df[col].astype(dtype)

#to keep only the required columns
df = df[list(schema.keys())]

print(df.dtypes)  # Verify the data types
# Display the first few rows of the cleaned DataFrame
print(df.head())

