import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


🐼 Pandas Analysis Functions
📊 #Data_Exploration

df.head() – View the first few rows.

df.tail() – View the last few rows.

df.info() – Summary of columns, data types, and missing values.

df.describe() – Descriptive statistics (mean, std, min, max, etc.).

df.shape – Returns the number of rows and columns.

df.columns – List of column names.

df.dtypes – Data types of each column.

df.sample() – Random sample of rows.

df.nunique() – Count of unique values per column.

🧹 #Data_Cleaning

df.isnull() / df.isna() – Detect missing values.

df.notnull() – Identify non-missing values.

df.dropna() – Remove missing values.

df.fillna(value) – Replace missing values.

df.duplicated() – Identify duplicate rows.

df.drop_duplicates() – Remove duplicates.

df.replace(old, new) – Replace specific values.

df.rename(columns={}) – Rename columns.

df.astype() – Convert data types.

df.apply(function) – Apply a custom function.

🔢 #Data_Selection_&_Filtering

df['column'] / df[['col1','col2']] – Select columns.

df.loc[] – Select by label.

df.iloc[] – Select by position.

df.query('condition') – Filter rows using expressions.

df[(df['col'] > value)] – Conditional filtering.

🧮 #Data_Aggregation_&_Statistics

df.sum() / df.mean() / df.median() / df.std() / df.var() – Basic stats.

df.min() / df.max() – Min and max values.

df.count() – Non-null counts.

df.corr() – Correlation matrix.

df.cov() – Covariance.

df.value_counts() – Frequency of unique values.

df.groupby('column').agg(['mean','sum','count']) – Grouped aggregation.

df.pivot_table() – Create summary pivot tables.

🔗 #Merging_&_Joining

pd.concat([df1, df2]) – Combine vertically or horizontally.

pd.merge(df1, df2, on='key') – Join on a common key.

df.join(df2) – Join using index.

📆 #Time_Series

pd.to_datetime(df['date']) – Convert to datetime.

df.set_index('date') – Set datetime as index.

df.resample('M').mean() – Resample data (monthly, yearly, etc.).

df.shift() – Shift index for time-based comparisons.

df.rolling(window).mean() – Rolling average.

📈 #Visualization

(Requires matplotlib/seaborn integration)

df.plot() – Quick plot.

df.hist() – Histogram.

df.boxplot() – Box plot.

df.plot(kind='bar') – Bar chart.

df.plot.scatter(x='col1', y='col2') – Scatter plot.

🔢 NumPy Analysis Functions
🧠 #Array_Creation

np.array() – Create an array.

np.zeros() / np.ones() – Arrays of zeros or ones.

np.arange() – Range of evenly spaced values.

np.linspace() – Linearly spaced values.

np.eye() – Identity matrix.

🧩 #Array_Inspection

arr.shape – Dimensions.

arr.ndim – Number of dimensions.

arr.size – Number of elements.

arr.dtype – Data type.

🔢 #Array_Manipulation

arr.reshape() – Reshape array.

np.transpose(arr) – Transpose matrix.

arr.flatten() – Flatten into 1D.

np.concatenate() – Join arrays.

np.split() – Split array into parts.

np.stack() – Stack arrays.

⚙️ #Mathematical_Operations

np.add() / np.subtract() / np.multiply() / np.divide() – Basic arithmetic.

np.power() / np.sqrt() – Power and square root.

np.exp() / np.log() – Exponential and log.

np.sin() / np.cos() / np.tan() – Trigonometric functions.

📊 #Statistical_Functions

np.mean() / np.median() / np.std() / np.var() – Descriptive statistics.

np.min() / np.max() / np.sum() / np.prod() – Aggregation functions.

np.percentile() / np.quantile() – Percentile calculations.

np.corrcoef() / np.cov() – Correlation and covariance.

🔍 #Logical_&_Comparison

np.where(condition, x, y) – Conditional selection.

np.all() / np.any() – Logical checks.

np.logical_and() / np.logical_or() – Combine conditions.

np.unique() – Unique elements.

🧮 #Linear_Algebra

np.dot() – Dot product.

np.matmul() – Matrix multiplication.

np.linalg.inv() – Matrix inverse.

np.linalg.det() – Determinant.

np.linalg.eig() – Eigenvalues and eigenvectors.

np.linalg.solve() – Solve linear equations.

🎲 #Random_Numbers

np.random.rand() – Uniform random numbers.

np.random.randn() – Normal distribution.

np.random.randint() – Random integers.

np.random.choice() – Random selection.

np.random.seed() – Set random seed for reproducibility.



