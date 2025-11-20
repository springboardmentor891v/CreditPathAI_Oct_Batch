import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from google.colab import drive
drive.mount('/content/drive')

# load dataset
df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Loan_Default.csv')

print(df.info())

print(df.describe())

print(df.head())

print(df.tail())

print("DataFrame Shape:", df.shape)

print("DataFrame Columns:", df.columns.tolist())

# 3️ Missing value inspection & filling
def find_missing_values(df):
    missing_values = df.isnull().sum()
    return missing_values[missing_values > 0]

print("Missing values before:\n", find_missing_values(df))

# 3.1 Heatmap of missing values
def plot_missing_values_heatmap(df):
    """Plot a heatmap showing missing values in the DataFrame."""
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
    plt.title('Missing Values Heatmap')
    plt.tight_layout()
    plt.show()
plot_missing_values_heatmap(df)

def fill_missing_values(df):
    """Fill missing values with median for numeric and mode for categorical."""
    for col in df.columns:
        if df[col].dtype == 'object':
            fill_val = df[col].mode()[0]
        else:
            fill_val = df[col].median()
        df[col] = df[col].fillna(fill_val)

    return df

df = fill_missing_values(df)
print("Missing values after:\n", find_missing_values(df))

# 4️ Remove duplicates
df = df.drop_duplicates()

# 5️ Drop irrelevant columns (safe version)
def drop_irrelevant_columns(df, target='Status', missing_thresh=0.9, low_var_thresh=0.01):
    to_drop = []
    for col in df.columns:
        if col == target:
            continue
        missing_ratio = df[col].isnull().mean()
        if missing_ratio > missing_thresh:
            to_drop.append(col)
        elif df[col].nunique() <= 1:
            to_drop.append(col)
        elif df[col].nunique() / len(df) < low_var_thresh and df[col].nunique() > 2:
            to_drop.append(col)
    df.drop(columns=to_drop, inplace=True)
    print(f"Dropped {len(to_drop)} low-value/missing columns: {to_drop}")
    return df

df = drop_irrelevant_columns(df)

# 6️ Data type conversion
def convert_data_types(df):
    """Convert text to numeric or category as appropriate."""
    binary_map = {'Yes': 1, 'No': 0, 'Y': 1, 'N': 0, 'True': 1, 'False': 0}
    df.replace(binary_map, inplace=True)

    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() <= 20:
            df[col] = df[col].astype('category')

    # Ensure Status is int
    if 'Status' in df.columns:
        df['Status'] = pd.to_numeric(df['Status'], errors='coerce').fillna(0).astype(int)

    return df

df = convert_data_types(df)

# 7️ Encode categorical variables
def encode_categorical_variables(df, target='Status'):
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_cols = [c for c in categorical_cols if c != target]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df

df = encode_categorical_variables(df)

# 8 Handle outliers (before scaling)
def remove_outliers(df, target='Status', iqr_factor=1.5):
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target]
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = np.clip(df[col], lower, upper)  # Cap instead of remove
    return df

df = remove_outliers(df)

# 9 Scale numeric features (after outlier handling)
from sklearn.preprocessing import MinMaxScaler
def scale_numeric_features(df, target='Status'):
    scaler = MinMaxScaler()
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target]
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

df = scale_numeric_features(df)

# 10 Save cleaned data
def save_cleaned_data(df, filepath):
    df.to_csv(filepath, index=False)
    print(f" Cleaned data saved to: {filepath}")

save_cleaned_data(df, "/content/drive/MyDrive/Colab Notebooks/Cleaned_Loan_Default.csv")

# 11 Reload cleaned data
def load_cleaned_data(filepath):
    return pd.read_csv(filepath)

df = load_cleaned_data("/content/drive/MyDrive/Colab Notebooks/Cleaned_Loan_Default.csv")

# 12 Analyze cleaned data
def analyze_data(df):
    print(" Data Summary:\n", df.describe())
    print("\n Target Value Counts:\n", df['Status'].value_counts(normalize=True))
    print("\n Remaining Dtypes:\n", df.dtypes.value_counts())
    print("\n Final Shape:", df.shape)

analyze_data(df)

# Load RAW once
df_raw = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Loan_Default.csv')

def plot_missing_values_heatmap(df, title='Missing Values Heatmap'):
    na_cols = df.columns[df.isna().any()]
    if len(na_cols) == 0:
        print(" No missing values to show.")
        return
    plt.figure(figsize=(12, 6))
    sns.heatmap(df[na_cols].isna(), cbar=True, yticklabels=False, cmap='rocket_r', vmin=0, vmax=1)
    plt.title(title)
    plt.tight_layout()
    plt.show()

# BEFORE imputation
plot_missing_values_heatmap(df_raw, title='Missing (BEFORE fill)')

df_clean = fill_missing_values(df_raw.copy())

# AFTER imputation
plot_missing_values_heatmap(df_clean, title='Missing (AFTER fill)')

# 13 Numeric Summarystatistics
def numeric_summary(df):
    """Return summary statistics for numeric columns in the DataFrame."""
    numeric_cols = df.select_dtypes(include=[np.number])
    return numeric_cols.describe()
numeric_stats = numeric_summary(df)
print("Numeric Summary Statistics:\n", numeric_stats)

# 14 Categorical Summary statistics
def categorical_summary(df, verbose=True):
    """Return summary stats for categorical columns; handle none-present case."""
    cats = df.select_dtypes(include=['object', 'category'])
    if cats.shape[1] == 0:
        if verbose:
            print(" No categorical columns found (likely after encoding).")
        # Return empty DataFrame to avoid ValueError
        return pd.DataFrame(columns=["count","unique","top","freq"])
    summary = cats.describe()
    if verbose:
        print("Categorical Summary Statistics:\n", summary)
    return summary

# 15 Plot distributions for numeric columns
def plot_distributions(df):
    """Plot distributions for numeric columns in the DataFrame."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        plt.figure(figsize=(10, 4))
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()
plot_distributions(df)

# 16 Plot count plots for categorical columns
def plot_categorical_counts(df):
    """Plot count plots for categorical columns in the DataFrame."""
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        plt.figure(figsize=(10, 4))
        sns.countplot(y=df[col], order=df[col].value_counts().index)
        plt.title(f'Count Plot of {col}')
        plt.xlabel('Count')
        plt.ylabel(col)
        plt.tight_layout()
        plt.show()
plot_categorical_counts(df)

# 17 Correlation matrix for numeric columns
def correlation_matrix(df):
    """Plot the correlation matrix for numeric columns in the DataFrame."""
    numeric_cols = df.select_dtypes(include=[np.number])
    corr = numeric_cols.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()
correlation_matrix(df)

# 18 Status vs numeric features(Boxplots)
def plot_box_by_target(df, target='Status'):
    """Plot boxplots of numeric features grouped by the target variable."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != target]
    for col in numeric_cols:
        plt.figure(figsize=(10, 4))
        sns.boxplot(x=df[target], y=df[col])
        plt.title(f'Boxplot of {col} by {target}')
        plt.xlabel(target)
        plt.ylabel(col)
        plt.tight_layout()
        plt.show()
plot_box_by_target(df)

# 19 Pairplot of numeric features
def plot_pairplot(df, target='Status'):
    """Plot pairplot of numeric features colored by the target variable."""
    top_cols = ['loan_amount', 'rate_of_interest', 'LTV','income', target]
    existing = [col for col in top_cols if col in df.columns]
    if len(existing) >= 2:
        sns.pairplot(df[existing], hue=target, diag_kind='kde')
        plt.suptitle('Pairplot of Selected Numeric Features', y=1.02)
        plt.tight_layout()
        plt.show()
    else:
        print("Not enough numeric columns for pairplot.")
plot_pairplot(df)

# 20 interactive plots with plotly
import plotly.express as px
def interactive_scatter_plot(df, x, y, color=None):
    """Create an interactive scatter plot using Plotly."""
    if 'loan_amount' in df.columns and 'income' in df.columns:
        fig = px.scatter(df, x=x, y=y, color=color, title=f'Interactive Scatter Plot of {y} vs {x}')
        fig.show()
    if 'rate_of_interest' in df.columns and 'LTV' in df.columns:
        fig = px.scatter(df, x='rate_of_interest', y='LTV', color=color, title='Interactive Scatter Plot of LTV vs Rate of Interest')
        fig.show()
    if 'loan_amount' in df.columns and 'Status' in df.columns:
        fig = px.histogram(df, x='loan_amount', color=color, title='Interactive Histogram of Loan Amount by Status', barmode='group')
        fig.show()
    if 'rate_of_interest' in df.columns and 'Status' in df.columns:
        fig = px.histogram(df, x='rate_of_interest', color=color, title='Interactive Histogram of Rate of Interest by Status', barmode='group')
        fig.show()
    if 'rate_of_interest' in df.columns and 'income' in df.columns:
        fig = px.box(df, x=color, y='rate_of_interest', title='Interactive Box Plot of Rate of Interest by Status')
        fig.show()
interactive_scatter_plot(df, x='loan_amount', y='income', color='Status')

# 21 Save visualizations
def save_visualizations(df):
    """Save key visualizations as image files."""
    # Example: Save correlation matrix
    numeric_cols = df.select_dtypes(include=[np.number])
    corr = numeric_cols.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig("correlation_matrix.png")
    plt.close()
    print(" Visualizations saved as image files.")
save_visualizations(df)

# 22 Export summary statistics
def export_summary_statistics(df, numeric_file='numeric_summary.csv', categorical_file='categorical_summary.csv'):
    """Export summary statistics to CSV files."""
    numeric_stats = numeric_summary(df)
    numeric_stats.to_csv(numeric_file)
    categorical_stats = categorical_summary(df)
    categorical_stats.to_csv(categorical_file)
    print(f" Summary statistics exported: {numeric_file}, {categorical_file}")
export_summary_statistics(df)

#train test split
from sklearn.model_selection import train_test_split
X = df.drop('Status', axis=1)
y = df['Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# Save train and test sets
def save_train_test_sets(X_train, X_test, y_train, y_test, prefix='loan_default'):
    """Save train and test sets to CSV files."""
    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)
    train.to_csv(f"{prefix}_train.csv", index=False)
    test.to_csv(f"{prefix}_test.csv", index=False)
    print(f" Train and test sets saved: {prefix}_train.csv, {prefix}_test.csv")
save_train_test_sets(X_train, X_test, y_train, y_test)

model_results = []
roc_data = {}
evaluation_results = []

#train linear regression model as a test
from sklearn.linear_model import LinearRegression
def train_linear_regression(X_train, y_train, X_test, y_test):
    """Train and evaluate a linear regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f" Linear Regression Train R^2: {train_score:.4f}, Test R^2: {test_score:.4f}")

    #  metrics & plots
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # Predictions on test set
    y_pred = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)   # redundant if you already printed .score(), but safe

    print(f" MAE:  {mae:.4f}")
    print(f" MSE:  {mse:.4f}")
    print(f" RMSE: {rmse:.4f}")
    print(f" R2 (test): {r2:.4f}")

    # Predicted vs Actual scatter
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', linewidth=1)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Linear Regression — Predicted vs Actual")
    plt.tight_layout()
    plt.show()

    # Residuals histogram + KDE
    residuals = y_test - y_pred
    plt.figure(figsize=(6,4))
    sns.histplot(residuals, kde=True)
    plt.xlabel("Residual (Actual - Predicted)")
    plt.title("Residuals Distribution")
    plt.tight_layout()
    plt.show()

    model_results.append(["Linear Regression", test_score])


train_linear_regression(X_train, y_train, X_test, y_test)

from sklearn.linear_model import LogisticRegression

def train_logistic_regression(X_train, y_train, X_test, y_test):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
    import matplotlib.pyplot as plt
    import seaborn as sns

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"LOGISTIC REGRESSION")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Logistic Regression - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1],[0,1],"k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Logistic Regression - ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()

    y_train_prob = model.predict_proba(X_train)[:, 1]
    y_train_pred = (y_train_prob >= 0.5).astype(int)

    train_precision = precision_score(y_train, y_train_pred, zero_division=0)
    train_recall    = recall_score(y_train, y_train_pred, zero_division=0)
    train_f1        = f1_score(y_train, y_train_pred, zero_division=0)
    train_auc       = roc_auc_score(y_train, y_train_prob)

    print(f" Train Accuracy: {train_score:.4f}, Test Accuracy: {test_score:.4f}")
    print(f" Train Precision: {train_precision:.4f}, Test Precision: {precision:.4f}")
    print(f" Train Recall: {train_recall:.4f}, Test Recall: {recall:.4f}")
    print(f" Train F1 Score: {train_f1:.4f}, Test F1 Score: {f1:.4f}")
    print(f" Train ROC-AUC: {train_auc:.4f}, Test ROC-AUC: {roc_auc:.4f}")

    model_results.append(["Logistic Regression", test_score])

    evaluation_results.append([
        "Logistic Regression",
        train_score, train_precision, train_recall, train_f1, train_auc,
        test_score, precision, recall, f1, roc_auc
    ])

train_logistic_regression(X_train, y_train, X_test, y_test)

from sklearn.tree import DecisionTreeClassifier

def train_decision_tree(X_train, y_train, X_test, y_test):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    print("Decision Tree")

    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
    import matplotlib.pyplot as plt, seaborn as sns

    y_pred = model.predict(X_test)

    try:
        y_score = model.predict_proba(X_test)[:, 1]
    except:
        try:
            y_score = model.decision_function(X_test)
        except:
            y_score = None

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_score) if y_score is not None else float('nan')

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Decision Tree - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    if y_score is not None:
        fpr, tpr, _ = roc_curve(y_test, y_score)
        plt.figure(figsize=(5,4))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0,1],[0,1],'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Decision Tree - ROC Curve")
        plt.legend()
        plt.tight_layout()
        plt.show()

    y_train_pred = model.predict(X_train)

    try:
        y_train_prob = model.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, y_train_prob)
    except:
        train_auc = float('nan')

    train_precision = precision_score(y_train, y_train_pred, zero_division=0)
    train_recall    = recall_score(y_train, y_train_pred, zero_division=0)
    train_f1        = f1_score(y_train, y_train_pred, zero_division=0)

    print(f" Train Accuracy: {train_score:.4f}, Test Accuracy: {test_score:.4f}")
    print(f" Train Precision: {train_precision:.4f}, Test Precision: {precision:.4f}")
    print(f" Train Recall: {train_recall:.4f}, Test Recall: {recall:.4f}")
    print(f" Train F1 Score: {train_f1:.4f}, Test F1 Score: {f1:.4f}")
    print(f" Train ROC-AUC: {train_auc:.4f}, Test ROC-AUC: {roc_auc:.4f}")

    model_results.append(["Decision Tree", test_score])

    evaluation_results.append([
        "Decision Tree",
        train_score, train_precision, train_recall, train_f1, train_auc,
        test_score, precision, recall, f1, roc_auc
    ])

train_decision_tree(X_train, y_train, X_test, y_test)

#train random forest model as a test
from sklearn.ensemble import RandomForestClassifier
def train_random_forest(X_train, y_train, X_test, y_test):
    """Train and evaluate a random forest classifier."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f" Random Forest Train Accuracy: {train_score:.4f}, Test Accuracy: {test_score:.4f}")

    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
    import matplotlib.pyplot as plt, seaborn as sns
    y_pred = model.predict(X_test)
    try:
        y_score = model.predict_proba(X_test)[:, 1]
    except Exception:
        try:
            y_score = model.decision_function(X_test)
        except Exception:
            y_score = None

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    print(f" Precision: {precision:.4f}")
    print(f" Recall:    {recall:.4f}")
    print(f" F1 Score:  {f1:.4f}")
    if y_score is not None:
        roc_auc = roc_auc_score(y_test, y_score)
        print(f" ROC-AUC:   {roc_auc:.4f}")
    else:
        print(" ROC-AUC:   N/A (no proba/score)")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Random Forest - Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.tight_layout(); plt.show()

    if y_score is not None:
        fpr, tpr, _ = roc_curve(y_test, y_score)
        plt.figure(figsize=(5,4))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0,1],[0,1],'k--')
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title("Random Forest - ROC")
        plt.legend(); plt.tight_layout(); plt.show()

    model_results.append(["Random Forest", test_score])


train_random_forest(X_train, y_train, X_test, y_test)

from sklearn.ensemble import RandomForestClassifier

def train_random_forest(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print("Random Forest")

    train_score = model.score(X_train, y_train)
    test_score  = model.score(X_test, y_test)

    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
    import matplotlib.pyplot as plt, seaborn as sns

    y_pred = model.predict(X_test)

    try:
        y_score = model.predict_proba(X_test)[:, 1]
    except:
        try:
            y_score = model.decision_function(X_test)
        except:
            y_score = None

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    roc_auc   = roc_auc_score(y_test, y_score) if y_score is not None else float('nan')

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Random Forest - Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.tight_layout(); plt.show()

    if y_score is not None:
        fpr, tpr, _ = roc_curve(y_test, y_score)
        plt.figure(figsize=(5,4))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0,1],[0,1],'k--')
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title("Random Forest - ROC Curve")
        plt.legend(); plt.tight_layout(); plt.show()

    y_train_pred = model.predict(X_train)

    try:
        y_train_prob = model.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, y_train_prob)
    except:
        train_auc = float('nan')

    train_precision = precision_score(y_train, y_train_pred, zero_division=0)
    train_recall    = recall_score(y_train, y_train_pred, zero_division=0)
    train_f1        = f1_score(y_train, y_train_pred, zero_division=0)

    print(f" Train Accuracy: {train_score:.4f}, Test Accuracy: {test_score:.4f}")
    print(f" Train Precision: {train_precision:.4f}, Test Precision: {precision:.4f}")
    print(f" Train Recall: {train_recall:.4f}, Test Recall: {recall:.4f}")
    print(f" Train F1 Score: {train_f1:.4f}, Test F1 Score: {f1:.4f}")
    print(f" Train ROC-AUC: {train_auc:.4f}, Test ROC-AUC: {roc_auc:.4f}")

    model_results.append(["Random Forest", test_score])

    evaluation_results.append([
        "Random Forest",
        train_score, train_precision, train_recall, train_f1, train_auc,
        test_score, precision, recall, f1, roc_auc
    ])
train_random_forest(X_train, y_train, X_test, y_test)

#hyperparameter tuning with grid search for random forest
from sklearn.model_selection import GridSearchCV
def hyperparameter_tuning_rf(X_train, y_train):
    """Perform hyperparameter tuning for Random Forest using GridSearchCV."""
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    print(" Best parameters found:", grid_search.best_params_)
    print(" Best cross-validation accuracy:", grid_search.best_score_)
hyperparameter_tuning_rf(X_train, y_train)


from sklearn.neighbors import KNeighborsClassifier

def train_knn(X_train, y_train, X_test, y_test, n_neighbors=5):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)

    print("KNN")

    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
    import matplotlib.pyplot as plt, seaborn as sns

    y_pred = model.predict(X_test)

    try:
        y_score = model.predict_proba(X_test)[:, 1]
    except:
        try:
            y_score = model.decision_function(X_test)
        except:
            y_score = None

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_score) if y_score is not None else float('nan')

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("KNN - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    if y_score is not None:
        fpr, tpr, _ = roc_curve(y_test, y_score)
        plt.figure(figsize=(5,4))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0,1],[0,1],'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("KNN - ROC Curve")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # TRAIN METRICS
    y_train_pred = model.predict(X_train)
    try:
        y_train_prob = model.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, y_train_prob)
    except:
        train_auc = float('nan')

    train_precision = precision_score(y_train, y_train_pred, zero_division=0)
    train_recall = recall_score(y_train, y_train_pred, zero_division=0)
    train_f1 = f1_score(y_train, y_train_pred, zero_division=0)

    print(f" Train Accuracy: {train_score:.4f}, Test Accuracy: {test_score:.4f}")
    print(f" Train Precision: {train_precision:.4f}, Test Precision: {precision:.4f}")
    print(f" Train Recall: {train_recall:.4f}, Test Recall: {recall:.4f}")
    print(f" Train F1 Score: {train_f1:.4f}, Test F1 Score: {f1:.4f}")
    print(f" Train ROC-AUC: {train_auc:.4f}, Test ROC-AUC: {roc_auc:.4f}")

    model_results.append(["KNN", test_score])

    evaluation_results.append([
        "KNN",
        train_score, train_precision, train_recall, train_f1, train_auc,
        test_score, precision, recall, f1, roc_auc
    ])
train_knn(X_train, y_train, X_test, y_test, n_neighbors=5)

from sklearn.naive_bayes import GaussianNB

def train_naive_bayes(X_train, y_train, X_test, y_test):
    model = GaussianNB()
    model.fit(X_train, y_train)

    print("Naive Bayes")

    train_score = model.score(X_train, y_train)
    test_score  = model.score(X_test, y_test)

    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
    import matplotlib.pyplot as plt, seaborn as sns

    y_pred = model.predict(X_test)

    try:
        y_score = model.predict_proba(X_test)[:, 1]
    except:
        try:
            y_score = model.decision_function(X_test)
        except:
            y_score = None

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    roc_auc   = roc_auc_score(y_test, y_score) if y_score is not None else float('nan')

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Naive Bayes - Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.tight_layout(); plt.show()

    if y_score is not None:
        fpr, tpr, _ = roc_curve(y_test, y_score)
        plt.figure(figsize=(5,4))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0,1],[0,1], 'k--')
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title("Naive Bayes - ROC Curve")
        plt.legend(); plt.tight_layout(); plt.show()

    # TRAIN METRICS
    y_train_pred = model.predict(X_train)
    try:
        y_train_prob = model.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, y_train_prob)
    except:
        train_auc = float('nan')

    train_precision = precision_score(y_train, y_train_pred, zero_division=0)
    train_recall    = recall_score(y_train, y_train_pred, zero_division=0)
    train_f1        = f1_score(y_train, y_train_pred, zero_division=0)

    print(f" Train Accuracy: {train_score:.4f}, Test Accuracy: {test_score:.4f}")
    print(f" Train Precision: {train_precision:.4f}, Test Precision: {precision:.4f}")
    print(f" Train Recall: {train_recall:.4f}, Test Recall: {recall:.4f}")
    print(f" Train F1 Score: {train_f1:.4f}, Test F1 Score: {f1:.4f}")
    print(f" Train ROC-AUC: {train_auc:.4f}, Test ROC-AUC: {roc_auc:.4f}")

    model_results.append(["Naive Bayes", test_score])

    evaluation_results.append([
        "Naive Bayes",
        train_score, train_precision, train_recall, train_f1, train_auc,
        test_score, precision, recall, f1, roc_auc
    ])
train_naive_bayes(X_train, y_train, X_test, y_test)

#handle class imbalance with SMOTE
from imblearn.over_sampling import SMOTE
def apply_smote(X_train, y_train):
    """Apply SMOTE to balance the training dataset."""
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(" After SMOTE, class distribution:\n", pd.Series(y_res).value_counts())
    return X_res, y_res
X_train, y_train = apply_smote(X_train, y_train)

from imblearn.ensemble import BalancedRandomForestClassifier

def train_balanced_random_forest(X_train, y_train, X_test, y_test):
    model = BalancedRandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print("Balanced Random Forest")

    train_score = model.score(X_train, y_train)
    test_score  = model.score(X_test, y_test)

    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
    import matplotlib.pyplot as plt, seaborn as sns

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    roc_auc   = roc_auc_score(y_test, y_prob)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Balanced Random Forest - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Balanced Random Forest - ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # TRAIN METRICS
    y_train_pred = model.predict(X_train)
    y_train_prob = model.predict_proba(X_train)[:, 1]

    train_precision = precision_score(y_train, y_train_pred, zero_division=0)
    train_recall    = recall_score(y_train, y_train_pred, zero_division=0)
    train_f1        = f1_score(y_train, y_train_pred, zero_division=0)
    train_auc       = roc_auc_score(y_train, y_train_prob)

    print(f" Train Accuracy: {train_score:.4f}, Test Accuracy: {test_score:.4f}")
    print(f" Train Precision: {train_precision:.4f}, Test Precision: {precision:.4f}")
    print(f" Train Recall: {train_recall:.4f}, Test Recall: {recall:.4f}")
    print(f" Train F1 Score: {train_f1:.4f}, Test F1 Score: {f1:.4f}")
    print(f" Train ROC-AUC: {train_auc:.4f}, Test ROC-AUC: {roc_auc:.4f}")

    model_results.append(["Balanced Random Forest", test_score])

    evaluation_results.append([
        "Balanced Random Forest",
        train_score, train_precision, train_recall, train_f1, train_auc,
        test_score, precision, recall, f1, roc_auc
    ])

    roc_data["Balanced Random Forest"] = (fpr, tpr, roc_auc)
train_balanced_random_forest(X_train, y_train, X_test, y_test)

!pip install catboost
!pip install lightgbm
!pip install xgboost

import xgboost as xgb

def train_xgboost(X_train, y_train, X_test, y_test):
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)

    print("XGBoost")

    train_score = model.score(X_train, y_train)
    test_score  = model.score(X_test, y_test)

    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
    import matplotlib.pyplot as plt, seaborn as sns

    y_pred = model.predict(X_test)

    try:
        y_score = model.predict_proba(X_test)[:, 1]
    except:
        try:
            y_score = model.decision_function(X_test)
        except:
            y_score = None

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    roc_auc   = roc_auc_score(y_test, y_score) if y_score is not None else float('nan')

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("XGBoost - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    if y_score is not None:
        fpr, tpr, _ = roc_curve(y_test, y_score)
        plt.figure(figsize=(5,4))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0,1], [0,1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("XGBoost - ROC Curve")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Train metrics
    y_train_pred = model.predict(X_train)

    try:
        y_train_prob = model.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, y_train_prob)
    except:
        train_auc = float('nan')

    train_precision = precision_score(y_train, y_train_pred, zero_division=0)
    train_recall    = recall_score(y_train, y_train_pred, zero_division=0)
    train_f1        = f1_score(y_train, y_train_pred, zero_division=0)

    print(f" Train Accuracy: {train_score:.4f}, Test Accuracy: {test_score:.4f}")
    print(f" Train Precision: {train_precision:.4f}, Test Precision: {precision:.4f}")
    print(f" Train Recall: {train_recall:.4f}, Test Recall: {recall:.4f}")
    print(f" Train F1 Score: {train_f1:.4f}, Test F1 Score: {f1:.4f}")
    print(f" Train ROC-AUC: {train_auc:.4f}, Test ROC-AUC: {roc_auc:.4f}")

    model_results.append(["XGBoost", test_score])

    evaluation_results.append([
        "XGBoost",
        train_score, train_precision, train_recall, train_f1, train_auc,
        test_score, precision, recall, f1, roc_auc
    ])
train_xgboost(X_train, y_train, X_test, y_test)

from catboost import CatBoostClassifier

def train_catboost(X_train, y_train, X_test, y_test):
    model = CatBoostClassifier(verbose=0, random_state=42)
    model.fit(X_train, y_train)

    print("CatBoost")

    train_score = model.score(X_train, y_train)
    test_score  = model.score(X_test, y_test)

    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
    import matplotlib.pyplot as plt, seaborn as sns

    y_pred = model.predict(X_test)

    try:
        y_score = model.predict_proba(X_test)[:, 1]
    except:
        try:
            y_score = model.predict(X_test, prediction_type='Probability')[:, 1]
        except:
            y_score = None

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    roc_auc   = roc_auc_score(y_test, y_score) if y_score is not None else float('nan')

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("CatBoost - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    if y_score is not None:
        fpr, tpr, _ = roc_curve(y_test, y_score)
        plt.figure(figsize=(5,4))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0,1], [0,1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("CatBoost - ROC Curve")
        plt.legend()
        plt.tight_layout()
        plt.show()

    y_train_pred = model.predict(X_train)

    try:
        y_train_prob = model.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, y_train_prob)
    except:
        train_auc = float('nan')

    train_precision = precision_score(y_train, y_train_pred, zero_division=0)
    train_recall    = recall_score(y_train, y_train_pred, zero_division=0)
    train_f1        = f1_score(y_train, y_train_pred, zero_division=0)

    print(f" Train Accuracy: {train_score:.4f}, Test Accuracy: {test_score:.4f}")
    print(f" Train Precision: {train_precision:.4f}, Test Precision: {precision:.4f}")
    print(f" Train Recall: {train_recall:.4f}, Test Recall: {recall:.4f}")
    print(f" Train F1 Score: {train_f1:.4f}, Test F1 Score: {f1:.4f}")
    print(f" Train ROC-AUC: {train_auc:.4f}, Test ROC-AUC: {roc_auc:.4f}")

    model_results.append(["CatBoost", test_score])

    evaluation_results.append([
        "CatBoost",
        train_score, train_precision, train_recall, train_f1, train_auc,
        test_score, precision, recall, f1, roc_auc
    ])
train_catboost(X_train, y_train, X_test, y_test)

import lightgbm as lgb

def train_lightgbm(X_train, y_train, X_test, y_test):
    model = lgb.LGBMClassifier(random_state=42)
    model.fit(X_train, y_train)

    print("LightGBM")

    train_score = model.score(X_train, y_train)
    test_score  = model.score(X_test, y_test)

    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
    import matplotlib.pyplot as plt, seaborn as sns

    y_pred = model.predict(X_test)

    try:
        y_score = model.predict_proba(X_test)[:, 1]
    except:
        try:
            y_score = model.predict(X_test)
        except:
            y_score = None

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    roc_auc   = roc_auc_score(y_test, y_score) if (y_score is not None and len(set(y_score)) > 1) else float('nan')

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("LightGBM - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    if y_score is not None:
        try:
            fpr, tpr, _ = roc_curve(y_test, y_score)
            plt.figure(figsize=(5,4))
            plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
            plt.plot([0,1], [0,1], 'k--')
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("LightGBM - ROC Curve")
            plt.legend()
            plt.tight_layout()
            plt.show()
        except:
            pass

    y_train_pred = model.predict(X_train)

    try:
        y_train_prob = model.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, y_train_prob)
    except:
        train_auc = float('nan')

    train_precision = precision_score(y_train, y_train_pred, zero_division=0)
    train_recall    = recall_score(y_train, y_train_pred, zero_division=0)
    train_f1        = f1_score(y_train, y_train_pred, zero_division=0)

    print(f" Train Accuracy: {train_score:.4f}, Test Accuracy: {test_score:.4f}")
    print(f" Train Precision: {train_precision:.4f}, Test Precision: {precision:.4f}")
    print(f" Train Recall: {train_recall:.4f}, Test Recall: {recall:.4f}")
    print(f" Train F1 Score: {train_f1:.4f}, Test F1 Score: {f1:.4f}")
    print(f" Train ROC-AUC: {train_auc:.4f}, Test ROC-AUC: {roc_auc:.4f}")

    model_results.append(["LightGBM", test_score])

    evaluation_results.append([
        "LightGBM",
        train_score, train_precision, train_recall, train_f1, train_auc,
        test_score, precision, recall, f1, roc_auc
    ])
train_lightgbm(X_train, y_train, X_test, y_test)

from sklearn.ensemble import VotingClassifier

def train_ensemble_model(X_train, y_train, X_test, y_test):
    model1 = LogisticRegression(max_iter=1000)
    model2 = DecisionTreeClassifier(random_state=42)
    model3 = RandomForestClassifier(n_estimators=100, random_state=42)

    ensemble = VotingClassifier(
        estimators=[('lr', model1), ('dt', model2), ('rf', model3)],
        voting='hard'
    )

    ensemble.fit(X_train, y_train)

    print("Voting Classifier")

    train_score = ensemble.score(X_train, y_train)
    test_score  = ensemble.score(X_test, y_test)

    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
    import matplotlib.pyplot as plt, seaborn as sns

    y_pred = ensemble.predict(X_test)

    try:
        y_prob = ensemble.predict_proba(X_test)[:, 1]
    except:
        y_prob = y_pred

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    roc_auc   = roc_auc_score(y_test, y_prob)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Voting Classifier - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Voting Classifier - ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # TRAIN METRICS
    y_train_pred = ensemble.predict(X_train)

    try:
        y_train_prob = ensemble.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, y_train_prob)
    except:
        train_auc = float('nan')

    train_precision = precision_score(y_train, y_train_pred, zero_division=0)
    train_recall    = recall_score(y_train, y_train_pred, zero_division=0)
    train_f1        = f1_score(y_train, y_train_pred, zero_division=0)

    print(f" Train Accuracy: {train_score:.4f}, Test Accuracy: {test_score:.4f}")
    print(f" Train Precision: {train_precision:.4f}, Test Precision: {precision:.4f}")
    print(f" Train Recall: {train_recall:.4f}, Test Recall: {recall:.4f}")
    print(f" Train F1 Score: {train_f1:.4f}, Test F1 Score: {f1:.4f}")
    print(f" Train ROC-AUC: {train_auc:.4f}, Test ROC-AUC: {roc_auc:.4f}")

    model_results.append(["Voting Classifier", test_score])

    evaluation_results.append([
        "Voting Classifier",
        train_score, train_precision, train_recall, train_f1, train_auc,
        test_score, precision, recall, f1, roc_auc
    ])

    roc_data["Voting Classifier"] = (fpr, tpr, roc_auc)
train_ensemble_model(X_train, y_train, X_test, y_test)

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def train_stacking_model(X_train, y_train, X_test, y_test):
    estimators = [
        ('lr', LogisticRegression(max_iter=1000)),
        ('dt', DecisionTreeClassifier(random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
    ]

    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(),
        cv=5,
        n_jobs=-1
    )

    stacking.fit(X_train, y_train)

    print("Stacking Classifier")

    train_score = stacking.score(X_train, y_train)
    test_score  = stacking.score(X_test, y_test)

    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
    import matplotlib.pyplot as plt, seaborn as sns
    import numpy as np

    y_pred = stacking.predict(X_test)

    try:
        y_score = stacking.predict_proba(X_test)[:, 1]
    except:
        try:
            y_score = stacking.decision_function(X_test)
        except:
            y_score = None

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    roc_auc   = roc_auc_score(y_test, y_score) if y_score is not None else float('nan')

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Stacking - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    if y_score is not None:
        try:
            fpr, tpr, _ = roc_curve(y_test, y_score)
            plt.figure(figsize=(5,4))
            plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
            plt.plot([0,1], [0,1], 'k--')
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Stacking - ROC Curve")
            plt.legend()
            plt.tight_layout()
            plt.show()
            roc_data["Stacking Classifier"] = (fpr, tpr, roc_auc)
        except:
            roc_data["Stacking Classifier"] = (None, None, np.nan)
    else:
        roc_data["Stacking Classifier"] = (None, None, np.nan)

    # TRAIN METRICS
    y_train_pred = stacking.predict(X_train)

    try:
        y_train_prob = stacking.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, y_train_prob)
    except:
        train_auc = float('nan')

    train_precision = precision_score(y_train, y_train_pred, zero_division=0)
    train_recall    = recall_score(y_train, y_train_pred, zero_division=0)
    train_f1        = f1_score(y_train, y_train_pred, zero_division=0)

    print(f" Train Accuracy: {train_score:.4f}, Test Accuracy: {test_score:.4f}")
    print(f" Train Precision: {train_precision:.4f}, Test Precision: {precision:.4f}")
    print(f" Train Recall: {train_recall:.4f}, Test Recall: {recall:.4f}")
    print(f" Train F1 Score: {train_f1:.4f}, Test F1 Score: {f1:.4f}")
    print(f" Train ROC-AUC: {train_auc:.4f}, Test ROC-AUC: {roc_auc:.4f}")

    model_results.append(["Stacking Classifier", test_score])

    evaluation_results.append([
        "Stacking Classifier",
        train_score, train_precision, train_recall, train_f1, train_auc,
        test_score, precision, recall, f1, roc_auc
    ])
train_stacking_model(X_train, y_train, X_test, y_test)

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

def train_bagging_model(X_train, y_train, X_test, y_test):
    base_model = DecisionTreeClassifier(random_state=42)
    bagging = BaggingClassifier(estimator=base_model, n_estimators=50, random_state=42)
    bagging.fit(X_train, y_train)

    print("Bagging Classifier")

    train_score = bagging.score(X_train, y_train)
    test_score  = bagging.score(X_test, y_test)

    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
    import matplotlib.pyplot as plt, seaborn as sns
    import numpy as np

    y_pred = bagging.predict(X_test)

    try:
        y_prob = bagging.predict_proba(X_test)[:, 1]
    except:
        y_prob = y_pred

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    roc_auc   = roc_auc_score(y_test, y_prob)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Bagging Classifier - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Bagging Classifier - ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()

    roc_data["Bagging Classifier"] = (fpr, tpr, roc_auc)

    # ==== TRAIN METRICS ====
    y_train_pred = bagging.predict(X_train)
    try:
        y_train_prob = bagging.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, y_train_prob)
    except:
        train_auc = float('nan')

    train_precision = precision_score(y_train, y_train_pred, zero_division=0)
    train_recall    = recall_score(y_train, y_train_pred, zero_division=0)
    train_f1        = f1_score(y_train, y_train_pred, zero_division=0)

    print(f" Train Accuracy: {train_score:.4f}, Test Accuracy: {test_score:.4f}")
    print(f" Train Precision: {train_precision:.4f}, Test Precision: {precision:.4f}")
    print(f" Train Recall: {train_recall:.4f}, Test Recall: {recall:.4f}")
    print(f" Train F1 Score: {train_f1:.4f}, Test F1 Score: {f1:.4f}")
    print(f" Train ROC-AUC: {train_auc:.4f}, Test ROC-AUC: {roc_auc:.4f}")

    model_results.append(["Bagging Classifier", test_score])

    evaluation_results.append([
        "Bagging Classifier",
        train_score, train_precision, train_recall, train_f1, train_auc,
        test_score, precision, recall, f1, roc_auc
    ])
train_bagging_model(X_train, y_train, X_test, y_test)

from sklearn.neural_network import MLPClassifier

def train_mlp(X_train, y_train, X_test, y_test):
    model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
    model.fit(X_train, y_train)

    print("MLP Classifier")

    train_score = model.score(X_train, y_train)
    test_score  = model.score(X_test, y_test)

    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
    import matplotlib.pyplot as plt, seaborn as sns
    import numpy as np

    y_pred = model.predict(X_test)

    try:
        y_prob = model.predict_proba(X_test)[:, 1]
    except:
        try:
            y_prob = model.decision_function(X_test)
        except:
            y_prob = y_pred

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    roc_auc   = roc_auc_score(y_test, y_prob)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("MLP Classifier - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("MLP Classifier - ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()

    roc_data["MLP Classifier"] = (fpr, tpr, roc_auc)

    # ==== TRAIN METRICS ====
    y_train_pred = model.predict(X_train)

    try:
        y_train_prob = model.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, y_train_prob)
    except:
        train_auc = float('nan')

    train_precision = precision_score(y_train, y_train_pred, zero_division=0)
    train_recall    = recall_score(y_train, y_train_pred, zero_division=0)
    train_f1        = f1_score(y_train, y_train_pred, zero_division=0)

    print(f" Train Accuracy: {train_score:.4f}, Test Accuracy: {test_score:.4f}")
    print(f" Train Precision: {train_precision:.4f}, Test Precision: {precision:.4f}")
    print(f" Train Recall: {train_recall:.4f}, Test Recall: {recall:.4f}")
    print(f" Train F1 Score: {train_f1:.4f}, Test F1 Score: {f1:.4f}")
    print(f" Train ROC-AUC: {train_auc:.4f}, Test ROC-AUC: {roc_auc:.4f}")

    model_results.append(["MLP Classifier", test_score])

    evaluation_results.append([
        "MLP Classifier",
        train_score, train_precision, train_recall, train_f1, train_auc,
        test_score, precision, recall, f1, roc_auc
    ])
train_mlp(X_train, y_train, X_test, y_test)

import tensorflow as tf

def train_simple_nn(X_train, y_train, X_test, y_test):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

    print("Simple ANN")

    # Train/Test Accuracy
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    test_loss,  test_acc  = model.evaluate(X_test, y_test, verbose=0)

    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
    import matplotlib.pyplot as plt, seaborn as sns
    import numpy as np

    # Test predictions
    y_prob = model.predict(X_test).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    roc_auc   = roc_auc_score(y_test, y_prob)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Simple ANN - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Simple ANN - ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()

    roc_data["Simple ANN"] = (fpr, tpr, roc_auc)

    # ==== TRAIN METRICS ====
    y_train_prob = model.predict(X_train).ravel()
    y_train_pred = (y_train_prob >= 0.5).astype(int)

    train_precision = precision_score(y_train, y_train_pred, zero_division=0)
    train_recall    = recall_score(y_train, y_train_pred, zero_division=0)
    train_f1        = f1_score(y_train, y_train_pred, zero_division=0)

    try:
        train_auc = roc_auc_score(y_train, y_train_prob)
    except:
        train_auc = float('nan')

    print(f" Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")
    print(f" Train Precision: {train_precision:.4f}, Test Precision: {precision:.4f}")
    print(f" Train Recall: {train_recall:.4f}, Test Recall: {recall:.4f}")
    print(f" Train F1 Score: {train_f1:.4f}, Test F1 Score: {f1:.4f}")
    print(f" Train ROC-AUC: {train_auc:.4f}, Test ROC-AUC: {roc_auc:.4f}")

    model_results.append(["Simple ANN", test_acc])

    evaluation_results.append([
        "Simple ANN",
        train_acc, train_precision, train_recall, train_f1, train_auc,
        test_acc, precision, recall, f1, roc_auc
    ])
train_simple_nn(X_train, y_train, X_test, y_test)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

acc_df = pd.DataFrame(model_results, columns=["Model", "Testing Accuracy"])

display(acc_df.sort_values(by="Testing Accuracy", ascending=False))

plt.figure(figsize=(12,6))
sns.barplot(data=acc_df.sort_values(by="Testing Accuracy", ascending=False),
            x="Model", y="Testing Accuracy", palette="viridis")
plt.title("Model Comparison – Testing Accuracy")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

import pandas as pd

columns = [
    "Model",
    "Train Accuracy", "Train Precision", "Train Recall", "Train F1 Score", "Train ROC-AUC",
    "Test Accuracy", "Test Precision", "Test Recall", "Test F1 Score", "Test ROC-AUC"
]

eval_df = pd.DataFrame(evaluation_results, columns=columns)

# Sort by highest test accuracy
eval_df = eval_df.sort_values(by="Test Accuracy", ascending=False)

display(eval_df.style.background_gradient(cmap='Blues'))