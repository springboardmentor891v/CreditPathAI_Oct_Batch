import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
import joblib

# 1. Load and merge data
df = pd.read_csv("Loan_default")
# 2. Create Default target
print(df['loanStatus'].value_counts())
df['Default'] = df['loanStatus'].apply(lambda x: 1 if x in ['Default', 'Charged Off'] else 0)

# 3. Initial Exploration
print(df.head())
print(f"Dataset Shape (Rows, Columns): {df.shape}")
print(df.info())
print(df.describe())
print(df.tail())
print("\nMissing Values per Column:\n", df.isnull().sum())
print("\nDuplicate Rows:", df.duplicated().sum())

target_col = 'Default'
print("\nDefault Value Counts:\n", df[target_col].value_counts(normalize=True))

# 4. EDA Plots
sns.countplot(x=target_col, data=df, palette='Set2')
plt.title("Loan Default Class Distribution")
plt.xlabel("Default (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop([target_col, 'LoanID'], errors='ignore').tolist()
num_cols = len(numeric_cols)
rows = math.ceil(num_cols / 3)
plt.figure(figsize=(20, 4 * rows))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(rows, 3, i)
    sns.histplot(df[col], kde=True, bins=30, color='skyblue')
    plt.title(f"Distribution of {col}")
    plt.tight_layout()
plt.show()

plt.figure(figsize=(25,30))
for i, col in enumerate(numeric_cols[:6], 1):
    plt.subplot(3, 3, i)
    sns.boxplot(x=target_col, y=col, data=df, palette='coolwarm')
    plt.title(f"{col} vs Default")
    plt.tight_layout()
plt.show()

categorical_cols = df.select_dtypes(include=['object']).columns.drop(['LoanID'], errors='ignore').tolist()
plt.figure(figsize=(25,30 ))
for i, col in enumerate(categorical_cols, 1):
    plt.subplot(4, 2, i)
    sns.countplot(x=col, data=df, hue=target_col, palette='Set3')
    plt.title(f"{col} vs Default")
    plt.xticks(rotation=45)
    plt.tight_layout()
plt.show()

# Optional scatterplots for known columns
scatter_x = 'CreditScore' if 'CreditScore' in df.columns else numeric_cols[0]
scatter_y = 'LoanAmount' if 'LoanAmount' in df.columns else numeric_cols[1]
plt.figure(figsize=(12, 10))
sns.scatterplot(x=scatter_x, y=scatter_y, hue=target_col, data=df, palette='coolwarm')
plt.title("Credit Score vs Loan Amount (by Default Status)")
plt.show()

scatter_x = 'Income' if 'Income' in df.columns else numeric_cols[2]
scatter_y = 'DTIRatio' if 'DTIRatio' in df.columns else numeric_cols[3]
plt.figure(figsize=(10, 10))
sns.scatterplot(x=scatter_x, y=scatter_y, hue=target_col, data=df, palette='viridis')
plt.title("Debt-to-Income Ratio vs Income (by Default)")
plt.show()

# 5. Correlation heatmap
numeric_df = df.select_dtypes(include=['int64', 'float64'])
plt.figure(figsize=(25,25))
corr_matrix = numeric_df.corr()
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap='coolwarm',
    fmt='.2f',
    linewidths=0.5
)
plt.title("Correlation Heatmap of Numeric Features", fontsize=14)
plt.show()

if target_col in corr_matrix.columns:
    plt.figure(figsize=(20,20))
    top_corr = corr_matrix[target_col].sort_values(ascending=False)
    sns.heatmap(
        corr_matrix.loc[top_corr.index, top_corr.index],
        annot=True,
        cmap='YlGnBu',
        fmt='.2f'
    )
    plt.title("Top Correlated Features with Default")
    plt.show()
else:
    print(f"'{target_col}' column not found in numeric columns.")

# 6. Preprocessing: Drop target & ID, encode, impute, split, scale
X = df.drop(columns=[target_col, 'LoanID'], errors='ignore')
y = df[target_col]
X = pd.get_dummies(X, drop_first=True)
imputer = SimpleImputer(strategy="median")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. Models to train
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Naive Bayes': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
    'KNN': KNeighborsClassifier(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    'MLP_relu': MLPClassifier(activation='relu', max_iter=500),
    'MLP_tanh': MLPClassifier(activation='tanh', max_iter=500),
    'MLP_logistic': MLPClassifier(activation='logistic', max_iter=500)
}

def get_metrics(model, X_train, X_test, y_train, y_test):
    results = {}
    model.fit(X_train, y_train)
    y_tr_pred = model.predict(X_train)
    try:
        y_tr_prob = model.predict_proba(X_train)[:, 1]
    except:
        y_tr_prob = y_tr_pred
    y_te_pred = model.predict(X_test)
    try:
        y_te_prob = model.predict_proba(X_test)[:, 1]
    except:
        y_te_prob = y_te_pred

    for split, X_, y_true, y_pred, y_prob in [('Train', X_train, y_train, y_tr_pred, y_tr_prob),
                                              ('Test', X_test, y_test, y_te_pred, y_te_prob)]:
        acc = accuracy_score(y_true, y_pred)
        sens = recall_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        spec = recall_score(y_true, y_pred, pos_label=0)
        fi = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_prob)
        results[split] = [acc, sens, prec, spec, fi, auc]

        if split == 'Test':
            print(f"{model.__class__.__name__} Confusion Matrix:")
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f"{model.__class__.__name__} Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.show()
    return results

all_results = {}
for name, model in models.items():
    print(f"Training {name}...")
    res = get_metrics(model, X_train_scaled, X_test_scaled, y_train, y_test)
    all_results[name] = res

results_train = []
results_test = []
model_names = []
for k, v in all_results.items():
    model_names.append(k)
    results_train.append(v['Train'])
    results_test.append(v['Test'])

columns = ['Accuracy', 'Sensitivity', 'Precision', 'Specificity', 'F1', 'AUC']
train_df = pd.DataFrame(results_train, columns=columns, index=model_names)
test_df = pd.DataFrame(results_test, columns=columns, index=model_names)

with open('result.txt', 'w') as f:
    f.write("Train Results:\n")
    f.write(train_df.to_string())
    f.write("\n\nTest Results:\n")
    f.write(test_df.to_string())

print("Train Results:")
print(train_df)
print("\nTest Results:")
print(test_df)

# 8. Feature importance for Random Forest
rf_model = models['Random Forest']
importances = rf_model.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[::-1][:15]
plt.figure(figsize=(10, 8))
sns.barplot(x=importances[indices], y=feature_names[indices], palette='viridis')
plt.title("Top 15 Feature Importances")
plt.show()

# 9. Save scaler and model
joblib.dump(scaler, "scaler.joblib")
joblib.dump(rf_model, "loan_default_rf_model.joblib")
print("Scaler and model saved.")
