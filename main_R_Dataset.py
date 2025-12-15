import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix, 
                             classification_report)
import os

print("✅ Script started successfully...")

# ==========================================
# 1. SMART LOAD & MERGE (Corrected for Separator Issues)
# ==========================================
def load_and_merge_data():
    print("🔄 [Step 1] Loading Microsoft R Server Datasets...")

    def read_file_smart(base_name):
        # We define options to try: (File Extension, Separator)
        # We include '|' and ';' just in case
        options = [
            ('.csv', ','), 
            ('.txt', ','), 
            ('.txt', '\t'), # Tab separated
            ('.txt', ';'),  # Semicolon separated
            ('.txt', '|'),  # Pipe separated
            ('.tsv', '\t')
        ]
        
        # Construct the base path
        base_path = f'data/{base_name}'

        for ext, sep in options:
            path = base_path + ext
            if os.path.exists(path):
                try:
                    # Try reading
                    df = pd.read_csv(path, sep=sep, low_memory=False)
                    
                    # CRITICAL CHECK: Did it actually split the columns?
                    if df.shape[1] > 1:
                        print(f"   ✅ Loaded '{path}' using sep='{sep}' (Shape: {df.shape})")
                        return df
                    else:
                        print(f"   ⚠️ File found at '{path}' but sep='{sep}' didn't work (1 column detected). Retrying...")
                        
                except Exception as e:
                    print(f"   ⚠️ Error reading '{path}': {e}")
        
        # If we exit the loop without returning
        print(f"   ❌ Error: Could not parse '{base_name}' correctly.")
        print("      Ensure the file exists in 'data/' and is not empty.")
        exit()

    # Load both files
    loans = read_file_smart('Loan')
    borrowers = read_file_smart('Borrower')
    
    # Merge Logic
    # 1. Check for exact match
    if 'memberId' in loans.columns and 'memberId' in borrowers.columns:
        merge_col = 'memberId'
    # 2. Check for lowercase match
    elif 'memberid' in [c.lower() for c in loans.columns] and 'memberid' in [c.lower() for c in borrowers.columns]:
        print("   ⚠️ exact 'memberId' not found. Normalizing column names to lowercase...")
        loans.columns = [c.lower() for c in loans.columns]
        borrowers.columns = [c.lower() for c in borrowers.columns]
        merge_col = 'memberid'
    else:
        print("   ❌ Merge Error: Could not find 'memberId' column in one of the files.")
        print(f"      Loan Columns: {loans.columns.tolist()[:5]}")
        print(f"      Borrower Columns: {borrowers.columns.tolist()[:5]}")
        exit()

    df = pd.merge(loans, borrowers, on=merge_col, how='inner')
    print(f"   ✅ Merge Complete. Final Shape: {df.shape}")
    return df

# ==========================================
# 2. PREPROCESSING
# ==========================================
def preprocess_dataset(df):
    print("\n🧹 [Step 2] Preprocessing Data...")
    
    # Check if 'loanStatus' or 'loanstatus' exists
    target_col = 'loanStatus' if 'loanStatus' in df.columns else 'loanstatus'
    
    # A. Create Target Variable 'Status' (1 = Default/Charged Off, 0 = Good)
    target_map = ['Charged Off', 'Default']
    df['Status'] = df[target_col].apply(lambda x: 1 if str(x).strip() in target_map else 0)
    
    # B. String Cleaning
    cols_to_clean = {
        'interestRate': '%',
        'interestrate': '%',
        'revolvingUtilizationRate': '%',
        'revolvingutilizationrate': '%',
        'term': ' months'
    }

    for col, symbol in cols_to_clean.items():
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].str.replace(symbol, '').astype(float)
        
    # 'yearsEmployment' extraction
    emp_col = 'yearsEmployment' if 'yearsEmployment' in df.columns else 'yearsemployment'
    if emp_col in df.columns:
        df[emp_col] = df[emp_col].astype(str).str.extract(r'(\d+)').astype(float)

    # C. Drop Irrelevant Columns
    drop_candidates = ['memberId', 'memberid', 'loanId', 'loanid', 'date', 'loanStatus', 'loanstatus']
    existing_drops = [c for c in drop_candidates if c in df.columns]
    df = df.drop(columns=existing_drops)
    
    # D. Missing Values (Imputation)
    print("   Filling missing values...")
    # Numeric -> Median
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    # Exclude Target
    num_cols = [c for c in num_cols if c != 'Status']
    
    if len(num_cols) > 0:
        imputer_num = SimpleImputer(strategy='median')
        df[num_cols] = imputer_num.fit_transform(df[num_cols])
    
    # Categorical -> Mode
    cat_cols = df.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        imputer_cat = SimpleImputer(strategy='most_frequent')
        df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])
    
    # E. Label Encoding
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
        
    return df

# ==========================================
# 3. EDA (Exploratory Data Analysis)
# ==========================================
def run_eda(df):
    print("\n📊 [Step 3] Running Exploratory Analysis...")
    
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Status', data=df)
    plt.title("Loan Status Distribution (0=Good, 1=Default)")
    plt.show()
    
    if len(df.columns) > 1:
        plt.figure(figsize=(10, 8))
        sns.heatmap(df.corr(), cmap='coolwarm', annot=False)
        plt.title("Feature Correlation Matrix")
        plt.show()

# ==========================================
# 4. MODEL TRAINING SUITE
# ==========================================
def train_and_evaluate(model, X_train, y_train, X_test, y_test, model_name):
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"   --> Accuracy: {acc:.4f}")
    print(f"   --> F1 Score: {f1:.4f}")
    
    return [model_name, acc, f1]

# ==========================================
# 5. MAIN EXECUTION FLOW
# ==========================================
if __name__ == "__main__":
    # A. Load
    df = load_and_merge_data()
    
    # B. Preprocess
    df_clean = preprocess_dataset(df)
    
    # C. Analyze
    run_eda(df_clean)
    
    # D. Split Data
    print("\n✂️ [Step 4] Splitting Data...")
    X = df_clean.drop(columns=['Status'])
    y = df_clean['Status']
    
    # Scale Data
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    print(f"   Training Shape: {X_train.shape}")
    print(f"   Testing Shape:  {X_test.shape}")
    
    # E. Train All Models
    print("\n🤖 [Step 5] Training Models...")
    
    results = []
    
    # 1. Logistic Regression
    lr = LogisticRegression(max_iter=1000, class_weight='balanced')
    results.append(train_and_evaluate(lr, X_train, y_train, X_test, y_test, "Logistic Regression"))
    
    # 2. Decision Tree
    dt = DecisionTreeClassifier(random_state=42, class_weight='balanced')
    results.append(train_and_evaluate(dt, X_train, y_train, X_test, y_test, "Decision Tree"))
    
    # 3. Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    results.append(train_and_evaluate(rf, X_train, y_train, X_test, y_test, "Random Forest"))
    
    # 4. Naive Bayes
    nb = GaussianNB()
    results.append(train_and_evaluate(nb, X_train, y_train, X_test, y_test, "Naive Bayes"))
    
    # 5. KNN (Only if dataset is small)
    if len(df_clean) < 50000:
        knn = KNeighborsClassifier(n_neighbors=5)
        results.append(train_and_evaluate(knn, X_train, y_train, X_test, y_test, "KNN"))
    else:
        print("\n   Skipping KNN (Dataset too large for quick training)")

    # F. Final Leaderboard
    print("\n" + "="*30)
    print("🏆 FINAL RESULTS LEADERBOARD")
    print("="*30)
    final_df = pd.DataFrame(results, columns=["Model", "Accuracy", "F1 Score"])
    print(final_df.sort_values(by="F1 Score", ascending=False).to_string(index=False))

    print("\n✅ Script Finished Successfully.")