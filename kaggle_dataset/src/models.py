# src/models.py


import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
    xgb_available = True
except ImportError:
    xgb_available = False



# -------------------------------
#  Get Models
# -------------------------------
def get_models():
    """Return a dictionary of commonly used classification models"""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=200),
        "XGBoost": XGBClassifier(eval_metric='logloss'),  
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(probability=True),
    }
    
    if xgb_available:
        models["XGBoost"] = XGBClassifier(eval_metric='logloss')

    return models

# -------------------------------
#  Evaluate Models
# -------------------------------
def evaluate_models(X, y, models=None, cv=5):
    """Cross-validate multiple models and return F1-score results"""
    if models is None:
        models = get_models()
    
    results = []
    for name, model in models.items():
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
            results.append({"Model": name, "F1-Score (CV)": np.mean(scores)})
        except Exception as e:
            results.append({"Model": name, "F1-Score (CV)": np.nan})
            print(f"Could not evaluate {name}: {e}")
    
    results_df = pd.DataFrame(results).sort_values(by="F1-Score (CV)", ascending=False).reset_index(drop=True)
    return results_df

# -------------------------------
#  Train Final Model
# -------------------------------
def train_final_model(X_train, y_train, model_name="Random Forest"):
    """Train a single model on the provided data"""
    models_dict = get_models()
    if model_name not in models_dict:
        raise ValueError(f"Model {model_name} not found. Choose from {list(models_dict.keys())}")
    
    model = models_dict[model_name]
    model.fit(X_train, y_train)
    return model

# -------------------------------
#  Example usage (synthetic dataset)
# -------------------------------
'''
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    import pandas as pd

    X, y = make_classification(n_samples=100, n_features=5, n_informative=3,
                               n_redundant=0, n_classes=2, random_state=42)
    X = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(5)])
    y = pd.Series(y, name="target")

    print("=== Cross-Validation ===")
    results = evaluate_models(X, y, cv=3)
    print(results)

    print("=== Train Final Model ===")
    model = train_final_model(X, y, model_name="Random Forest")
    print(model)

'''