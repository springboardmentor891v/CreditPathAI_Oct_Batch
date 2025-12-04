# models.py

import joblib
import numpy as np
import os

class ModelLoader:
    def __init__(self):
        self.models = {}
        self.loaded = False

    def load_all(self):
        def safe_load(name, filename):
            try:
                if os.path.exists(filename):
                    self.models[name] = joblib.load(filename)
                else:
                    print(f"Warning: {filename} not found.")
            except Exception as e:
                print(f"Warning: could not load {name}: {e}")

        safe_load("Logistic Regression",      "models/logistic_regression.pkl")
        safe_load("Decision Tree",           "models/decision_tree.pkl")
        safe_load("Random Forest",           "models/random_forest.pkl")
        safe_load("Naive Bayes",            "models/naive_bayes.pkl")
        safe_load("KNN",                    "models/knn.pkl")
        safe_load("Balanced Random Forest",  "models/balanced_rf.pkl")
        safe_load("XGBoost",                "models/xgboost.pkl")
        safe_load("LightGBM",               "models/lightgbm.pkl")
        safe_load("CatBoost",               "models/catboost.pkl")
        safe_load("Voting Classifier",      "models/voting_classifier.pkl")
        safe_load("Stacking Classifier",    "models/stacking_classifier.pkl")

        self.loaded = True

    def predict(self, model_name: str, X):
        if not self.loaded:
            raise RuntimeError("Models not loaded. Call .load_all() first.")

        model = self.models[model_name]

        # Try to get probability (Soft Voting)
        prob = 0.0
        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)
                prob = float(proba[0, 1])
            elif hasattr(model, "decision_function"):
                score = model.decision_function(X)
                prob = float(1.0 / (1.0 + np.exp(-score[0])))
            else:
                prob = float(model.predict(X)[0])
        except:
            # Fallback
            prob = float(model.predict(X)[0])

        pred = int(prob >= 0.5)
        return pred, prob