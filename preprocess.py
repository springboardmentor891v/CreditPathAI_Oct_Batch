# preprocess.py

import pandas as pd
import joblib
import numpy as np
import os

class Preprocessor:
    def __init__(self):
        self.label_encoders = None
        self.scaler = None
        self.num_imputer = None
        self.cat_imputer = None
        self.categorical_cols = None
        self.numeric_cols = None
        self.feature_columns = None

    def load(self):
        # Load all artifacts
        self.label_encoders = joblib.load("preprocessor/label_encoders.pkl")
        self.scaler = joblib.load("preprocessor/scaler.pkl")
        self.num_imputer = joblib.load("preprocessor/num_imputer.pkl")
        self.cat_imputer = joblib.load("preprocessor/cat_imputer.pkl")
        self.categorical_cols = joblib.load("preprocessor/categorical_cols.pkl")
        self.numeric_cols = joblib.load("preprocessor/numeric_cols.pkl")
        
        # Load master column order
        if os.path.exists("preprocessor/feature_columns.pkl"):
            self.feature_columns = joblib.load("preprocessor/feature_columns.pkl")
        else:
            self.feature_columns = self.categorical_cols + self.numeric_cols

    def transform(self, input_dict):
        # 1. Convert dictionary to DataFrame
        df = pd.DataFrame([input_dict])

        # 2. Ensure all columns exist
        for col in self.categorical_cols + self.numeric_cols:
            if col not in df.columns:
                df[col] = np.nan

        # 3. Impute Missing
        df[self.numeric_cols] = self.num_imputer.transform(df[self.numeric_cols])
        df[self.categorical_cols] = self.cat_imputer.transform(df[self.categorical_cols])

        # 4. Label Encode
        for col in self.categorical_cols:
            if col in df.columns:
                le = self.label_encoders[col]
                # Handle unseen labels by mapping to first known class
                df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                df[col] = le.transform(df[col].astype(str))

        # 5. Scale
        df[self.numeric_cols] = self.scaler.transform(df[self.numeric_cols])

        # 6. Reorder columns to match training exactly
        if self.feature_columns:
            df = df[self.feature_columns]

        return df