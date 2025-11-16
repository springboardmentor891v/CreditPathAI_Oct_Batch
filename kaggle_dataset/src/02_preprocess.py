import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


class DataPreprocessor:
    """
    A fully reusable, professional preprocessing engine.
    Apply to ANY dataset by supplying configurations.
    """

    def __init__(self):
        self.scaler = None
        self.encoder = None
        self.column_transformer = None

    # ---------------------------------------------------------
    # 1. HANDLE MISSING VALUES
    # ---------------------------------------------------------
    def handle_missing_values(self, df):
        """ Automatically handles missing values for numeric & categorical columns. """

        df = df.copy()

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        # Numeric imputation → Median
        if numeric_cols:
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

        # Categorical imputation → Mode
        if categorical_cols:
            for col in categorical_cols:
                df[col] = df[col].fillna(df[col].mode()[0])

        return df

    # ---------------------------------------------------------
    # 2. TYPE CONVERSION
    # ---------------------------------------------------------
    def convert_types(self, df, percent_columns=None, datetime_columns=None):
        """
        Converts selected columns to numeric (percent) or datetime.
        Provide column names as lists.
        """

        df = df.copy()

        # Convert percent → float
        if percent_columns:
            for col in percent_columns:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.replace('%', '', regex=False)
                    df[col] = pd.to_numeric(df[col], errors='coerce')

        # Convert to datetime
        if datetime_columns:
            for col in datetime_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')

        return df

    # ---------------------------------------------------------
    # 3. ENCODE CATEGORICAL VARIABLES
    # ---------------------------------------------------------
    def encode_categorical(self, df):
        """ Performs One-Hot Encoding on all categorical variables. """

        df = df.copy()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        if not categorical_cols:
            return df

        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        return df

    # ---------------------------------------------------------
    # 4. OUTLIER HANDLING (OPTIONAL)
    # ---------------------------------------------------------
    def handle_outliers(self, df, method="iqr"):
        """
        Removes or caps outliers using IQR rule.
        method = "iqr", "cap", or None
        """

        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if method is None:
            return df

        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            if method == "iqr":
                df = df[(df[col] >= lower) & (df[col] <= upper)]

            elif method == "cap":
                df[col] = np.where(df[col] < lower, lower, df[col])
                df[col] = np.where(df[col] > upper, upper, df[col])

        return df

    # ---------------------------------------------------------
    # 5. FEATURE ENGINEERING (GENERIC)
    # ---------------------------------------------------------
    def feature_engineering(self, df):
        """ Generic example features. Extend based on dataset. """

        df = df.copy()

        # Example: log transform of skewed variables
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].skew() > 1:
                df[f"{col}_log"] = np.log1p(df[col])

        return df

    # ---------------------------------------------------------
    # 6. SCALING
    # ---------------------------------------------------------
    def scale_features(self, df, method="standard"):
        """ Standard scaling or Min-Max scaling. """

        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_cols:
            return df

        if method == "standard":
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()

        df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])

        return df

    # ---------------------------------------------------------
    # 7. FULL PIPELINE
    # ---------------------------------------------------------
    def preprocess(
        self,
        df,
        percent_columns=None,
        datetime_columns=None,
        outlier_method="iqr",
        scale_method="standard"
    ):
        """ Runs the entire preprocessing sequence on any dataset. """

        df = df.copy()

        df = self.handle_missing_values(df)
        df = self.convert_types(df, percent_columns, datetime_columns)
        df = self.encode_categorical(df)
        df = self.handle_outliers(df, method=outlier_method)
        df = self.feature_engineering(df)
        df = self.scale_features(df, method=scale_method)

        return df

