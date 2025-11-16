"""
eda.py
------
Reusable & dataset-agnostic EDA module.

Covers:
1. Dataset Overview
2. Missing Values Analysis
3. Outlier Detection (IQR)
4. Univariate Analysis (histograms, skewness)
5. Bivariate Analysis (correlation, numeric vs numeric)
6. Categorical Analysis
7. Target Variable Analysis (auto-detected if binary)
8. Pairwise Relationships
9. PCA Visualization

Usage:
------
from src.eda import run_eda

run_eda(df, target_col="repay_fail")
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ===========================================================================================
# MAIN FUNCTION
# ===========================================================================================
def run_eda(df: pd.DataFrame, target_col=None, max_plots=50):
    print("\n" + "="*90)
    print("                         🔍 EXPLORATORY DATA ANALYSIS REPORT")
    print("="*90)

    # ==========================================================
    # 1. DATASET OVERVIEW
    # ==========================================================
    print("\n DATASET SHAPE:", df.shape)
    print("\n COLUMN LIST:\n", df.columns.tolist())

    print("\n DATA TYPES:")
    print(df.dtypes)

    print("\n SUMMARY STATISTICS:")
    print(df.describe(include="all").transpose().head())

    print("\n TOTAL MISSING VALUES:", df.isnull().sum().sum())
    print("📌 DUPLICATE ROWS:", df.duplicated().sum())

    # ==========================================================
    # 2. MISSING VALUES ANALYSIS
    # ==========================================================
    missing_summary = df.isnull().sum()
    missing_summary = missing_summary[missing_summary > 0].sort_values(ascending=False)
    print("\n Missing Values Summary:\n", missing_summary)

    if missing_summary.shape[0] > 0:
        plt.figure(figsize=(12, 6))
        sns.heatmap(df.isnull(), cbar=False)
        plt.title("Missing Values Heatmap")
        plt.show()

    # ==========================================================
    # 3. OUTLIER DETECTION
    # ==========================================================
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        Q1 = df[numeric_cols].quantile(0.25)
        Q3 = df[numeric_cols].quantile(0.75)
        IQR = Q3 - Q1

        outliers = ((df[numeric_cols] < (Q1 - 1.5 * IQR)) |
                    (df[numeric_cols] > (Q3 + 1.5 * IQR)))

        print("\n Outliers per column:")
        print(outliers.sum().sort_values(ascending=False))

    # ==========================================================
    # 4. UNIVARIATE ANALYSIS
    # ==========================================================
    print("\n" + "="*40)
    print(" UNIVARIATE ANALYSIS - NUMERIC")
    print("="*40)

    for col in numeric_cols:
        data = df[col].dropna()
        if len(data) == 0:
            continue

        skew_val = data.skew()
        plt.figure(figsize=(6, 4))
        sns.histplot(data, bins=30, kde=True)
        plt.title(f"{col} | Skew = {skew_val:.2f}")
        plt.show()

    # ==========================================================
    # Boxplots for outliers
    # ==========================================================
    for col in numeric_cols[:10]:  # limit
        plt.figure(figsize=(6, 3))
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot of {col}')
        plt.show()

    # ==========================================================
    # 5. BIVARIATE ANALYSIS (CORRELATION)
    # ==========================================================
    print("\n CORRELATION MATRIX")

    corr_matrix = df[numeric_cols].corr()

    plt.figure(figsize=(12, 9))
    sns.heatmap(corr_matrix, annot=False, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

    # ==========================================================
    # Scatter plots with target (if numeric)
    # ==========================================================
    if target_col and target_col in df.columns:
        for col in numeric_cols:
            if col == target_col:
                continue
            
            plt.figure(figsize=(6, 4))
            sns.scatterplot(x=df[col], y=df[target_col])
            plt.title(f"{col} vs {target_col}")
            plt.show()

    # ==========================================================
    # 6. CATEGORICAL ANALYSIS
    # ==========================================================
    categorical_cols = df.select_dtypes(include="object").columns.tolist()

    print("\n UNIVARIATE ANALYSIS - CATEGORICAL")

    for col in categorical_cols[:20]:  # safe limit
        print(f"\nValue counts for {col}:")
        print(df[col].value_counts())

        plt.figure(figsize=(6, 4))
        sns.countplot(y=df[col], order=df[col].value_counts().index)
        plt.title(f"Countplot of {col}")
        plt.show()

        if target_col and target_col in df.columns:
            plt.figure(figsize=(6, 4))
            sns.barplot(x=df[col], y=df[target_col])
            plt.title(f"{col} vs {target_col}")
            plt.xticks(rotation=45)
            plt.show()

    # ==========================================================
    # 7. TARGET VARIABLE ANALYSIS
    # ==========================================================
    if target_col and target_col in df.columns:
        print("\n🎯 TARGET VARIABLE ANALYSIS")

        plt.figure(figsize=(6, 4))
        sns.countplot(x=df[target_col])
        plt.title(f"Target Distribution: {target_col}")
        plt.show()

        print(df[target_col].value_counts())
        print(df[target_col].value_counts(normalize=True) * 100)

    # ==========================================================
    # 8. PAIRWISE RELATIONSHIPS
    # ==========================================================
    if len(numeric_cols) >= 3:
        sample_cols = numeric_cols[:5]  # reduce overload
        sns.pairplot(df[sample_cols + ([target_col] if target_col else [])],
                     hue=target_col,
                     diag_kind="kde",
                     height=2.2)
        plt.show()

    # ==========================================================
    # 9. PCA
    # ==========================================================
    print("\n PCA Projection")

    if len(numeric_cols) > 2:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df[numeric_cols].fillna(0))

        pca = PCA(n_components=2)
        pcs = pca.fit_transform(scaled)

        plt.figure(figsize=(7, 5))
        sns.scatterplot(x=pcs[:, 0], y=pcs[:, 1],
                        hue=df[target_col] if target_col else None,
                        alpha=0.6)
        plt.title("PCA (2 Components)")
        plt.show()

    print("\n EDA COMPLETED SUCCESSFULLY\n")


# ===========================================================================================
# END
# ===========================================================================================
