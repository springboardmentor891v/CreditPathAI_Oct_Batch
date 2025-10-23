import os
import subprocess
import pandas as pd
import numpy as np

# --- Install dependencies ---
subprocess.run(["conda", "install", "numpy=1.26.4", "-y"], check=False)
subprocess.run(["pip", "install", "kagglehub"], check=False)
subprocess.run(["pip", "install", "kaggle"], check=False)

# --- Download dataset using kagglehub ---
import kagglehub

path = kagglehub.dataset_download("nikhil1e9/loan-default")
print("Path to dataset files:", path)

# --- Locate CSV and load it into pandas ---
path = os.path.expanduser("~/.cache/kagglehub/datasets/nikhil1e9/loan-default/versions/2")
csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
print("CSV files found:", csv_files)

df = pd.read_csv(os.path.join(path, csv_files[0]))

# --- Explore dataset ---
print("\n--- HEAD ---")
print(df.head())

print("\n--- INFO ---")
print(df.info())

print("\n--- DESCRIPTION ---")
print(df.describe())

print("\n--- TAIL ---")
print(df.tail())

print("\n--- OVERALL NULL VALUES ---")
print(df.isnull())

print("\n--- SUM OF NULL VALUES ---")
print(df.isnull().sum())
