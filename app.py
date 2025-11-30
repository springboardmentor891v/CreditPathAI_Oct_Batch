
import streamlit as st
import pandas as pd
import numpy as np
import joblib


st.title("Loan Default Risk Prediction")

# Load trained models
models = joblib.load("models.pkl")

# Load scaler
scaler = joblib.load("scaler.pkl")

# Load encoder
encoder = joblib.load("encoder.pkl")

# Load columns and numeric columns
columns = joblib.load("columns.pkl")           # final encoded column names
numeric_cols = joblib.load("numeric_cols.pkl") # numeric columns for scaling


# MODEL SELECTION
model_name = st.selectbox("Select Model", list(models.keys()))
model = models[model_name]

st.write(f"Selected Model: **{model_name}**")


# USER INPUT

st.subheader("Enter Applicant Details:")

# Load original dataset to get column names and types
df_template = pd.read_csv("data/Loan_default.csv")

user_input = {}
for col in df_template.columns:
    if col == "Status":
        continue

    # skip target
    if pd.api.types.is_integer_dtype(df_template[col]):
        # Integer columns (ID, year, etc.)
        val = st.number_input(
            f"{col}",
            value=int(df_template[col].median()) if not df_template[col].isnull().all() else 0,
            step=1,
            format="%d"
        )
    elif pd.api.types.is_float_dtype(df_template[col]):
        # Float columns (loan amount, interest rate, etc.)
        val = st.number_input(
            f"{col}",
            value=float(df_template[col].median()) if not df_template[col].isnull().all() else 0.0
        )
    else:
        # Categorical columns
        options = df_template[col].dropna().unique().tolist()
        default_index = 0 if options else None
        val = st.selectbox(f"{col}", options=options, index=default_index)

    user_input[col] = val


# Convert input to DataFrame
input_df = pd.DataFrame([user_input])


#  PREPROCESS INPUT

# 1. OneHotEncode categorical features
cat_cols = input_df.select_dtypes(include=['object', 'category']).columns.tolist()

if len(cat_cols) > 0:
    encoded_input = pd.DataFrame(
        encoder.transform(input_df[cat_cols]),
        columns=encoder.get_feature_names_out(cat_cols)
    )
    input_df = input_df.drop(columns=cat_cols)
    input_df = pd.concat([input_df, encoded_input], axis=1)

# 2. Add missing columns (if any)
for col in columns:
    if col not in input_df.columns:
        input_df[col] = 0

# 3. Ensure correct column order
input_df = input_df[columns]

# 4. Scale numeric columns
input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

#  MAKE PREDICTION

if st.button("Predict"):
    pred = model.predict(input_df)[0]
    pred_proba = model.predict_proba(input_df)[0][1]

    result = "Default" if pred == 1 else "Non-Default"
    st.success(f"Prediction: **{result}**")
    st.info(f"Probability of Default: **{pred_proba:.2f}**")