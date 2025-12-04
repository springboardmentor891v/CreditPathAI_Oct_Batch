import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Loan Default Dashboard", layout="wide")

models = joblib.load("models.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")
columns = joblib.load("columns.pkl")
numeric_cols = joblib.load("numeric_cols.pkl")
metrics = joblib.load("metrics.pkl")
conf_mats = joblib.load("conf_mats.pkl")
roc_data = joblib.load("roc_data.pkl")
df_template = pd.read_csv("Loan_Default.csv")

st.title("Loan Default Prediction and Evaluation Dashboard")

st.subheader("Enter Applicant Details")

user_input = {}
col1, col2 = st.columns(2)
for idx, col in enumerate(df_template.columns):
    if col == "Loan_Default":
        continue
    default_val = str(df_template[col].median()) if pd.api.types.is_numeric_dtype(df_template[col]) else ""
    if pd.api.types.is_numeric_dtype(df_template[col]):
        with (col1 if idx % 2 == 0 else col2):
            user_input[col] = st.text_input(col, value=default_val)
    else:
        options = [""] + df_template[col].dropna().unique().tolist()
        with (col1 if idx % 2 == 0 else col2):
            user_input[col] = st.selectbox(col, options=options, index=0)

st.sidebar.header("Model Selection")
selected_models = st.sidebar.multiselect("Choose Models", list(models.keys()))
run_btn = st.sidebar.button("Run")

def preprocess(raw):
    df_in = pd.DataFrame([raw])
    for c in df_in.columns:
        if c in numeric_cols:
            df_in[c] = float(df_in[c])
        else:
            df_in[c] = str(df_in[c])
    cat_cols = df_in.select_dtypes(include=["object"]).columns.tolist()
    if cat_cols:
        enc_df = pd.DataFrame(
            encoder.transform(df_in[cat_cols]),
            columns=encoder.get_feature_names_out(cat_cols)
        )
        df_in = df_in.drop(columns=cat_cols)
        df_in = pd.concat([df_in.reset_index(drop=True), enc_df.reset_index(drop=True)], axis=1)
    for col in columns:
        if col not in df_in.columns:
            df_in[col] = 0
    df_in = df_in[columns]
    df_in[numeric_cols] = scaler.transform(df_in[numeric_cols])
    return df_in

if run_btn:
    if not selected_models:
        st.error("Select at least one model.")
    else:
        processed = preprocess(user_input)

        st.sidebar.subheader("Entered Data")
        st.sidebar.dataframe(pd.DataFrame([user_input]), use_container_width=True)

        st.subheader("Processed Data")
        st.dataframe(processed.T, use_container_width=True)

        auc_scores = {}
        predictions = {}

        st.subheader("Model Outputs")

        for name in selected_models:
            st.markdown(f"### {name}")

            model = models[name]
            pred = model.predict(processed)[0]
            proba = model.predict_proba(processed)[0][1]
            result = "DEFAULT" if pred == 1 else "NON-DEFAULT"
            predictions[name] = result

            st.write("Prediction:", result)
            st.write("Probability of Default:", round(proba, 4))

            m = metrics[name]
            df_metrics = pd.DataFrame([[
                m["Accuracy"], m["Sensitivity"], m["Specificity"],
                m["Precision"], m["F1"], m["AUC"]
            ]], columns=["Accuracy", "Sensitivity", "Specificity",
                         "Precision", "F1", "AUC"])
            st.write("Performance Metrics")
            st.dataframe(df_metrics, use_container_width=True)

            cm = conf_mats[name]
            fig_cm, ax_cm = plt.subplots(figsize=(1.6, 1.6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
            st.write("Confusion Matrix")
            st.pyplot(fig_cm)

        st.subheader("Combined ROC Curve")
        fig_roc, ax_roc = plt.subplots(figsize=(2.5, 2.2))

        for name in selected_models:
            fpr = roc_data[name]["fpr"]
            tpr = roc_data[name]["tpr"]
            auc_val = roc_data[name]["auc"]
            auc_scores[name] = auc_val
            ax_roc.plot(fpr, tpr, linewidth=1, label=f"{name} ({auc_val:.3f})")

        ax_roc.plot([0, 1], [0, 1], "k--", linewidth=0.6)
        ax_roc.set_xlabel("FPR", fontsize=8)
        ax_roc.set_ylabel("TPR", fontsize=8)
        ax_roc.legend(fontsize=6)
        st.pyplot(fig_roc)

        best_model = max(auc_scores, key=auc_scores.get)
        best_pred = predictions[best_model]

        st.subheader(f"Best Model: {best_model} | Prediction: {best_pred}")