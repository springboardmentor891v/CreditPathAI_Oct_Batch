import streamlit as st

st.set_page_config(page_title="Loan Default Risk Prediction", page_icon=" ", layout="wide")

st.title("Loan Default Risk Prediction using AI")
st.markdown("Enter applicant's details to predict their loan default risk")

st.sidebar.header("Applicant's Details")

annual_income=st.sidebar.slider("Annual Income (Rs.)", 1000000, 2000000, 3000000)
applicant_age=st.sidebar.selectbox("Age (Years)", list(range(18,55)))

#buttons
st.sidebar.markdown("### Actions")
preprocess_btn = st.sidebar.button("🔄 Preprocess Data")
predict_btn = st.sidebar.button("🔍 Predict Risk")