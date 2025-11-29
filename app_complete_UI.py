import streamlit as st
import pandas as pd

st.set_page_config(page_title="Streamlit UI Complete", layout="wide")

# Title
st.title("Streamlit UI – Complete Demo")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Choose Section", ["Text", "Inputs", "Data", "Charts"])

# TEXT UI
if page == "Text":
    st.header("Text Components")
    st.write("This is st.write()")
    st.subheader("Subheader Example")
    st.markdown("**Bold Text**, *Italic Text*")
    st.caption("This is a caption")
    st.code("print('Hello Streamlit')")
    st.latex(r"a^2 + b^2 = c^2")

# INPUT UI
elif page == "Inputs":
    st.header("Input Components")

    name = st.text_input("Enter your name")
    age = st.slider("Select your age", 18, 60)
    agree = st.checkbox("I accept the terms")
    option = st.selectbox("Choose domain", ["AI", "Data", "Cloud", "UI"])
    submit = st.button("Submit")

    if submit:
        st.success(f"Hello {name}, Age: {age}, Domain: {option}")

# DATA UI
elif page == "Data":
    st.header("Data Display")

    df = pd.DataFrame({
        "Student": ["A", "B", "C"],
        "Marks": [85, 90, 88]
    })
    st.dataframe(df)
    st.table(df)

# CHART UI
elif page == "Charts":
    st.header("Charts")

    chart_data = pd.DataFrame({
        "Values": [5, 15, 10, 20]
    })
    st.line_chart(chart_data)
    st.bar_chart(chart_data)

st.success("All Streamlit UI components implemented ")
