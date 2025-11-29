import streamlit as st

st.write("Hello Streamlit from VS Code!")

st.write("Welcome to Infosys Internship – Streamlit UI Demo")

st.title("Streamlit User Interface Demo")
st.header("Basic UI Components")

st.markdown("""
This application demonstrates **Streamlit UI elements**
based on the DataCamp Streamlit tutorial.
""")

st.subheader("Text Elements")
st.caption("This is a caption text example")

st.subheader("Code Display")
st.code("""
x = 10
y = 20
print(x + y)
""")

st.subheader("Mathematical Expression")
st.latex(r"x^2 + y^2 = z^2")

st.success("UI components rendered successfully !")
