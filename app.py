import streamlit as st

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Model Demo"])

    if page == "Home":
        st.title("Welcome to CreditPathAI Demo")
        st.write("This is a basic Streamlit application demonstrating a potential layout.")
        st.write("Use the sidebar to navigate to the Model Demo page.")

        feature1 = st.sidebar.slider("Feature 1", 0, 100, 50)
        feature2 = st.sidebar.selectbox("Feature 2", ["Option A", "Option B", "Option C"])
        
        if st.button("Run Model"):
          
            st.write(f"Running model with Feature 1: {feature1}, Feature 2: {feature2}")
            st.success("Model output would appear here!")
            st.write("Predicted Risk: Low (placeholder)")

if __name__ == "__main__":
    main()
