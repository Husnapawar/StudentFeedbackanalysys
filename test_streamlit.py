import streamlit as st

st.title("Test Streamlit App")
st.write("This is a test app to check if Streamlit is working correctly.")

if st.button("Click me"):
    st.success("Button clicked!")
