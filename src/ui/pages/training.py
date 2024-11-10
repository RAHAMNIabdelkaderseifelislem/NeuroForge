import streamlit as st

def display():
    st.title("Training Dashboard")
    st.write("Training in progress... Metrics will be displayed here.")
    # Placeholder for training metrics
    st.line_chart({"Accuracy": [0.1, 0.2, 0.4, 0.6]})
