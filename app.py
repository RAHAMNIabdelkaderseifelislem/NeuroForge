import streamlit as st
from src.ui.pages import home, model_building, training#, data_preprocessing, export


st.sidebar.title("NeuroForge")
page = st.sidebar.radio("Navigation", ("Home", "Data Preprocessing", "Model Building", "Training", "Export"))

if page == "Home":
    home.display()
elif page == "Model Building":
    model_building.display()
elif page == "Training":
    training.display()
"""
elif page == "Data Preprocessing":
    data_preprocessing.display()
elif page == "Export":
    export.display()
"""