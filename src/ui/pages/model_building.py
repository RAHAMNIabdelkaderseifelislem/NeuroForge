import streamlit as st
from src.core.model_builder import ModelBuilder

def display():
    st.title("Model Building")
    builder = ModelBuilder()

    layer = st.selectbox("Add Layer", ["Conv2D", "Dense", "ReLU", "Dropout"])
    if st.button("Add Layer"):
        builder.add_layer(layer)
    st.write("Current Model Architecture:", builder.build_model())
