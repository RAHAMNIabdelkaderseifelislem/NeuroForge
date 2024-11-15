import streamlit as st
import plotly.express as px
from src.backend import preprocess_data, get_layers, train_model, visualize_training

# Page Title
st.title("NeuroForge 🚀")
st.sidebar.header("Menu")

# Sidebar Options
menu = st.sidebar.selectbox("Choose an option", ["Home", "Data Preprocessing", "Model Training", "Real-time Metrics"])

if menu == "Home":
    st.write("Welcome to **NeuroForge**! 🎨 Build, train, and deploy neural networks with ease.")

elif menu == "Data Preprocessing":
    st.write("📊 **Data Preprocessing**")
    file = st.file_uploader("Upload a dataset", type=["csv"])
    if file:
        df, stats = preprocess_data(file)
        st.write("Dataset Preview:")
        st.dataframe(df)
        st.write("Data Statistics:")
        st.json(stats)
        st.write("Visualization:")
        st.plotly_chart(px.scatter(df, x=df.columns[0], y=df.columns[1]))

elif menu == "Model Training":
    st.write("⚡ **Model Training**")
    st.write("Select Layers:")
    layers = st.multiselect("Choose layers", get_layers())
    if st.button("Start Training"):
        metrics = train_model(layers)
        st.write("Training Complete!")
        st.json(metrics)

elif menu == "Real-time Metrics":
    st.write("🔍 **Real-time Metrics**")
    fig = visualize_training()
    st.plotly_chart(fig)
