import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from src.core.data_processor import PandasDataProcessor
from src.core.model_builder import ModelBuilder
from src.core.model_trainer import ModelTrainer

class NeuroForgeApp:
    def __init__(self):
        self.data_processor = PandasDataProcessor()
        self.model_builder = ModelBuilder()
    
    def run(self):
        st.title("NeuroForge Beta")
        
        # Data Upload Section
        st.header("Data Processing")
        uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])
        
        if uploaded_file:
            data = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('csv') \
                  else pd.read_excel(uploaded_file)
            processed_data = self.data_processor.process(data)
            
            # Basic Data Visualization
            st.subheader("Data Preview")
            st.dataframe(processed_data.head())
            
            if st.button("Generate Basic Plots"):
                for column in processed_data.select_dtypes(include=[np.number]).columns:
                    fig = px.histogram(processed_data, x=column)
                    st.plotly_chart(fig)
        
        # Model Building Section
        st.header("Neural Network Builder")
        layer_type = st.selectbox("Add Layer", ["Linear", "Conv2d"])
        
        if layer_type == "Linear":
            in_features = st.number_input("Input Features", min_value=1)
            out_features = st.number_input("Output Features", min_value=1)
            if st.button("Add Linear Layer"):
                self.model_builder.add_layer("Linear", {
                    "in_features": in_features,
                    "out_features": out_features
                })
        
        # Training Configuration
        st.header("Training Configuration")
        epochs = st.slider("Number of Epochs", 1, 100, 10)
        
        if st.button("Train Model"):
            # Add training logic here
            st.info("Training functionality will be available in the next release")

if __name__ == "__main__":
    app = NeuroForgeApp()
    app.run()