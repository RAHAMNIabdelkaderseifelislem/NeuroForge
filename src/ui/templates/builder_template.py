import streamlit as st
from ..core.data_manager import DataManager
from ..core.model_builder import ModelBuilder
from ..core.model_trainer import ModelTrainer
from ..core.visualization import Visualization
from ..config.data_config import DataConfig
from ..config.model_config import ModelConfig, LayerConfig

def builder_template():
    st.title("Neural Network Builder")

    # Data loading and preprocessing
    data_path = st.file_uploader("Upload dataset", type=["csv", "excel", "parquet", "json"])
    if data_path is not None:
        data_config = DataConfig(data_path=data_path)
        data_manager = DataManager(data_config)
        data = data_manager.load_data()
        data = data_manager.preprocess_data()
        st.write("Data loaded and preprocessed successfully.")

        # Model building
        input_shape = data_manager.get_feature_dimensions()['input_dim']
        output_shape = data_manager.get_feature_dimensions()['output_dim']
        layers = []

        st.subheader("Add Layers")
        layer_type = st.selectbox("Layer Type", ["Linear", "ReLU", "Sigmoid", "Tanh", "Dropout"])
        if layer_type == "Linear":
            output_dim = st.number_input("Output Dimension", min_value=1, value=1)
            layers.append(LayerConfig(layer_type="linear", layer_name=f"linear_{len(layers)}", layer_params={"output_dim": output_dim}))
        elif layer_type == "Dropout":
            p = st.slider("Dropout Probability", min_value=0.0, max_value=1.0, value=0.5)
            layers.append(LayerConfig(layer_type="dropout", layer_name=f"dropout_{len(layers)}", layer_params={"p": p}))
        else:
            layers.append(LayerConfig(layer_type=layer_type.lower(), layer_name=f"{layer_type.lower()}_{len(layers)}"))

        if st.button("Add Layer"):
            st.write(f"Added {layer_type} layer.")

        if layers:
            model_config = ModelConfig(input_shape=[input_shape], output_shape=[output_shape], layers=layers)
            model_builder = ModelBuilder(model_config.input_shape, model_config.output_shape, model_config.layers)
            model = model_builder.build_model()
            st.write("Model built successfully.")

            # Model training
            train_loader, val_loader = data_manager.prepare_data_loaders()
            model_trainer = ModelTrainer(model, train_loader, val_loader, data_config)
            if st.button("Train Model"):
                model_trainer.train()
                st.write("Model trained successfully.")

            # Visualization
            visualization = Visualization(data)
            if st.button("Visualize Data"):
                visualization.plot_data_distribution(data.columns)
            if st.button("Visualize Training Results"):
                train_losses = model_trainer.train_losses
                val_losses = model_trainer.val_losses
                visualization.plot_training_results(train_losses, val_losses)
