import pandas as pd
from multiprocessing import Pool
import torch
import torch.nn as nn
import torch.optim as optim
import plotly.graph_objects as go

def preprocess_data(file):
    df = pd.read_csv(file)

    with Pool() as pool:
        stats = pool.map(calculate_statistics, [df[col] for col in df.columns if df[col].dtype != 'object'])

    stats_dict = {col: stat for col, stat in zip(df.columns, stats)}
    return df, stats_dict

def calculate_statistics(column):
    return {
        "mean": column.mean(),
        "std_dev": column.std(),
        "min": column.min(),
        "max": column.max(),
    }

def get_layers():
    return ["Linear", "ReLU", "Conv2d", "MaxPool2d", "Dropout", "BatchNorm2d"]

def train_model(layers):
    model = nn.Sequential()
    for layer in layers:
        if layer == "Linear":
            model.add_module("Linear", nn.Linear(10, 10))
        elif layer == "ReLU":
            model.add_module("ReLU", nn.ReLU())
        elif layer == "Conv2d":
            model.add_module("Conv2d", nn.Conv2d(1, 32, kernel_size=3))
        elif layer == "MaxPool2d":
            model.add_module("MaxPool2d", nn.MaxPool2d(kernel_size=2))
        elif layer == "Dropout":
            model.add_module("Dropout", nn.Dropout(0.5))
        elif layer == "BatchNorm2d":
            model.add_module("BatchNorm2d", nn.BatchNorm2d(32))

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    inputs = torch.randn(64, 1, 28, 28)
    targets = torch.randn(64, 10)
    for epoch in range(5):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

    return {"final_loss": loss.item()}

def visualize_training():
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=[0.5, 0.4, 0.3, 0.2, 0.1], mode='lines', name='Loss'))
    fig.update_layout(title="Training Metrics", xaxis_title="Epoch", yaxis_title="Loss")
    return fig
