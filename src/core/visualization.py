import plotly.graph_objects as go
import pandas as pd

class Visualization:
    def __init__(self, data):
        self.data = data

    def plot_data_distribution(self, columns):
        fig = go.Figure()

        for column in columns:
            fig.add_trace(go.Histogram(x=self.data[column], name=column))

        fig.update_layout(barmode='overlay')
        fig.update_traces(opacity=0.75)
        fig.show()

    def plot_training_results(self, train_losses, val_losses):
        fig = go.Figure()

        fig.add_trace(go.Scatter(y=train_losses, mode='lines', name='Train Loss'))
        fig.add_trace(go.Scatter(y=val_losses, mode='lines', name='Val Loss'))

        fig.update_layout(title='Training Results', xaxis_title='Epoch', yaxis_title='Loss')
        fig.show()
