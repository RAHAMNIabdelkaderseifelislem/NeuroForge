import torch.nn as nn

class ModelBuilder:
    def __init__(self, input_size):
        self.input_size = input_size
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def build_model(self):
        model = nn.Sequential()
        for i, layer in enumerate(self.layers):
            model.add_module(f'layer_{i}', layer)
        return model
