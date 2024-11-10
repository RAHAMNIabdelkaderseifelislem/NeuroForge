import torch.nn as nn

class ModelBuilder:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def build_model(self):
        return nn.Sequential(*self.layers)
