import torch
import torch.nn as nn

class LayerFactory:
    @staticmethod
    def create_layer(layer_type, config):
        if layer_type == "Linear":
            return nn.Linear(config["in_features"], config["out_features"])
        elif layer_type == "Conv2d":
            return nn.Conv2d(config["in_channels"], config["out_channels"], 
                           kernel_size=config["kernel_size"])
        # TODO: Add more layer types
        return None

class ModelBuilder:
    def __init__(self):
        self.layers = []
        self.factory = LayerFactory()
    
    def add_layer(self, layer_type, config):
        layer = self.factory.create_layer(layer_type, config)
        if layer:
            self.layers.append(layer)
    
    def build(self):
        return nn.Sequential(*self.layers)
