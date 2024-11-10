import torch.nn as nn

class ModelBuilder:
    def __init__(self, input_shape, output_shape, layers):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.layers = layers

    def build_model(self):
        model_layers = []
        input_dim = self.input_shape[0]

        for layer_config in self.layers:
            layer_type = layer_config['layer_type']
            layer_params = layer_config['layer_params']

            if layer_type == 'linear':
                layer = nn.Linear(input_dim, layer_params['output_dim'])
            elif layer_type == 'relu':
                layer = nn.ReLU()
            elif layer_type == 'sigmoid':
                layer = nn.Sigmoid()
            elif layer_type == 'tanh':
                layer = nn.Tanh()
            elif layer_type == 'dropout':
                layer = nn.Dropout(layer_params['p'])
            # Add more layer types as needed

            model_layers.append(layer)
            input_dim = layer_params.get('output_dim', input_dim)

        model = nn.Sequential(*model_layers)
        return model
