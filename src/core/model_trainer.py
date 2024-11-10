import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class ModelTrainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = config.get_device()
        self.model.to(self.device)

        self.optimizer = self._get_optimizer()
        self.criterion = self._get_criterion()

    def _get_optimizer(self):
        optimizer_name = self.config.training.optimizer
        optimizer_params = self.config.training.optimizer_params

        if optimizer_name == 'adam':
            return optim.Adam(self.model.parameters(), **optimizer_params)
        elif optimizer_name == 'sgd':
            return optim.SGD(self.model.parameters(), **optimizer_params)
        # Add more optimizers as needed

    def _get_criterion(self):
        loss_function = self.config.model.loss_function

        if loss_function == 'mse':
            return nn.MSELoss()
        elif loss_function == 'cross_entropy':
            return nn.CrossEntropyLoss()
        elif loss_function == 'mae':
            return nn.L1Loss()

    def train(self):
        for epoch in range(self.config.training.epochs):
            self.model.train()
            train_loss = 0.0

            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * inputs.size(0)

            train_loss = train_loss / len(self.train_loader.dataset)
            val_loss = self.evaluate()

            print(f'Epoch {epoch+1}/{self.config.training.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    def evaluate(self):
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        val_loss = val_loss / len(self.val_loader.dataset)
        return val_loss
