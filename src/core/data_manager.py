from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import torch
from torch.utils.data import Dataset, DataLoader

from ..config.data_config import DataConfig
from ..utils.validators import DataValidator
from ..utils.helpers import ensure_directory

class CustomDataset(Dataset):
    """Custom PyTorch Dataset for handling tabular data."""
    def __init__(self, features: torch.Tensor, targets: torch.Tensor):
        self.features = features
        self.targets = targets
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]

class DataManager:
    """Manages data loading, preprocessing, and preparation for training."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.data: Optional[pd.DataFrame] = None
        self.scaler: Optional[Union[StandardScaler, MinMaxScaler, RobustScaler]] = None
        self._validate_config()
        
    def _validate_config(self) -> None:
        """Validate the data configuration."""
        is_valid, errors = DataValidator.validate_dataframe(
            pd.DataFrame(),  # Empty DataFrame for schema validation
            required_columns=[self.config.target_column],
            categorical_columns=self.config.categorical_columns,
            numerical_columns=self.config.numerical_columns
        )
        if not is_valid:
            raise ValueError(f"Invalid data configuration: {errors}")
    
    def load_data(self, data_path: Optional[Path] = None) -> pd.DataFrame:
        """Load data from specified path or config path."""
        path = data_path or self.config.data_path
        
        if self.config.data_format == 'csv':
            self.data = pd.read_csv(path)
        elif self.config.data_format == 'excel':
            self.data = pd.read_excel(path)
        elif self.config.data_format == 'parquet':
            self.data = pd.read_parquet(path)
        elif self.config.data_format == 'json':
            self.data = pd.read_json(path)
        else:
            raise ValueError(f"Unsupported data format: {self.config.data_format}")
            
        self._validate_loaded_data()
        return self.data
    
    def _validate_loaded_data(self) -> None:
        """Validate loaded data against configuration."""
        if self.data is None:
            raise ValueError("No data loaded")
            
        is_valid, errors = DataValidator.validate_dataframe(
            self.data,
            required_columns=[self.config.target_column],
            categorical_columns=self.config.categorical_columns,
            numerical_columns=self.config.numerical_columns
        )
        if not is_valid:
            raise ValueError(f"Invalid data: {errors}")
    
    def preprocess_data(self) -> pd.DataFrame:
        """Preprocess the loaded data according to configuration."""
        if self.data is None:
            raise ValueError("No data loaded")
            
        # Handle missing values
        if self.config.handle_missing:
            self._handle_missing_values()
        
        # Convert categorical variables
        if self.config.categorical_columns:
            self._encode_categorical_variables()
        
        # Normalize numerical variables
        if self.config.normalize:
            self._normalize_numerical_variables()
        
        return self.data
    
    def _handle_missing_values(self) -> None:
        """Handle missing values according to configuration."""
        for column in self.data.columns:
            if self.data[column].isnull().any():
                if column in (self.config.categorical_columns or []):
                    self.data[column].fillna(self.data[column].mode()[0], inplace=True)
                else:
                    if self.config.missing_strategy == 'mean':
                        self.data[column].fillna(self.data[column].mean(), inplace=True)
                    elif self.config.missing_strategy == 'median':
                        self.data[column].fillna(self.data[column].median(), inplace=True)
    
    def _encode_categorical_variables(self) -> None:
        """Encode categorical variables using one-hot encoding."""
        if not self.config.categorical_columns:
            return
            
        self.data = pd.get_dummies(
            self.data,
            columns=self.config.categorical_columns,
            drop_first=True
        )
    
    def _normalize_numerical_variables(self) -> None:
        """Normalize numerical variables according to configuration."""
        if not self.config.numerical_columns:
            return
            
        if self.config.normalization_method == 'standard':
            self.scaler = StandardScaler()
        elif self.config.normalization_method == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.config.normalization_method == 'robust':
            self.scaler = RobustScaler()
            
        self.data[self.config.numerical_columns] = self.scaler.fit_transform(
            self.data[self.config.numerical_columns]
        )
    
    def prepare_data_loaders(
        self,
        batch_size: int = 32,
        shuffle: bool = True
    ) -> Tuple[DataLoader, DataLoader]:
        """Prepare train and validation DataLoaders."""
        if self.data is None:
            raise ValueError("No data loaded")
            
        # Split features and target
        X = self.data.drop(columns=[self.config.target_column])
        y = self.data[self.config.target_column]
        
        # Split data into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            train_size=self.config.train_split,
            random_state=self.config.random_seed
        )
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train.values)
        y_train_tensor = torch.FloatTensor(y_train.values)
        X_val_tensor = torch.FloatTensor(X_val.values)
        y_val_tensor = torch.FloatTensor(y_val.values)
        
        # Create datasets
        train_dataset = CustomDataset(X_train_tensor, y_train_tensor)
        val_dataset = CustomDataset(X_val_tensor, y_val_tensor)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        return train_loader, val_loader
    
    def get_feature_dimensions(self) -> Dict[str, int]:
        """Get input and output dimensions for model creation."""
        if self.data is None:
            raise ValueError("No data loaded")
            
        return {
            "input_dim": len(self.data.drop(columns=[self.config.target_column]).columns),
            "output_dim": 1 if pd.api.types.is_numeric_dtype(self.data[self.config.target_column]) else len(self.data[self.config.target_column].unique())
        }