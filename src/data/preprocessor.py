from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from src.config.data_config import DataConfig

class DataPreprocessor:
    """Handles all data preprocessing operations."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self._scalers: Dict[str, object] = {}
        
    def load_data(self) -> pd.DataFrame:
        """Load data based on format specified in config."""
        if self.config.data_format == "csv":
            return pd.read_csv(self.config.data_path)
        elif self.config.data_format == "excel":
            return pd.read_excel(self.config.data_path)
        elif self.config.data_format == "parquet":
            return pd.read_parquet(self.config.data_path)
        elif self.config.data_format == "json":
            return pd.read_json(self.config.data_path)
        raise ValueError(f"Unsupported data format: {self.config.data_format}")

    def preprocess(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
        """Apply preprocessing steps based on config."""
        if self.config.handle_missing:
            df = self._handle_missing_values(df)
            
        if self.config.normalize:
            df = self._normalize_features(df)
            
        metadata = {
            "feature_columns": self.config.feature_columns,
            "target_column": self.config.target_column,
            "scalers": self._scalers
        }
        
        return df, metadata
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values using strategy from config."""
        if self.config.missing_strategy == "mean":
            return df.fillna(df.mean())
        elif self.config.missing_strategy == "median":
            return df.fillna(df.median())
        elif self.config.missing_strategy == "drop":
            return df.dropna()
        return df
    
    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize numerical features using method from config."""
        numerical_columns = (self.config.numerical_columns or 
                           df.select_dtypes(include=[np.number]).columns)
        
        scaler = None
        if self.config.normalization_method == "standard":
            scaler = StandardScaler()
        elif self.config.normalization_method == "minmax":
            scaler = MinMaxScaler()
        elif self.config.normalization_method == "robust":
            scaler = RobustScaler()
            
        if scaler and len(numerical_columns) > 0:
            df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
            self._scalers["numerical"] = scaler
            
        return df

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and test sets."""
        return train_test_split(
            df, 
            train_size=self.config.train_split,
            random_state=self.config.random_seed
        )