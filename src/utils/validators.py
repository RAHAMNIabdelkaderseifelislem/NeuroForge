from typing import Any, Dict, List, Optional, Union, Tuple
import torch
import torch.nn as nn
from pathlib import Path
import pandas as pd
import numpy as np

class DataValidator:
    @staticmethod
    def validate_dataframe(
        df: pd.DataFrame,
        required_columns: Optional[List[str]] = None,
        categorical_columns: Optional[List[str]] = None,
        numerical_columns: Optional[List[str]] = None
    ) -> Tuple[bool, List[str]]:
        """
        Validate a pandas DataFrame against specified requirements.
        Returns (is_valid, error_messages).
        """
        errors = []
        
        # Check if DataFrame is empty
        if df.empty:
            errors.append("DataFrame is empty")
            return False, errors
            
        # Validate required columns
        if required_columns:
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                errors.append(f"Missing required columns: {missing_cols}")
        
        # Validate categorical columns
        if categorical_columns:
            for col in categorical_columns:
                if col not in df.columns:
                    continue
                if not pd.api.types.is_categorical_dtype(df[col]) and not pd.api.types.is_object_dtype(df[col]):
                    errors.append(f"Column {col} should be categorical")
        
        # Validate numerical columns
        if numerical_columns:
            for col in numerical_columns:
                if col not in df.columns:
                    continue
                if not pd.api.types.is_numeric_dtype(df[col]):
                    errors.append(f"Column {col} should be numeric")
        
        return len(errors) == 0, errors

class ModelValidator:
    @staticmethod
    def validate_layer_compatibility(
        layers: List[nn.Module],
        input_shape: Tuple[int, ...]
    ) -> Tuple[bool, List[str]]:
        """
        Validate if layers can be connected together.
        Returns (is_valid, error_messages).
        """
        errors = []
        current_shape = input_shape
        
        try:
            x = torch.randn(1, *input_shape)
            for i, layer in enumerate(layers):
                try:
                    x = layer(x)
                    current_shape = tuple(x.shape[1:])
                except Exception as e:
                    errors.append(f"Layer {i} ({layer.__class__.__name__}) is incompatible: {str(e)}")
                    break
        except Exception as e:
            errors.append(f"Error during layer validation: {str(e)}")
        
        return len(errors) == 0, errors

    @staticmethod
    def validate_optimizer_config(
        optimizer_name: str,
        optimizer_params: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Validate optimizer configuration.
        Returns (is_valid, error_messages).
        """
        errors = []
        valid_optimizers = {
            'adam': torch.optim.Adam,
            'sgd': torch.optim.SGD,
            'adamw': torch.optim.AdamW,
            'rmsprop': torch.optim.RMSprop
        }
        
        if optimizer_name.lower() not in valid_optimizers:
            errors.append(f"Invalid optimizer: {optimizer_name}. Valid options: {list(valid_optimizers.keys())}")
            return False, errors
            
        try:
            # Try to instantiate optimizer with a dummy parameter
            dummy_param = nn.Parameter(torch.empty(1))
            optimizer_class = valid_optimizers[optimizer_name.lower()]
            optimizer_class([dummy_param], **optimizer_params)
        except Exception as e:
            errors.append(f"Invalid optimizer parameters: {str(e)}")
        
        return len(errors) == 0, errors

class PathValidator:
    @staticmethod
    def validate_file_path(
        path: Union[str, Path],
        required_suffix: Optional[str] = None,
        must_exist: bool = True
    ) -> Tuple[bool, List[str]]:
        """
        Validate a file path.
        Returns (is_valid, error_messages).
        """
        errors = []
        path = Path(path)
        
        if must_exist and not path.exists():
            errors.append(f"File does not exist: {path}")
        
        if required_suffix and path.suffix != required_suffix:
            errors.append(f"File must have suffix {required_suffix}, got {path.suffix}")
        
        return len(errors) == 0, errors

    @staticmethod
    def validate_directory_path(
        path: Union[str, Path],
        must_exist: bool = True,
        must_be_empty: bool = False
    ) -> Tuple[bool, List[str]]:
        """
        Validate a directory path.
        Returns (is_valid, error_messages).
        """
        errors = []
        path = Path(path)
        
        if must_exist and not path.exists():
            errors.append(f"Directory does not exist: {path}")
        elif path.exists():
            if not path.is_dir():
                errors.append(f"Path exists but is not a directory: {path}")
            elif must_be_empty and any(path.iterdir()):
                errors.append(f"Directory is not empty: {path}")
        
        return len(errors) == 0, errors

def validate_device(device: str) -> Tuple[bool, List[str]]:
    """
    Validate if a device string is valid and available.
    Returns (is_valid, error_messages).
    """
    errors = []
    
    if device == "cpu":
        return True, errors
        
    if not torch.cuda.is_available():
        errors.append("CUDA is not available on this system")
        return False, errors
        
    if device.startswith("cuda:"):
        try:
            device_idx = int(device.split(":")[1])
            if device_idx >= torch.cuda.device_count():
                errors.append(f"CUDA device {device_idx} is not available. "
                            f"Available devices: 0 to {torch.cuda.device_count()-1}")
        except ValueError:
            errors.append(f"Invalid CUDA device format: {device}")
    else:
        errors.append(f"Invalid device string: {device}")
    
    return len(errors) == 0, errors