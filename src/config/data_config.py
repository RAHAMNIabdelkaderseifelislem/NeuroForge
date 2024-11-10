from typing import List, Optional, Union
from pathlib import Path

from pydantic import BaseModel, Field, validator

class DataConfig(BaseModel):
    """Configuration for data processing."""
    
    # Data source
    data_path: Path
    data_format: str = Field(default="csv")
    
    # Preprocessing
    target_column: str
    feature_columns: Optional[List[str]] = None
    categorical_columns: Optional[List[str]] = None
    numerical_columns: Optional[List[str]] = None
    
    # Data splitting
    train_split: float = Field(default=0.8, ge=0, le=1)
    random_seed: int = Field(default=42)
    
    # Data preprocessing
    normalize: bool = Field(default=True)
    normalization_method: str = Field(default="standard")
    handle_missing: bool = Field(default=True)
    missing_strategy: str = Field(default="mean")
    
    @validator("data_format")
    def validate_data_format(cls, v):
        valid_formats = ["csv", "excel", "parquet", "json"]
        if v.lower() not in valid_formats:
            raise ValueError(f"Data format must be one of {valid_formats}")
        return v.lower()
    
    @validator("normalization_method")
    def validate_normalization(cls, v):
        valid_methods = ["standard", "minmax", "robust"]
        if v.lower() not in valid_methods:
            raise ValueError(f"Normalization method must be one of {valid_methods}")
        return v.lower()