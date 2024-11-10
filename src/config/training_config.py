from typing import List, Optional, Union

from pydantic import BaseModel, Field, validator

class TrainingConfig(BaseModel):
    """Configuration for model training."""
    
    # Training hyperparameters
    batch_size: int = Field(default=32, ge=1)
    learning_rate: float = Field(default=0.001, gt=0)
    epochs: int = Field(default=10, ge=1)
    
    # Optimizer settings
    optimizer: str = Field(default="adam", description="PyTorch optimizer to use")
    optimizer_params: dict = Field(default_factory=dict)
    
    # Learning rate scheduler
    lr_scheduler: Optional[str] = Field(default=None)
    lr_scheduler_params: dict = Field(default_factory=dict)
    
    # Early stopping
    early_stopping_patience: int = Field(default=5, ge=0)
    early_stopping_min_delta: float = Field(default=1e-4, ge=0)
    
    # Validation
    validation_split: float = Field(default=0.2, ge=0, le=1)
    
    @validator("optimizer")
    def validate_optimizer(cls, v):
        valid_optimizers = ["adam", "sgd", "adamw", "rmsprop"]
        if v.lower() not in valid_optimizers:
            raise ValueError(f"Optimizer must be one of {valid_optimizers}")
        return v.lower()