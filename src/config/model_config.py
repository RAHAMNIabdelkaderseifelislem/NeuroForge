from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator

class LayerConfig(BaseModel):
    """Configuration for a single neural network layer."""
    
    layer_type: str
    layer_name: str
    layer_params: Dict[str, Any] = Field(default_factory=dict)
    activation: Optional[str] = None
    dropout_rate: Optional[float] = Field(default=None, ge=0, le=1)

class ModelConfig(BaseModel):
    """Configuration for the neural network model."""
    
    # Model architecture
    input_shape: List[int]
    output_shape: List[int]
    layers: List[LayerConfig]
    
    # Model compilation
    loss_function: str = Field(default="mse")
    metrics: List[str] = Field(default=["mae", "mse"])
    
    # Model saving
    model_name: str
    save_format: str = Field(default="pytorch")
    checkpoint_frequency: int = Field(default=1, ge=1)
    
    @validator("loss_function")
    def validate_loss(cls, v):
        valid_losses = ["mse", "cross_entropy", "bce", "mae"]
        if v.lower() not in valid_losses:
            raise ValueError(f"Loss function must be one of {valid_losses}")
        return v.lower()
    
    @validator("save_format")
    def validate_format(cls, v):
        valid_formats = ["pytorch", "onnx", "torchscript"]
        if v.lower() not in valid_formats:
            raise ValueError(f"Save format must be one of {valid_formats}")
        return v.lower()