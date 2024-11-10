from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

class BaseConfig(BaseSettings):
    """Base configuration class with common settings."""
    
    # Project paths
    PROJECT_ROOT: Path = Field(
        default=Path(__file__).parent.parent.parent,
        description="Root directory of the project"
    )
    DATA_DIR: Path = Field(default=PROJECT_ROOT / "data")
    MODELS_DIR: Path = Field(default=PROJECT_ROOT / "models")
    LOGS_DIR: Path = Field(default=PROJECT_ROOT / "logs")
    
    # Environment settings
    ENV: str = Field(default="development", description="Environment (development/production)")
    DEBUG: bool = Field(default=True, description="Debug mode flag")
    
    # CUDA settings
    CUDA_ENABLED: bool = Field(default=True, description="Enable CUDA if available")
    CUDA_DEVICE: str = Field(default="cuda:0", description="CUDA device to use")
    
    # Logging settings
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    
    class Config:
        env_prefix = "NEUROFORGE_"
        case_sensitive = False
        
    def get_device(self) -> str:
        """Get the appropriate device (cuda/cpu) based on settings and availability."""
        import torch
        if self.CUDA_ENABLED and torch.cuda.is_available():
            return self.CUDA_DEVICE
        return "cpu"