import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import torch
import yaml
import json
from datetime import datetime

logger = logging.getLogger(__name__)

def setup_logging(log_path: Optional[Path] = None, level: str = "INFO") -> None:
    """Configure logging with custom format and optional file output."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=getattr(logging, level), format=log_format)
    
    if log_path:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)

def load_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML file safely."""
    try:
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading YAML file {file_path}: {str(e)}")
        raise

def save_yaml(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """Save data to a YAML file."""
    try:
        with open(file_path, 'w') as f:
            yaml.safe_dump(data, f, default_flow_style=False)
    except Exception as e:
        logger.error(f"Error saving YAML file {file_path}: {str(e)}")
        raise

def get_available_devices() -> List[str]:
    """Get list of available CUDA devices."""
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.extend([f'cuda:{i}' for i in range(torch.cuda.device_count())])
    return devices

def generate_model_name(prefix: str = "model") -> str:
    """Generate a unique model name with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"

def count_model_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count trainable and non-trainable parameters in a PyTorch model."""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    return {
        "trainable": trainable_params,
        "non_trainable": total_params - trainable_params,
        "total": total_params
    }

def create_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: Union[str, Path]
) -> None:
    """Create a model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

def load_checkpoint(
    path: Union[str, Path],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None
) -> Dict[str, Any]:
    """Load a model checkpoint."""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint

def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure a directory exists, create if it doesn't."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def format_time(seconds: float) -> str:
    """Format time in seconds to human readable string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{seconds}s")
    
    return " ".join(parts)

class TensorUtils:
    @staticmethod
    def to_device(data: Any, device: str) -> Any:
        """Move tensor or collection of tensors to specified device."""
        if isinstance(data, (torch.Tensor, torch.nn.Module)):
            return data.to(device)
        elif isinstance(data, dict):
            return {key: TensorUtils.to_device(value, device) for key, value in data.items()}
        elif isinstance(data, (list, tuple)):
            return type(data)(TensorUtils.to_device(item, device) for item in data)
        return data

    @staticmethod
    def get_tensor_info(tensor: torch.Tensor) -> Dict[str, Any]:
        """Get information about a tensor."""
        return {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "memory_format": str(tensor.memory_format),
            "requires_grad": tensor.requires_grad
        }