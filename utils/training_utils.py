import shutil
from pathlib import Path
import torch


def get_device_info():
    device_info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    }
    
    if torch.cuda.is_available():
        device_info.update({
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "memory_allocated": torch.cuda.memory_allocated(0),
            "memory_reserved": torch.cuda.memory_reserved(0),
            "max_memory": torch.cuda.max_memory_allocated(0)
        })
    
    return device_info


def move_model_to_device(model, device):
    try:
        if isinstance(device, str):
            device = torch.device(device)
        
        model = model.to(device)
        print(f"✅ Moved model to {device}")
        return model
    except Exception as e:
        print(f"⚠️ Warning: Failed to move model to {device}: {e}")
        return model


def freeze_model_parameters(model, freeze=True):
    """
    Freeze or unfreeze model parameters for training control.
    
    Args:
        model: PyTorch model
        freeze: Whether to freeze (True) or unfreeze (False) parameters
    """
    for param in model.parameters():
        param.requires_grad = not freeze
    
    status = "frozen" if freeze else "unfrozen"
    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📊 Model parameters {status}: {param_count:,} total, {trainable_count:,} trainable")


def get_model_parameter_info(model):
    """
    Get detailed information about model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        dict: Parameter information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    # Calculate memory usage
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)  # MB
    
    # Get parameter breakdown by layer type
    layer_info = {}
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            module_params = sum(p.numel() for p in module.parameters())
            if module_params > 0:
                layer_type = type(module).__name__
                if layer_type not in layer_info:
                    layer_info[layer_type] = {"count": 0, "params": 0}
                layer_info[layer_type]["count"] += 1
                layer_info[layer_type]["params"] += module_params
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params, 
        "frozen_parameters": frozen_params,
        "memory_mb": param_memory,
        "trainable_percentage": (trainable_params / total_params * 100) if total_params > 0 else 0,
        "layer_breakdown": layer_info
    }


def setup_training_device(config):
    """
    Setup training device based on config and hardware availability.
    
    Args:
        config: Training configuration dict
        
    Returns:
        torch.device: Configured device for training
    """
    device_config = config.get('device', 'auto')
    
    if device_config == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"🚀 Auto-selected CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            print("💻 Using CPU (CUDA not available)")
    else:
        device = torch.device(device_config)
        if device.type == 'cuda' and not torch.cuda.is_available():
            print("⚠️ CUDA requested but not available, falling back to CPU")
            device = torch.device('cpu')
    
    return device


def cleanup_temp_files(temp_dir="./temp"):
    """
    Clean up temporary files and directories.
    
    Args:
        temp_dir: Directory to clean up
    """
    temp_path = Path(temp_dir)
    if temp_path.exists():
        try:
            shutil.rmtree(temp_path)
            print(f"🧹 Cleaned up temporary files in {temp_dir}")
        except Exception as e:
            print(f"⚠️ Warning: Failed to clean up {temp_dir}: {e}")


def save_training_state(model, optimizer, epoch, loss, save_path):
    """
    Save complete training state for resuming training.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss value
        save_path: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'model_info': get_model_parameter_info(model)
    }
    
    torch.save(checkpoint, save_path)
    print(f"💾 Training state saved to {save_path}")


def load_training_state(model, optimizer, load_path, device):
    """
    Load training state for resuming training.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer to restore state
        load_path: Path to checkpoint
        device: Device to load tensors on
        
    Returns:
        tuple: (epoch, loss) from saved state
    """
    checkpoint = torch.load(load_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"📂 Training state loaded from {load_path} (epoch {epoch}, loss {loss:.4f})")
    return epoch, loss


def monitor_gpu_memory():
    """
    Monitor and report GPU memory usage during training.
    
    Returns:
        dict: GPU memory statistics
    """
    if not torch.cuda.is_available():
        return {"message": "CUDA not available"}
    
    allocated = torch.cuda.memory_allocated(0) / (1024**3)  # GB
    reserved = torch.cuda.memory_reserved(0) / (1024**3)    # GB
    max_allocated = torch.cuda.max_memory_allocated(0) / (1024**3)  # GB
    
    # Get total GPU memory
    device_properties = torch.cuda.get_device_properties(0)
    total_memory = device_properties.total_memory / (1024**3)  # GB
    
    memory_stats = {
        "allocated_gb": allocated,
        "reserved_gb": reserved,
        "max_allocated_gb": max_allocated,
        "total_memory_gb": total_memory,
        "utilization_percent": (allocated / total_memory * 100) if total_memory > 0 else 0
    }
    
    return memory_stats


def optimize_model_for_training(model, config):
    """
    Apply training optimizations based on config.
    
    Args:
        model: PyTorch model
        config: Training configuration
        
    Returns:
        model: Optimized model
    """
    # Apply mixed precision if enabled
    if config.get('mixed_precision', False):
        print("⚡ Enabling mixed precision training")
        # Note: Actual mixed precision setup would be done in training loop
        # with torch.cuda.amp.GradScaler and autocast
    
    # Enable gradient checkpointing if specified
    if config.get('gradient_checkpointing', False):
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            print("💾 Enabled gradient checkpointing")
    
    # Set model to training mode
    model.train()
    
    return model