import os
import yaml
from pathlib import Path
from typing import Dict, Any, List


# Default configuration template
DEFAULT_CONFIG = {
    "data": {
        "metadata_path": "../data/processed_datasets/metadata/all_voice_metadata.json",
        "emotions": ["neutral", "happy", "sad", "angry", "surprised", "fearful", "disgusted"],
        "sample_rate": 22050,
        "samples_per_emotion": None,
        "av_ranges": {
            "arousal": [0.0, 1.0],
            "valence": [0.0, 1.0]
        },
        "default_av_config": {
            "neutral": [0.5, 0.5],
            "sample_rate": 22050
        }
    },
    "model": {
        "emotion_embedding_dim": 256,
        "unfreeze_last_n_layers": 0
    },
    "training": {
        "num_epochs": 10,
        "learning_rate": 5e-5,
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
        "checkpoint_every": 500,
        "save_best_only": True,
        "warmup_steps": 100
    },
    "optimization": {
        "weight_decay": 0.01,
        "gradient_clip_norm": 1.0
    },
    "paths": {
        "checkpoint_dir": "checkpoints/emotion_xtts",
        "log_dir": "logs/emotion_xtts",
        "model_dir": "./models/xtts_v2"
    },
    "logging": {
        "save_frequency": 100,
        "use_wandb": False,
        "use_tensorboard": True,
        "tensorboard": {
            "log_dir": "logs/tensorboard"
        }
    },
    "device": "cuda",
    "mixed_precision": True,
    "num_workers": 4,
    "seed": 42
}


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file with fallback to defaults.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        dict: Loaded and validated configuration
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"✅ Loaded configuration from {config_path}")
        
        # Merge with defaults to ensure all required fields exist
        merged_config = merge_configs(DEFAULT_CONFIG, config)
        
        # Validate the configuration
        validation_errors = validate_config(merged_config)
        if validation_errors:
            print("⚠️ Configuration validation warnings:")
            for error in validation_errors:
                print(f"  - {error}")
        
        return merged_config
        
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML configuration: {e}")
    except Exception as e:
        raise RuntimeError(f"Error loading configuration: {e}")


def merge_configs(default_config: Dict, user_config: Dict) -> Dict:
    """
    Recursively merge user config with default config.
    
    Args:
        default_config: Default configuration template
        user_config: User-provided configuration
        
    Returns:
        dict: Merged configuration
    """
    merged = default_config.copy()
    
    for key, value in user_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def validate_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate configuration values and return list of warnings/errors.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        list: List of validation warnings/errors
    """
    warnings = []
    
    # Validate data section
    data_config = config.get('data', {})
    
    # Check metadata path
    metadata_path = data_config.get('metadata_path')
    if metadata_path and not os.path.exists(metadata_path):
        warnings.append(f"Metadata file not found: {metadata_path}")
    
    # Validate AV ranges
    av_ranges = data_config.get('av_ranges', {})
    for emotion, range_vals in av_ranges.items():
        if not isinstance(range_vals, list) or len(range_vals) != 2:
            warnings.append(f"Invalid AV range for {emotion}: expected [min, max]")
        elif range_vals[0] >= range_vals[1]:
            warnings.append(f"Invalid AV range for {emotion}: min >= max")
    
    # Validate sample rate
    sample_rate = data_config.get('sample_rate', 22050)
    if not isinstance(sample_rate, int) or sample_rate <= 0:
        warnings.append("Invalid sample_rate: must be positive integer")
    
    # Validate training parameters
    training_config = config.get('training', {})
    
    learning_rate = training_config.get('learning_rate', 5e-5)
    if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
        warnings.append("Invalid learning_rate: must be positive number")
    
    batch_size = training_config.get('batch_size', 1)
    if not isinstance(batch_size, int) or batch_size <= 0:
        warnings.append("Invalid batch_size: must be positive integer")
    
    # Validate paths
    paths_config = config.get('paths', {})
    for path_name, path_value in paths_config.items():
        if path_value and not isinstance(path_value, str):
            warnings.append(f"Invalid path for {path_name}: must be string")
    
    # Validate device
    device = config.get('device', 'cuda')
    if device not in ['cuda', 'cpu', 'auto']:
        warnings.append(f"Invalid device: {device}. Expected 'cuda', 'cpu', or 'auto'")
    
    return warnings


def save_config(config: Dict[str, Any], save_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save configuration
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"💾 Configuration saved to {save_path}")


def get_config_summary(config: Dict[str, Any]) -> str:
    """
    Generate a human-readable summary of configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        str: Configuration summary
    """
    summary_lines = [
        "📋 Configuration Summary:",
        f"  Model: {config['model']['emotion_embedding_dim']}-dim emotion embedding",
        f"  Training: {config['training']['num_epochs']} epochs, LR={config['training']['learning_rate']:.1e}",
        f"  Batch: {config['training']['batch_size']} × {config['training']['gradient_accumulation_steps']} accumulation",
        f"  Audio: {config['data']['sample_rate']}Hz sample rate",
        f"  Device: {config['device']}",
        f"  Emotions: {len(config['data']['emotions'])} classes"
    ]
    
    if config['model']['unfreeze_last_n_layers'] > 0:
        summary_lines.append(f"  Fine-tuning: Last {config['model']['unfreeze_last_n_layers']} XTTS layers")
    
    if config.get('mixed_precision', False):
        summary_lines.append("  Mixed precision: Enabled")
    
    return "\n".join(summary_lines)


def update_config_paths(config: Dict[str, Any], base_path: str = None) -> Dict[str, Any]:
    """
    Update relative paths in config to be absolute or relative to base_path.
    
    Args:
        config: Configuration dictionary
        base_path: Base path for resolving relative paths
        
    Returns:
        dict: Configuration with updated paths
    """
    if base_path is None:
        base_path = os.getcwd()
    
    base_path = Path(base_path)
    updated_config = config.copy()
    
    # Update paths section
    if 'paths' in updated_config:
        for path_key, path_value in updated_config['paths'].items():
            if path_value and not os.path.isabs(path_value):
                updated_config['paths'][path_key] = str(base_path / path_value)
    
    # Update data paths
    if 'data' in updated_config:
        metadata_path = updated_config['data'].get('metadata_path')
        if metadata_path and not os.path.isabs(metadata_path):
            updated_config['data']['metadata_path'] = str(base_path / metadata_path)
    
    return updated_config


def create_config_template(output_path: str):
    """
    Create a template configuration file with comments.
    
    Args:
        output_path: Path to save template configuration
    """
    template_content = """
# EmotionXTTS Configuration Template
# This file contains all configurable parameters for training and inference

data:
  metadata_path: "../data/processed_datasets/metadata/all_voice_metadata.json"
  emotions: ["neutral", "happy", "sad", "angry", "surprised", "fearful", "disgusted"]
  sample_rate: 22050
  samples_per_emotion: null  # Use all available data
  
  # Arousal-Valence configuration (range 0-1)
  av_ranges:
    arousal: [0.0, 1.0]
    valence: [0.0, 1.0]
  
  default_av_config:
    neutral: [0.5, 0.5]  # [arousal, valence] for neutral emotion

model:
  emotion_embedding_dim: 256      # Dimension of emotion embeddings
  unfreeze_last_n_layers: 0       # Number of XTTS layers to fine-tune (0 = adapter only)

training:
  num_epochs: 10
  learning_rate: 5e-5
  batch_size: 1                   # XTTS works best with batch_size=1
  gradient_accumulation_steps: 8  # Effective batch size = batch_size × accumulation_steps
  
  # Checkpointing
  checkpoint_every: 500
  save_best_only: true
  
  # Learning rate schedule
  warmup_steps: 100

optimization:
  weight_decay: 0.01
  gradient_clip_norm: 1.0

paths:
  checkpoint_dir: "checkpoints/emotion_xtts"
  log_dir: "logs/emotion_xtts"
  model_dir: "./models/xtts_v2"

# Logging configuration
logging:
  save_frequency: 100
  use_wandb: false
  use_tensorboard: true
  tensorboard:
    log_dir: "logs/tensorboard"

# Hardware settings
device: "cuda"           # "cuda", "cpu", or "auto"
mixed_precision: true    # Enable automatic mixed precision
num_workers: 4          # DataLoader workers

# Reproducibility
seed: 42
""".strip()
    
    with open(output_path, 'w') as f:
        f.write(template_content)
    
    print(f"📝 Configuration template created at {output_path}")


if __name__ == "__main__":
    # Example usage
    create_config_template("config_template.yaml")
    
    # Test loading and validation
    try:
        config = load_config("config.yaml")
        print(get_config_summary(config))
    except FileNotFoundError:
        print("No config.yaml found - use config_template.yaml as starting point")