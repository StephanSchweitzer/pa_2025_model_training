"""
XTTS Model Loading and Management Utilities

Handles downloading, loading, and verification of XTTS v2 models.
Provides clean separation of model management from other utilities.
"""

import os
import shutil
from pathlib import Path
from TTS.utils.manage import ModelManager
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


DEFAULT_XTTS_CONFIG = {
    "model_name": "tts_models/multilingual/multi-dataset/xtts_v2",
    "local_dir": "./models/xtts_v2",
    "sample_rate": 22050,
    "supported_languages": ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja"]
}


def ensure_local_model_exists(local_model_dir="./models/xtts_v2", model_name="tts_models/multilingual/multi-dataset/xtts_v2"):
    """
    Download XTTS model if it doesn't exist locally.
    
    Args:
        local_model_dir: Directory to store the model
        model_name: TTS model name to download
        
    Returns:
        str: Path to config.json file
    """
    local_model_dir = Path(local_model_dir)
    config_file = local_model_dir / "config.json"
    
    if config_file.exists():
        print(f"Model already exists at {local_model_dir}")
        return str(config_file)
    
    print(f"Downloading {model_name} to {local_model_dir}")
    
    local_model_dir.mkdir(parents=True, exist_ok=True)
    
    manager = ModelManager()
    
    try:
        temp_model_path, temp_config_path, _ = manager.download_model(model_name)
        temp_model_dir = Path(temp_model_path)
        
        print(f"Downloaded to temporary location: {temp_model_dir}")
        
        if temp_model_dir.is_dir():
            for item in temp_model_dir.rglob("*"):
                if item.is_file():
                    relative_path = item.relative_to(temp_model_dir)
                    dest_path = local_model_dir / relative_path
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, dest_path)
                    print(f"Copied {relative_path}")
        else:
            shutil.copy2(temp_config_path, local_model_dir / "config.json")
            if Path(temp_model_path).exists():
                shutil.copy2(temp_model_path, local_model_dir)
        
        print(f"Model successfully saved to {local_model_dir}")
        return str(local_model_dir / "config.json")
        
    except Exception as e:
        raise RuntimeError(f"Failed to download and save XTTS model: {e}")


def load_xtts_from_local(local_model_dir="./models/xtts_v2"):
    """
    Load XTTS model from local directory.
    
    Args:
        local_model_dir: Path to local model directory
        
    Returns:
        tuple: (xtts_model, config)
    """
    local_model_dir = Path(local_model_dir)
    config_path = local_model_dir / "config.json"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    try:
        config = XttsConfig()
        config.load_json(str(config_path))
        print(f"Loaded config from {config_path}")
        
        xtts = Xtts.init_from_config(config)
        print("🔧 Initialized XTTS from config")
        
        xtts.load_checkpoint(
            config, 
            checkpoint_dir=str(local_model_dir), 
            use_deepspeed=False
        )
        print(f"🔄 Loaded checkpoint from {local_model_dir}")
        
        return xtts, config
        
    except Exception as e:
        raise RuntimeError(f"Failed to load XTTS model from {local_model_dir}: {e}")


def load_xtts_from_paths(config_path, checkpoint_path):
    """
    Load XTTS model from specific config and checkpoint paths.
    
    Args:
        config_path: Path to config.json
        checkpoint_path: Path to checkpoint directory
        
    Returns:
        tuple: (xtts_model, config)
    """
    try:
        config = XttsConfig()
        config.load_json(config_path)
        xtts = Xtts.init_from_config(config)
        xtts.load_checkpoint(config, checkpoint_path, use_deepspeed=False)
        print(f"Loaded from provided paths: {config_path}, {checkpoint_path}")
        return xtts, config
    except Exception as e:
        raise RuntimeError(f"Failed to load from provided paths: {e}")


def load_xtts_model(config_path=None, checkpoint_path=None, local_model_dir="./models/xtts_v2"):
    """
    Universal XTTS model loader - automatically chooses loading method.
    
    Args:
        config_path: Optional path to config.json
        checkpoint_path: Optional path to checkpoint
        local_model_dir: Local model directory for fallback
        
    Returns:
        tuple: (xtts_model, config)
    """
    if config_path and checkpoint_path:
        return load_xtts_from_paths(config_path, checkpoint_path)
    else:
        ensure_local_model_exists(local_model_dir)
        return load_xtts_from_local(local_model_dir)


def verify_xtts_components(xtts_model):
    """
    Verify that XTTS model components are loaded correctly.
    
    Args:
        xtts_model: Loaded XTTS model instance
    """
    if not hasattr(xtts_model, 'gpt') or xtts_model.gpt is None:
        raise RuntimeError("XTTS GPT model is None - model not loaded properly")
    
    vocoder_found = False
    vocoder_attrs = ['hifigan', 'vocoder', 'decoder', 'hifigan_decoder']
    
    for attr_name in vocoder_attrs:
        if hasattr(xtts_model, attr_name) and getattr(xtts_model, attr_name) is not None:
            print(f"✅ Found vocoder component: {attr_name}")
            vocoder_found = True
            break
    
    if not vocoder_found:
        print("⚠️  Warning: No vocoder component found")
        print("Available XTTS attributes:", [attr for attr in dir(xtts_model) if not attr.startswith('_')])


def get_xtts_model_info(xtts_model, config, local_model_dir="./models/xtts_v2"):
    """
    Get comprehensive information about loaded XTTS model.
    
    Args:
        xtts_model: Loaded XTTS model
        config: XTTS config object
        local_model_dir: Model directory path
        
    Returns:
        dict: Model information
    """
    local_model_dir = Path(local_model_dir)
    
    return {
        "model_dir": str(local_model_dir),
        "config_loaded": config is not None,
        "xtts_loaded": xtts_model is not None,
        "gpt_available": hasattr(xtts_model, 'gpt') and xtts_model.gpt is not None,
        "model_files": list(local_model_dir.glob("*")) if local_model_dir.exists() else [],
        "model_device": next(xtts_model.parameters()).device if xtts_model else "unknown",
        "gpt_n_model_channels": getattr(config.model_args, 'gpt_n_model_channels', None) if config else None
    }