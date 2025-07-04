import os
import shutil
from pathlib import Path
import torch
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
    if config_path and checkpoint_path:
        return load_xtts_from_paths(config_path, checkpoint_path)
    else:
        ensure_local_model_exists(local_model_dir)
        return load_xtts_from_local(local_model_dir)


def verify_xtts_components(xtts_model):
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


def get_model_info(xtts_model, config, local_model_dir="./models/xtts_v2"):
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


def move_model_to_device(xtts_model, device):
    try:
        if isinstance(device, str):
            device = torch.device(device)
        
        xtts_model = xtts_model.to(device)
        print(f"Moved XTTS model to {device}")
        return xtts_model
    except Exception as e:
        print(f"Warning: Failed to move XTTS model to {device}: {e}")
        return xtts_model


def freeze_model_parameters(model, freeze=True):
    for param in model.parameters():
        param.requires_grad = not freeze
    
    status = "frozen" if freeze else "unfrozen"
    print(f"Model parameters {status}")


def get_device_info():
    return {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    }


def cleanup_temp_files(temp_dir="./temp"):
    temp_path = Path(temp_dir)
    if temp_path.exists():
        try:
            shutil.rmtree(temp_path)
            print(f"🧹 Cleaned up temporary files in {temp_dir}")
        except Exception as e:
            print(f"⚠️  Warning: Failed to clean up {temp_dir}: {e}")