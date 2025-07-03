def setup_for_spyder():
    """
    Main setup function designed to run directly in Spyder.
    Uses the configuration variables defined at the top of this file.
    """
    
    print("🚀 Setting up GPT Trainer for Emotion XTTS")
    print("=" * 50)
    print(f"📄 Using config: {EMOTION_CONFIG_PATH}")
    print(f"🤖 Model directory: {MODEL_DIR}")
    print(f"⚙️ Output config: {OUTPUT_CONFIG_PATH}")
    
    # Step 1: Check if emotion config exists
    if not os.path.exists(EMOTION_CONFIG_PATH):
        print(f"❌ Emotion config not found: {EMOTION_CONFIG_PATH}")
        print("Please make sure your config.yaml file exists in the current directory.")
        return False
    
    # Step 2: Setup directories
    print("\n📁 Setting up directories...")
    setup_directories()
    
    # Step 3: Download XTTS model if requested
    if DOWNLOAD_XTTS_MODEL:
        print(f"\n📥 Downloading XTTS v2 model to {MODEL_DIR}...")
        if not download_xtts_model(MODEL_DIR):
            print("❌ Model download failed")
            return False
    else:
        print(f"\n⏭️ Skipping XTTS model download (DOWNLOAD_XTTS_MODEL = False)")
    
    # Step 4: Convert config
    print(f"\n⚙️ Converting configuration...")
    gpt_config = convert_config_for_gpt_trainer(EMOTION_CONFIG_PATH, OUTPUT_CONFIG_PATH)
    
    # Step 5: Validate setup
    print("\n🔍 Validating setup...")
    if not validate_setup():
        print("❌ Setup validation failed")
        print("\n💡 Common issues:")
        print("   - Make sure your config.yaml exists")
        print("   - Check if XTTS model files downloaded correctly")
        print("   - Verify your dataset paths in the config")
        return False
    
    print("\n✅ Setup complete!")
    print("=" * 50)
    print("📊 Setup Summary:")
    print(f"   🎯 Original config: {EMOTION_CONFIG_PATH}")
    print(f"   ⚙️ GPT config: {OUTPUT_CONFIG_PATH}")
    print(f"   🤖 XTTS model: {MODEL_DIR}")
    print(f"   📁 Working directory: {os.getcwd()}")
    
    # Load the emotion config to show dataset info
    try:
        with open(EMOTION_CONFIG_PATH, 'r') as f:
            emotion_config = yaml.safe_load(f)
        
        dataset_path = emotion_config.get('data', {}).get('dataset_path', './data')
        print(f"   📊 Dataset path: {dataset_path}")
        
        # Check if dataset files exist
        if os.path.exists(dataset_path):
            print(f"   ✅ Dataset directory found")
            
            # Look for metadata files
            train_csv = os.path.join(dataset_path, 'metadata_train.csv')
            val_csv = os.path.join(dataset_path, 'metadata_val.csv')
            
            if os.path.exists(train_csv):
                with open(train_csv, 'r') as f:
                    train_lines = len(f.readlines())
                print(f"   📝 Training samples: {train_lines}")
            
            if os.path.exists(val_csv):
                with open(val_csv, 'r') as f:
                    val_lines = len(f.readlines())
                print(f"   📝 Validation samples: {val_lines}")
        else:
            print(f"   ⚠️ Dataset directory not found: {dataset_path}")
            print("   Make sure to update the dataset_path in your config.yaml")
    
    except Exception as e:
        print(f"   ⚠️ Could not read dataset info: {e}")
    
    # Step 6: Run training if requested
    if RUN_TRAINING_AFTER_SETUP:
        print("\n🎯 Starting training...")
        try:
            run_training_from_setup()
        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n🎯 To start training, you can now run:")
        print(f"   python run_emotion_training.py --config {EMOTION_CONFIG_PATH}")
        print(f"   Or set RUN_TRAINING_AFTER_SETUP = True in this script")
    
    return True


def run_training_from_setup():
    """Run training directly from the setup script."""
    try:
        from emotion_gpt_trainer import create_emotion_gpt_config, EmotionGPTTrainer
        from TTS.trainer import Trainer, TrainerArgs
        
        # Create config and model
        config = create_emotion_gpt_config(EMOTION_CONFIG_PATH)
        model = EmotionGPTTrainer(config)
        
        # Create trainer
        trainer = Trainer(
            TrainerArgs(
                restore_path=None,
                skip_train_epoch=False,
                start_with_eval=False,
                grad_accum_steps=1,
            ),
            config,
            output_path="./runs/emotion_gpt_trainer",
            model=model,
            train_samples=[],  # Dummy samples for now
            eval_samples=[],
        )
        
        print("🚀 Training started!")
        trainer.fit()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure emotion_gpt_trainer.py is in the same directory")
        print("And that TTS is installed: pip install TTS")
    except Exception as e:
        print(f"❌ Training setup failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """
    Main function - now just calls setup_for_spyder() with the configured variables.
    This allows the script to work both in Spyder and from command line.
    """
    return setup_for_spyder()


# =============================================================================
# 🚀 SPYDER EXECUTION - This runs when you hit F5 in Spyder
# =============================================================================

if __name__ == "__main__":
    print("🔬 Running in Spyder/Direct execution mode")
    print("🔧 Using configuration variables defined at the top of this file")
    print()
    
    success = main()
    
    if success:
        print("\n🎉 Setup completed successfully!")
        print("📚 Next steps:")
        print("   1. Check that your dataset paths are correct in config.yaml")
        print("   2. Run emotion_gpt_trainer.py or run_emotion_training.py")
        print("   3. Monitor training progress in the ./runs/ directory")
    else:
        print("\n❌ Setup failed!")
        print("📋 Please check the error messages above and fix any issues.")#!/usr/bin/env python3
"""
Setup script to convert your existing emotion XTTS setup to use GPTTrainer.

This script:
1. Takes your existing config.yaml
2. Creates the necessary files for GPTTrainer
3. Downloads XTTS model if needed
4. Sets up directories and validates everything

SPYDER USAGE:
Just modify the configuration variables below and run this script (F5 in Spyder).
No command line arguments needed!
"""

import os
import sys
import yaml
import torch
import torchaudio
import numpy as np
from pathlib import Path
import urllib.request
from tqdm import tqdm

# =============================================================================
# 🔧 CONFIGURATION - EDIT THESE VARIABLES FOR YOUR SETUP
# =============================================================================

# Path to your existing emotion config file
EMOTION_CONFIG_PATH = "config.yaml"

# Whether to download XTTS model (set to False if you already have it)
DOWNLOAD_XTTS_MODEL = True

# Whether to run training immediately after setup
RUN_TRAINING_AFTER_SETUP = False

# Output paths (you can change these if needed)
OUTPUT_CONFIG_PATH = "gpt_config.yaml"
MODEL_DIR = "./models/xtts_v2"

# =============================================================================


def download_xtts_model(model_dir: str = "./models/xtts_v2"):
    """Download XTTS v2 model files."""
    
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📥 Downloading XTTS v2 to: {model_path}")
    
    files_to_download = {
        "config.json": "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/config.json",
        "model.pth": "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth",
        "vocab.json": "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json",
        "dvae.pth": "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth",
        "mel_stats.pth": "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"
    }
    
    for filename, url in files_to_download.items():
        filepath = model_path / filename
        
        if filepath.exists():
            print(f"✅ {filename} already exists")
            continue
        
        print(f"📥 Downloading {filename}...")
        try:
            urllib.request.urlretrieve(url, filepath)
            print(f"✅ Downloaded {filename}")
        except Exception as e:
            print(f"❌ Failed to download {filename}: {e}")
            return False
    
    print("✅ XTTS v2 model download complete!")
    return True


def convert_config_for_gpt_trainer(emotion_config_path: str, output_config_path: str = "gpt_config.yaml"):
    """Convert your emotion config to GPTTrainer format."""
    
    with open(emotion_config_path, 'r') as f:
        emotion_config = yaml.safe_load(f)
    
    # Create GPTTrainer compatible config
    gpt_config = {
        # Preserve your original emotion config
        'emotion_config_path': emotion_config_path,
        
        # Model configuration
        'model_args': {
            'max_conditioning_length': 132300,  # 6 secs
            'min_conditioning_length': 66150,   # 3 secs
            'max_wav_length': 255995,           # ~11.6 seconds
            'max_text_length': 200,
            'mel_norm_file': './models/xtts_v2/mel_stats.pth',
            'dvae_checkpoint': './models/xtts_v2/dvae.pth',
            'tokenizer_file': './models/xtts_v2/vocab.json',
            'gpt_n_model_channels': emotion_config.get('model', {}).get('emotion_embedding_dim', 1024),
        },
        
        # Training configuration
        'run_name': f"emotion_xtts_gpt_{emotion_config.get('experiment_name', 'default')}",
        'epochs': emotion_config.get('training', {}).get('num_epochs', 100),
        'lr': emotion_config.get('training', {}).get('learning_rate', 1e-5),
        'batch_size': 1,  # XTTS works best with batch_size=1
        'eval_every': 1000,
        'save_every': 1000,
        'print_every': 100,
        
        # Optimizer settings
        'wd': emotion_config.get('optimization', {}).get('weight_decay', 0.01),
        'grad_clip': emotion_config.get('optimization', {}).get('gradient_clip_norm', 1.0),
        
        # Dataset configuration
        'datasets': [{
            'formatter': 'ljspeech',
            'dataset_name': 'emotion_dataset',
            'path': emotion_config.get('data', {}).get('dataset_path', './data'),
            'meta_file_train': 'metadata_train.csv',
            'meta_file_val': 'metadata_val.csv',
            'language': emotion_config.get('data', {}).get('language', 'en'),
        }],
        
        # Paths
        'output_path': './runs/emotion_gpt_trainer',
        'checkpoint_dir': './checkpoints'
    }
    
    # Save the config
    with open(output_config_path, 'w') as f:
        yaml.dump(gpt_config, f, default_flow_style=False, indent=2)
    
    print(f"✅ Converted config saved to: {output_config_path}")
    return gpt_config


def setup_directories():
    """Create necessary directories."""
    
    directories = [
        "./models/xtts_v2",
        "./data", 
        "./runs",
        "./checkpoints",
        "./logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")


def validate_setup():
    """Validate that everything is set up correctly."""
    
    print("🔍 Validating setup...")
    
    required_files = [
        "./models/xtts_v2/config.json",
        "./models/xtts_v2/model.pth", 
        "./models/xtts_v2/vocab.json",
        "./models/xtts_v2/dvae.pth",
        "./models/xtts_v2/mel_stats.pth"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    # Check if data exists
    if not os.path.exists("./data/metadata_train.csv"):
        print("⚠️ No training data found")
        return False
    
    print("✅ Setup validation passed!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Setup GPT Trainer for Emotion XTTS")
    parser.add_argument("--config", type=str, default="config.yaml", 
                       help="Path to your emotion config file")
    parser.add_argument("--create_dummy_data", action="store_true",
                       help="Create dummy dataset for testing")
    parser.add_argument("--skip_download", action="store_true",
                       help="Skip XTTS model download")
    parser.add_argument("--run_training", action="store_true",
                       help="Run training after setup")
    
    args = parser.parse_args()
    
    print("🚀 Setting up GPT Trainer for Emotion XTTS")
    print("=" * 50)
    
    # Step 1: Check if emotion config exists
    if not os.path.exists(args.config):
        print(f"❌ Emotion config not found: {args.config}")
        print("Please create your config.yaml file first.")
        return False
    
    # Step 2: Setup directories
    print("\n📁 Setting up directories...")
    setup_directories()
    
    # Step 3: Download XTTS model
    if not args.skip_download:
        print("\n📥 Downloading XTTS v2 model...")
        if not download_xtts_model():
            print("❌ Model download failed")
            return False
    

    # Step 4: Convert config
    print("\n⚙️ Converting configuration...")
    gpt_config = convert_config_for_gpt_trainer(args.config, "gpt_config.yaml")
    
    # Step 5: Validate setup
    print("\n🔍 Validating setup...")
    if not validate_setup():
        print("❌ Setup validation failed")
        return False
    
    print("\n✅ Setup complete!")
    print("=" * 50)
    print("📊 Setup Summary:")
    print(f"   🎯 Original config: {args.config}")
    print(f"   ⚙️ GPT config: gpt_config.yaml")
    print(f"   🤖 XTTS model: ./models/xtts_v2/")
    print(f"   📁 Data: ./data/")
    print(f"   💾 Outputs: ./runs/")
    
    # Step 7: Run training if requested
    if args.run_training:
        print("\n🎯 Starting training...")
        try:
            from emotion_gpt_trainer import create_emotion_gpt_config, EmotionGPTTrainer
            from TTS.trainer import Trainer, TrainerArgs
            
            # Create config and model
            config = create_emotion_gpt_config(args.config)
            model = EmotionGPTTrainer(config)
            
            # Create trainer
            trainer = Trainer(
                TrainerArgs(
                    restore_path=None,
                    skip_train_epoch=False,
                    start_with_eval=False,
                    grad_accum_steps=1,
                ),
                config,
                output_path="./runs/emotion_gpt_trainer",
                model=model,
                train_samples=[],  # Dummy samples for now
                eval_samples=[],
            )
            
            print("🚀 Training started!")
            trainer.fit()
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n🎯 To start training, run:")
        print(f"   python emotion_gpt_trainer.py --config {args.config}")
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)