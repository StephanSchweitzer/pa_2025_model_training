#!/usr/bin/env python3
"""
Simple script to run emotion XTTS training with GPTTrainer.

SPYDER USAGE:
Just modify the configuration variables below and run this script (F5 in Spyder).
No command line arguments needed!

TERMINAL USAGE (optional):
python run_emotion_training.py --config config.yaml --epochs 50
"""

import os
import sys
import torch
import yaml
import argparse
from pathlib import Path

# =============================================================================
# 🔧 CONFIGURATION - EDIT THESE VARIABLES FOR YOUR SETUP
# =============================================================================

# Path to your emotion config file
CONFIG_PATH = "config.yaml"

# Number of training epochs (None = use config value)
EPOCHS = None

# Path to checkpoint to resume from (None = start fresh)
RESUME_FROM = None

# Output directory for training
OUTPUT_DIR = "./runs"

# Whether to run setup before training (if not already done)
RUN_SETUP_FIRST = False

# =============================================================================


def check_requirements():
    """Check if all required files and dependencies exist."""
    
    print("🔍 Checking requirements...")
    
    # Check if core files exist
    required_files = [
        "new_emotion_model.py",
        "utils.py",
        "config.yaml"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    # Check if TTS is installed
    try:
        import TTS
        print("✅ TTS library found")
    except ImportError:
        print("❌ TTS library not found. Install with: pip install TTS")
        return False
    
    # Check if XTTS model exists
    model_files = [
        "./models/xtts_v2/config.json",
        "./models/xtts_v2/model.pth",
        "./models/xtts_v2/vocab.json"
    ]
    
    missing_model_files = []
    for file_path in model_files:
        if not os.path.exists(file_path):
            missing_model_files.append(file_path)
    
    if missing_model_files:
        print(f"⚠️ Missing XTTS model files: {missing_model_files}")
        print("Run setup first: python setup_gpt_training.py")
        return False
    
    print("✅ All requirements satisfied")
    return True


def run_training(config_path: str = "config.yaml", 
                epochs: int = None,
                resume_from: str = None,
                output_dir: str = "./runs"):
    """Run the emotion XTTS training."""
    
    print("🚀 Starting Emotion XTTS Training with GPTTrainer")
    print("=" * 60)
    
    try:
        # Import the trainer
        from emotion_gpt_trainer import create_emotion_gpt_config, EmotionGPTTrainer
        from TTS.trainer import Trainer, TrainerArgs
        
        # Load and modify config if needed
        with open(config_path, 'r') as f:
            emotion_config = yaml.safe_load(f)
        
        if epochs is not None:
            emotion_config['training']['num_epochs'] = epochs
            print(f"🔧 Override epochs: {epochs}")
        
        # Save modified config if needed
        if epochs is not None:
            temp_config_path = "temp_config.yaml"
            with open(temp_config_path, 'w') as f:
                yaml.dump(emotion_config, f)
            config_path = temp_config_path
        
        # Create GPTTrainer config
        print("⚙️ Creating training configuration...")
        config = create_emotion_gpt_config(config_path)
        
        # Initialize model
        print("🤖 Initializing EmotionGPTTrainer...")
        model = EmotionGPTTrainer(config)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create trainer arguments
        trainer_args = TrainerArgs(
            restore_path=resume_from,
            skip_train_epoch=False,
            start_with_eval=False,
            grad_accum_steps=1,
        )
        
        # For now, we'll use empty training samples since we're focusing on emotion adapter training
        # In a real scenario, you'd load your actual dataset here
        train_samples = []
        eval_samples = []
        
        print("📚 Training samples: Using emotion adapter training mode")
        print(f"💾 Output directory: {output_path.absolute()}")
        
        # Create trainer
        print("🎯 Creating trainer...")
        trainer = Trainer(
            trainer_args,
            config,
            output_path=str(output_path),
            model=model,
            train_samples=train_samples,
            eval_samples=eval_samples,
        )
        
        # Print training info
        print("\n📊 Training Configuration:")
        print(f"   🎯 Model: EmotionXTTS with GPTTrainer")
        print(f"   📈 Epochs: {config.epochs}")
        print(f"   🎓 Learning Rate: {config.lr}")
        print(f"   📦 Batch Size: {config.batch_size}")
        print(f"   🔧 Emotion Adapter: Trainable")
        print(f"   🔒 Base XTTS: Frozen")
        
        if hasattr(model, 'emotion_config'):
            unfreeze_layers = model.emotion_config.get('model', {}).get('unfreeze_last_n_layers', 0)
            if unfreeze_layers > 0:
                print(f"   🔓 GPT Layers: Last {unfreeze_layers} unfrozen")
        
        if resume_from:
            print(f"   🔄 Resume from: {resume_from}")
        
        print(f"\n🎬 Starting training...")
        print("   Monitor progress in the output directory")
        print("   Use Ctrl+C to stop training")
        
        # Start training
        trainer.fit()
        
        print("\n✅ Training completed successfully!")
        print(f"📁 Checkpoints saved in: {output_path}")
        print(f"📊 Logs available in: {output_path}")
        
        # Clean up temporary config if created
        if epochs is not None and os.path.exists("temp_config.yaml"):
            os.remove("temp_config.yaml")
        
        return True
        
    except KeyboardInterrupt:
        print("\n⏹️ Training stopped by user")
        return True
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_training_for_spyder():
    """
    Main training function designed to run directly in Spyder.
    Uses the configuration variables defined at the top of this file.
    """
    
    print("🚀 Starting Emotion XTTS Training with GPTTrainer")
    print("=" * 60)
    print(f"📄 Using config: {CONFIG_PATH}")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    if EPOCHS is not None:
        print(f"📈 Epochs override: {EPOCHS}")
    if RESUME_FROM is not None:
        print(f"🔄 Resume from: {RESUME_FROM}")
    
    # Run setup first if requested
    if RUN_SETUP_FIRST:
        print("\n🔧 Running setup first...")
        try:
            from setup_gpt_training import setup_for_spyder
            setup_success = setup_for_spyder()
            if not setup_success:
                print("❌ Setup failed")
                return False
        except Exception as e:
            print(f"❌ Setup failed: {e}")
            return False
    
    # Check requirements
    if not check_requirements():
        print("\n💡 To fix missing requirements:")
        print("   1. Make sure you have config.yaml, new_emotion_model.py, and utils.py")
        print("   2. Install TTS: pip install TTS")
        print("   3. Run setup_gpt_training.py first")
        return False
    
    # Run training
    success = run_training(
        config_path=CONFIG_PATH,
        epochs=EPOCHS,
        resume_from=RESUME_FROM,
        output_dir=OUTPUT_DIR
    )
    
    if success:
        print("\n🎉 Training completed!")
        print("📊 Check training logs in the output directory")
        print("🧪 You can now test your trained model")
    else:
        print("\n❌ Training failed!")
        print("💡 Check the error messages above for troubleshooting")
    
    return success


def main():
    """
    Main function that works both in Spyder and command line.
    In Spyder: uses configuration variables at top of file
    In terminal: uses command line arguments
    """
    
    # Check if running with command line arguments
    if len(sys.argv) > 1:
        # Command line mode
        return main_with_args()
    else:
        # Spyder mode - use configuration variables
        return run_training_for_spyder()


def main_with_args():
    """Command line version with argparse."""
    parser = argparse.ArgumentParser(description="Run Emotion XTTS Training")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Path to emotion config file")
    parser.add_argument("--epochs", type=int, default=None,
                       help="Number of training epochs (overrides config)")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--output_dir", type=str, default="./runs",
                       help="Output directory for training")
    parser.add_argument("--setup", action="store_true",
                       help="Run setup before training")
    parser.add_argument("--check_only", action="store_true",
                       help="Only check requirements, don't train")
    
    args = parser.parse_args()
    
    # Run setup if requested
    if args.setup:
        print("🔧 Running setup first...")
        try:
            from setup_gpt_training import setup_for_spyder
            setup_success = setup_for_spyder()
            if not setup_success:
                print("❌ Setup failed")
                return False
        except Exception as e:
            print(f"❌ Setup failed: {e}")
            return False
    
    # Check requirements
    if not check_requirements():
        print("\n💡 To fix missing requirements:")
        print("   1. Run: python setup_gpt_training.py")
        print("   2. Or install TTS: pip install TTS")
        print("   3. Make sure you have new_emotion_model.py and utils.py")
        return False
    
    # If only checking, stop here
    if args.check_only:
        print("✅ Requirements check complete!")
        return True
    
    # Run training
    success = run_training(
        config_path=args.config,
        epochs=args.epochs,
        resume_from=args.resume,
        output_dir=args.output_dir
    )
    
    if success:
        print("\n🎉 All done!")
        print("\nNext steps:")
        print("   📊 Check training logs in the output directory")
        print("   🧪 Test your trained model")
        print("   🔧 Adjust config and retrain if needed")
    else:
        print("\n💡 Troubleshooting:")
        print("   🔍 Check the error messages above")
        print("   📋 Verify your config.yaml is correct")
        print("   🤖 Make sure XTTS model files are downloaded")
        print("   📦 Ensure all dependencies are installed")
    
    return success
        print("   🤖 Make sure XTTS model files are downloaded")
        print("   📦 Ensure all dependencies are installed")


# =============================================================================
# 🚀 SPYDER EXECUTION - This runs when you hit F5 in Spyder
# =============================================================================

if __name__ == "__main__":
    # Detect execution mode
    if len(sys.argv) > 1:
        print("🖥️ Running in command line mode")
        success = main_with_args()
    else:
        print("🔬 Running in Spyder/Direct execution mode")
        print("🔧 Using configuration variables defined at the top of this file")
        print()
        success = run_training_for_spyder()
    
    if not success:
        print("❌ Execution failed!")
    else:
        print("✅ Execution completed!")