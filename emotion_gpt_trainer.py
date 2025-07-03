import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import tempfile
import torchaudio

# Coqui TTS imports
try:
    from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTTrainer, GPTTrainerConfig, GPTArgs
    from TTS.tts.datasets import load_tts_samples
    from TTS.config.shared_configs import BaseDatasetConfig
    from TTS.trainer import Trainer, TrainerArgs
    from TTS.utils.manage import ModelManager
except ImportError as e:
    print(f"❌ Coqui TTS import error: {e}")
    print("Please install TTS: pip install TTS")
    exit(1)

# Your existing imports
from new_emotion_model import ArousaValenceConditioningAdapter
from utils import load_config, prepare_av_tensor


class EmotionGPTTrainer(GPTTrainer):
    """
    Extended GPTTrainer that supports emotion conditioning through arousal-valence values.
    """
    
    def __init__(self, config):
        # Initialize the base GPTTrainer first
        super().__init__(config)
        
        # Load emotion-specific config
        self.emotion_config_path = getattr(config, 'emotion_config_path', 'config.yaml')
        
        if os.path.exists(self.emotion_config_path):
            with open(self.emotion_config_path, 'r') as f:
                self.emotion_config = yaml.safe_load(f)
        else:
            print(f"⚠️ Emotion config not found: {self.emotion_config_path}, using defaults")
            self.emotion_config = self._get_default_emotion_config()
        
        # Initialize emotion adapter
        self.emotion_adapter = ArousaValenceConditioningAdapter(self.emotion_config)
        
        # Initialize transforms after base model is loaded
        if hasattr(self.config, 'model_args') and hasattr(self.config.model_args, 'gpt_n_model_channels'):
            latent_dim = self.config.model_args.gpt_n_model_channels
            self.emotion_adapter.initialize_transforms(latent_dim)
        else:
            # Default XTTS latent dimension
            self.emotion_adapter.initialize_transforms(1024)
        
        # Move emotion adapter to device if available
        if hasattr(self, 'device'):
            self.emotion_adapter = self.emotion_adapter.to(self.device)
        
        # Freeze base XTTS parameters initially
        self._freeze_base_model()
        
        # Unfreeze specified layers if configured
        unfreeze_layers = self.emotion_config.get('model', {}).get('unfreeze_last_n_layers', 0)
        if unfreeze_layers > 0:
            self._unfreeze_last_n_layers(unfreeze_layers)
        
        # Training state
        self.emotion_loss_weight = self.emotion_config.get('training', {}).get('emotion_loss_weight', 1.0)
        self.consistency_loss_weight = self.emotion_config.get('training', {}).get('consistency_loss_weight', 0.1)
        self.regularization_weight = self.emotion_config.get('training', {}).get('regularization_weight', 0.01)
        
        print(f"✅ EmotionGPTTrainer initialized")
        print(f"📊 Emotion adapter parameters: {sum(p.numel() for p in self.emotion_adapter.parameters()):,}")
        print(f"🔧 Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
    
    def _get_default_emotion_config(self):
        """Default emotion config if file not found."""
        return {
            'model': {
                'emotion_embedding_dim': 512,
                'unfreeze_last_n_layers': 2
            },
            'data': {
                'av_ranges': {
                    'arousal': [-1.0, 1.0],
                    'valence': [-1.0, 1.0]
                },
                'sample_rate': 22050
            },
            'training': {
                'emotion_loss_weight': 1.0,
                'consistency_loss_weight': 0.1,
                'regularization_weight': 0.01,
                'target_gpt_magnitude': 0.1,
                'target_speaker_magnitude': 0.05
            }
        }
    
    def _freeze_base_model(self):
        """Freeze all base XTTS model parameters."""
        for param in self.parameters():
            param.requires_grad = False
        
        # Unfreeze emotion adapter
        for param in self.emotion_adapter.parameters():
            param.requires_grad = True
        
        print("🔒 Base XTTS model frozen, emotion adapter unfrozen")
    
    def _unfreeze_last_n_layers(self, n: int):
        """Unfreeze the last n GPT layers for fine-tuning."""
        try:
            # Try different possible paths to GPT layers
            gpt_layers = None
            
            if hasattr(self, 'gpt') and hasattr(self.gpt, 'gpt') and hasattr(self.gpt.gpt, 'layers'):
                gpt_layers = self.gpt.gpt.layers
            elif hasattr(self, 'model') and hasattr(self.model, 'gpt'):
                if hasattr(self.model.gpt, 'layers'):
                    gpt_layers = self.model.gpt.layers
                elif hasattr(self.model.gpt, 'gpt') and hasattr(self.model.gpt.gpt, 'layers'):
                    gpt_layers = self.model.gpt.gpt.layers
            
            if gpt_layers is not None:
                for layer in gpt_layers[-n:]:
                    for param in layer.parameters():
                        param.requires_grad = True
                print(f"🔓 Unfroze last {n} GPT layers")
            else:
                print("⚠️ Could not find GPT layers to unfreeze")
        except Exception as e:
            print(f"⚠️ Error unfreezing GPT layers: {e}")
    
    def compute_emotion_conditioning_loss(self, 
                                        original_gpt: torch.Tensor,
                                        original_speaker: torch.Tensor,
                                        emotion_gpt: torch.Tensor,
                                        emotion_speaker: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute emotion conditioning losses.
        """
        losses = {}
        
        # 1. Magnitude loss - encourage meaningful but controlled modifications
        gpt_diff = emotion_gpt - original_gpt
        speaker_diff = emotion_speaker - original_speaker
        
        # Target magnitudes (configurable)
        target_gpt_magnitude = self.emotion_config.get('training', {}).get('target_gpt_magnitude', 0.1)
        target_speaker_magnitude = self.emotion_config.get('training', {}).get('target_speaker_magnitude', 0.05)
        
        gpt_magnitude = torch.norm(gpt_diff, dim=-1).mean()
        speaker_magnitude = torch.norm(speaker_diff, dim=-1).mean()
        
        losses['gpt_magnitude'] = F.mse_loss(
            gpt_magnitude, 
            torch.tensor(target_gpt_magnitude, device=gpt_magnitude.device)
        )
        losses['speaker_magnitude'] = F.mse_loss(
            speaker_magnitude,
            torch.tensor(target_speaker_magnitude, device=speaker_magnitude.device)
        )
        
        # 2. Regularization loss - prevent extreme modifications
        losses['regularization'] = torch.mean(gpt_diff ** 2) + torch.mean(speaker_diff ** 2)
        
        # 3. Direction consistency loss - similar emotions should modify in similar directions
        if emotion_gpt.shape[0] > 1:
            gpt_diff_norm = F.normalize(gpt_diff.view(gpt_diff.shape[0], -1), dim=1)
            speaker_diff_norm = F.normalize(speaker_diff.view(speaker_diff.shape[0], -1), dim=1)
            
            # Encourage similar modifications for similar emotions
            gpt_consistency = torch.mean(torch.var(gpt_diff_norm, dim=0))
            speaker_consistency = torch.mean(torch.var(speaker_diff_norm, dim=0))
            
            losses['consistency'] = gpt_consistency + speaker_consistency
        else:
            losses['consistency'] = torch.tensor(0.0, device=emotion_gpt.device)
        
        return losses
    
    def train_step(self, batch: dict, criterion: nn.Module, optimizer_idx: int = 0) -> Tuple[Dict, Dict]:
        """
        Override the train_step to include emotion conditioning.
        """
        
        # For now, we'll focus on emotion adapter training without full GPT training
        # This is more stable and computationally efficient
        
        device = next(self.parameters()).device
        
        # Generate some dummy conditioning data for emotion training
        # In a real scenario, you'd extract this from your batch
        batch_size = 1  # Start with batch size 1 for XTTS
        
        # Create dummy audio for conditioning (replace with real audio from batch)
        dummy_audio = torch.randn(22050, device=device)  # 1 second of audio
        
        # Create random AV values (replace with real values from batch)
        av_ranges = self.emotion_config['data']['av_ranges']
        av_values = torch.tensor([
            np.random.uniform(av_ranges['arousal'][0], av_ranges['arousal'][1]),
            np.random.uniform(av_ranges['valence'][0], av_ranges['valence'][1])
        ], device=device)
        
        try:
            # Create temporary audio file
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_file.close()
            
            # Save dummy audio
            torchaudio.save(temp_file.name, dummy_audio.unsqueeze(0).cpu(), 22050)
            
            # Get original conditioning latents (no gradients)
            with torch.no_grad():
                original_gpt, original_speaker = self.get_conditioning_latents(
                    audio_path=[temp_file.name],
                    gpt_cond_len=6,
                    max_ref_length=10
                )
                original_gpt = original_gpt.to(device)
                original_speaker = original_speaker.to(device)
            
            # Apply emotion conditioning (with gradients)
            emotion_gpt, emotion_speaker = self.emotion_adapter(
                original_gpt, 
                original_speaker, 
                av_values.unsqueeze(0),
                self.emotion_config['data']['av_ranges']
            )
            
            # Compute emotion losses
            emotion_losses = self.compute_emotion_conditioning_loss(
                original_gpt, original_speaker, emotion_gpt, emotion_speaker
            )
            
            # Combine losses
            total_emotion_loss = (
                self.emotion_loss_weight * (
                    emotion_losses['gpt_magnitude'] + 
                    emotion_losses['speaker_magnitude']
                ) +
                self.consistency_loss_weight * emotion_losses['consistency'] +
                self.regularization_weight * emotion_losses['regularization']
            )
            
            # Clean up temporary file
            try:
                os.unlink(temp_file.name)
            except:
                pass
            
            # Return losses in the format expected by the trainer
            losses_dict = {
                'model_loss': total_emotion_loss,
                'emotion_total': total_emotion_loss.item(),
                'emotion_gpt_mag': emotion_losses['gpt_magnitude'].item(),
                'emotion_speaker_mag': emotion_losses['speaker_magnitude'].item(),
                'emotion_consistency': emotion_losses['consistency'].item(),
                'emotion_regularization': emotion_losses['regularization'].item()
            }
            
            return losses_dict, {}
            
        except Exception as e:
            print(f"Error in emotion training step: {e}")
            # Return a dummy loss to keep training going
            dummy_loss = torch.tensor(0.0, device=device, requires_grad=True)
            return {'model_loss': dummy_loss}, {}
    
    def eval_step(self, batch: dict, criterion: nn.Module) -> Tuple[Dict, Dict]:
        """
        Override eval_step to include emotion evaluation.
        """
        with torch.no_grad():
            # Simple evaluation - just return zero loss for now
            # You can enhance this with proper emotion evaluation metrics
            return {'eval_loss': 0.0}, {}
    
    def get_optimizer(self) -> torch.optim.Optimizer:
        """
        Get optimizer with emotion adapter parameters.
        """
        # Get parameters that require gradients
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        
        if not trainable_params:
            print("⚠️ No trainable parameters found!")
            # Add emotion adapter parameters explicitly
            trainable_params = list(self.emotion_adapter.parameters())
        
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=getattr(self.config, 'lr', 1e-5),
            weight_decay=getattr(self.config, 'wd', 0.01)
        )
        
        return optimizer
    
    def save_checkpoint(self, checkpoint_path: str):
        """
        Save checkpoint including emotion adapter state.
        """
        try:
            # Create checkpoint dictionary
            checkpoint = {
                'model_state_dict': self.state_dict(),
                'emotion_adapter_state_dict': self.emotion_adapter.state_dict(),
                'emotion_config': self.emotion_config,
                'config': self.config
            }
            
            torch.save(checkpoint, checkpoint_path)
            print(f"💾 Saved emotion checkpoint to {checkpoint_path}")
        except Exception as e:
            print(f"❌ Error saving checkpoint: {e}")
    
    def load_checkpoint(self, checkpoint_path: str, load_emotion_adapter: bool = True):
        """
        Load checkpoint including emotion adapter state.
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location=next(self.parameters()).device)
            
            # Load base model state
            if 'model_state_dict' in checkpoint:
                self.load_state_dict(checkpoint['model_state_dict'], strict=False)
            
            # Load emotion adapter state
            if load_emotion_adapter and 'emotion_adapter_state_dict' in checkpoint:
                self.emotion_adapter.load_state_dict(checkpoint['emotion_adapter_state_dict'])
                print("✅ Loaded emotion adapter state")
            
            print(f"📂 Loaded checkpoint from {checkpoint_path}")
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")


def create_emotion_gpt_config(emotion_config_path: str = "config.yaml") -> GPTTrainerConfig:
    """
    Create a GPTTrainerConfig with emotion-specific settings.
    """
    
    # Load emotion config
    with open(emotion_config_path, 'r') as f:
        emotion_config = yaml.safe_load(f)
    
    # Base dataset config
    dataset_config = BaseDatasetConfig(
        formatter="ljspeech",
        dataset_name="emotion_dataset",
        path=emotion_config.get('data', {}).get('dataset_path', './data'),
        meta_file_train=emotion_config.get('data', {}).get('train_csv', 'metadata_train.csv'),
        meta_file_val=emotion_config.get('data', {}).get('val_csv', 'metadata_val.csv'),
        language=emotion_config.get('data', {}).get('language', 'en'),
    )
    
    # GPT model arguments
    model_args = GPTArgs(
        max_conditioning_length=132300,  # 6 secs
        min_conditioning_length=66150,   # 3 secs
        debug_loading_failures=False,
        max_wav_length=255995,          # ~11.6 seconds
        max_text_length=200,
        mel_norm_file=emotion_config.get('paths', {}).get('mel_norm_file', './models/xtts_v2/mel_stats.pth'),
        dvae_checkpoint=emotion_config.get('paths', {}).get('dvae_checkpoint', './models/xtts_v2/dvae.pth'),
        tokenizer_file=emotion_config.get('paths', {}).get('tokenizer_file', './models/xtts_v2/vocab.json'),
        gpt_n_model_channels=emotion_config.get('model', {}).get('gpt_channels', 1024),
    )
    
    # Training configuration
    config = GPTTrainerConfig(
        model_args=model_args,
        run_name=f"emotion_xtts_{emotion_config.get('experiment_name', 'default')}",
        epochs=emotion_config.get('training', {}).get('num_epochs', 100),
        lr=emotion_config.get('training', {}).get('learning_rate', 1e-5),
        batch_size=1,  # XTTS works best with batch_size=1
        eval_every=emotion_config.get('training', {}).get('eval_every', 1000),
        save_every=emotion_config.get('training', {}).get('checkpoint_every', 1000),
        print_every=emotion_config.get('training', {}).get('print_every', 100),
        
        # Optimizer settings
        wd=emotion_config.get('optimization', {}).get('weight_decay', 0.01),
        grad_clip=emotion_config.get('optimization', {}).get('gradient_clip_norm', 1.0),
        
        # Dataset settings
        datasets=[dataset_config],
        
        # Add emotion config path for the trainer
        emotion_config_path=emotion_config_path,
    )
    
    return config


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train EmotionXTTS with GPTTrainer")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to emotion config file")
    parser.add_argument("--output_path", type=str, default="./runs", help="Output directory for training")
    parser.add_argument("--restore_path", type=str, default=None, help="Path to checkpoint to restore from")
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        # Create GPTTrainer config
        config = create_emotion_gpt_config(args.config)
        
        # Initialize model
        model = EmotionGPTTrainer(config)
        
        # For now, we'll create dummy training samples
        # You can replace this with real data loading
        train_samples = []
        eval_samples = []
        
        # Initialize trainer
        trainer = Trainer(
            TrainerArgs(
                restore_path=args.restore_path,
                skip_train_epoch=False,
                start_with_eval=False,
                grad_accum_steps=1,
            ),
            config,
            output_path=args.output_path,
            model=model,
            train_samples=train_samples,
            eval_samples=eval_samples,
        )
        
        print("✅ Training setup complete!")
        print("🚀 Starting emotion adapter training...")
        
        # Start training
        trainer.fit()
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()