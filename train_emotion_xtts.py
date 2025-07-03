import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml
import numpy as np
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
import json
import tempfile
import torchaudio

from dataset import EmotionDataset, collate_fn
from emotion_model import EmotionXTTS


class EmotionXTTSTrainer:
    def __init__(self, config_path):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        
        # Set device
        self.device = torch.device(self.config['device'] if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize model
        print("Loading EmotionXTTS model...")
        
        # Check if we have custom XTTS model paths in config
        xtts_config_path = self.config.get('xtts', {}).get('config_path', None)
        xtts_checkpoint_path = self.config.get('xtts', {}).get('checkpoint_path', None)
        
        self.model = EmotionXTTS(
            config_path=xtts_config_path,
            checkpoint_path=xtts_checkpoint_path,
            num_emotions=len(self.config['data']['emotions'])
        ).to(self.device)
        
        # Unfreeze emotion adapter
        self.model.unfreeze_emotion_adapter()
        
        # Optionally unfreeze last GPT layers
        if self.config['model'].get('unfreeze_last_n_layers', 0) > 0:
            self.model.unfreeze_last_n_gpt_layers(self.config['model']['unfreeze_last_n_layers'])
            print(f"Unfroze last {self.config['model']['unfreeze_last_n_layers']} GPT layers")
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")
        
        # Initialize dataset and dataloader
        print("Loading dataset...")
        self.dataset = EmotionDataset(self.config)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=1,  # XTTS works best with batch_size=1
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4
        )
        
        # Initialize optimizer
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=float(self.config['training']['learning_rate']),
            weight_decay=float(self.config['optimization']['weight_decay'])
        )
        
        # Initialize scheduler
        total_steps = len(self.dataloader) * self.config['training']['num_epochs']
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=float(self.config['training']['learning_rate']) * 0.1
        )
        
        # Setup logging
        self.setup_logging()
        
        # Training state
        self.global_step = 0
        self.best_loss = float('inf')
        
    def setup_logging(self):
        """Setup logging directories."""
        # Create directories
        self.checkpoint_dir = Path(self.config['paths']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(self.config['paths']['log_dir'])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a simple log file
        self.log_file = self.log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.training_stats = []
        
        # Log initial config
        config_path = self.log_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        print(f"Logging to: {self.log_file}")
    
    def safe_squeeze_tensor(self, tensor, target_dims=1):
        """Safely squeeze tensor to target dimensions."""
        while tensor.dim() > target_dims and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        return tensor
    
    def create_temp_audio_file(self, audio_tensor, sample_rate=22050):
        """Create a temporary audio file from tensor."""
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_path = temp_file.name
        temp_file.close()
        
        try:
            # Ensure correct shape for saving [1, samples] or [channels, samples]
            if audio_tensor.dim() == 1:
                audio_to_save = audio_tensor.unsqueeze(0)
            elif audio_tensor.dim() == 3:
                audio_to_save = audio_tensor.squeeze(0)
            else:
                audio_to_save = audio_tensor
            
            # Move to CPU for saving
            audio_to_save = audio_to_save.cpu()
            
            # Save audio
            torchaudio.save(temp_path, audio_to_save, sample_rate)
            return temp_path
        except Exception as e:
            # Clean up on error
            try:
                os.unlink(temp_path)
            except:
                pass
            raise e
    
    def cleanup_temp_file(self, temp_path):
        """Safely clean up temporary file."""
        try:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        except Exception as e:
            print(f"Warning: Failed to clean up temporary file {temp_path}: {e}")
    
    def compute_emotion_adapter_loss(self, batch):
        """
        Compute loss focusing on emotion adapter training.
        This approach is more stable than full audio generation.
        """
        texts = batch['texts']
        speaker_refs = batch['speaker_refs'].to(self.device)
        emotion_ids = batch['emotion_ids'].to(self.device)
        
        total_loss = 0
        valid_samples = 0
        
        for i in range(len(texts)):
            temp_path = None
            try:
                print(f"\nProcessing sample {i}")
                print(f"Text: {texts[i][:50]}...")
                print(f"Speaker ref shape: {speaker_refs[i].shape}")
                print(f"Emotion ID: {emotion_ids[i]}")
                
                # Prepare speaker reference audio
                speaker_audio = self.safe_squeeze_tensor(speaker_refs[i], target_dims=1)
                print(f"Processed speaker audio shape: {speaker_audio.shape}")
                
                # Create temporary file for XTTS conditioning
                temp_path = self.create_temp_audio_file(speaker_audio)
                
                # Get original conditioning latents (no gradients)
                with torch.no_grad():
                    original_gpt, original_speaker = self.model.xtts.get_conditioning_latents(
                        audio_path=[temp_path]
                    )
                    
                    # Move to device and ensure proper shapes
                    original_gpt = original_gpt.to(self.device)
                    original_speaker = original_speaker.to(self.device)
                
                print(f"Original latent shapes - GPT: {original_gpt.shape}, Speaker: {original_speaker.shape}")
                
                # Get emotion-modified latents (with gradients for emotion adapter)
                emotion_gpt, emotion_speaker = self.model.get_conditioning_latents_with_emotion(
                    temp_path,
                    emotion_ids[i],
                    training=True
                )
                
                print(f"Emotion latent shapes - GPT: {emotion_gpt.shape}, Speaker: {emotion_speaker.shape}")
                
                # Compute emotion conditioning loss
                # We want the emotion adapter to produce meaningful but controlled modifications
                
                # 1. Ensure emotion modifications are in a reasonable range
                gpt_diff = emotion_gpt - original_gpt
                speaker_diff = emotion_speaker - original_speaker
                
                # Target: moderate but consistent emotion modifications
                target_gpt_magnitude = 0.1  # 10% of latent magnitude
                target_speaker_magnitude = 0.05  # 5% of speaker embedding magnitude
                
                gpt_magnitude = torch.norm(gpt_diff, dim=-1).mean()
                speaker_magnitude = torch.norm(speaker_diff, dim=-1).mean()
                
                # Loss to encourage meaningful emotion conditioning
                gpt_loss = F.mse_loss(gpt_magnitude, torch.tensor(target_gpt_magnitude, device=self.device))
                speaker_loss = F.mse_loss(speaker_magnitude, torch.tensor(target_speaker_magnitude, device=self.device))
                
                # 2. Regularization to prevent extreme modifications
                reg_loss = 0.01 * (torch.mean(gpt_diff ** 2) + torch.mean(speaker_diff ** 2))
                
                # 3. Consistency loss - same emotion should produce similar patterns
                # (This would require batching multiple samples with same emotion)
                
                # Combine losses
                sample_loss = gpt_loss + speaker_loss + reg_loss
                
                total_loss += sample_loss
                valid_samples += 1
                
                print(f"Sample {i} losses - GPT mag: {gpt_magnitude.item():.4f}, Speaker mag: {speaker_magnitude.item():.4f}")
                print(f"Sample {i} losses - GPT: {gpt_loss.item():.4f}, Speaker: {speaker_loss.item():.4f}, Reg: {reg_loss.item():.4f}")
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                import traceback
                traceback.print_exc()
                
            finally:
                # Always clean up temporary file
                if temp_path:
                    self.cleanup_temp_file(temp_path)
        
        if valid_samples == 0:
            # Return a zero loss tensor if no valid samples
            return torch.tensor(0.0, requires_grad=True, device=self.device)
        
        avg_loss = total_loss / valid_samples
        print(f"Batch average loss: {avg_loss.item():.4f}")
        return avg_loss
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0
        num_batches = 0
        progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Clear gradients
                self.optimizer.zero_grad()
                
                # Compute loss using emotion adapter approach
                loss = self.compute_emotion_adapter_loss(batch)
                
                # Check if loss is valid
                if not torch.isfinite(loss):
                    print(f"Warning: Non-finite loss at batch {batch_idx}, skipping...")
                    continue
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    float(self.config['optimization']['gradient_clip_norm'])
                )
                
                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                
                # Update tracking
                self.global_step += 1
                epoch_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': loss.item(),
                    'avg_loss': epoch_loss / num_batches,
                    'lr': self.scheduler.get_last_lr()[0]
                })
                
                # Log to file
                log_entry = {
                    'step': self.global_step,
                    'epoch': epoch,
                    'batch': batch_idx,
                    'loss': loss.item(),
                    'lr': self.scheduler.get_last_lr()[0],
                    'timestamp': datetime.now().isoformat()
                }
                self.training_stats.append(log_entry)
                
                # Save logs periodically
                if self.global_step % 100 == 0:
                    with open(self.log_file, 'w') as f:
                        json.dump(self.training_stats, f, indent=2)
                
                # Save checkpoint
                if self.global_step % self.config['training']['checkpoint_every'] == 0:
                    self.save_checkpoint(epoch)
                    
            except Exception as e:
                print(f"Error in training step {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                # Clear gradients and continue
                self.optimizer.zero_grad()
                continue
        
        return epoch_loss / max(num_batches, 1)
    
    def save_checkpoint(self, epoch):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'best_loss': self.best_loss
        }
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{self.global_step}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint at step {self.global_step}")
        
        # Save best model
        if hasattr(self, 'current_loss') and self.current_loss < self.best_loss:
            self.best_loss = self.current_loss
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"Saved best model with loss: {self.best_loss:.4f}")
    
    def train(self):
        """Main training loop."""
        print("Starting training...")
        print(f"Total epochs: {self.config['training']['num_epochs']}")
        print(f"Total batches per epoch: {len(self.dataloader)}")
        
        for epoch in range(int(self.config['training']['num_epochs'])):
            print(f"\n--- Epoch {epoch + 1}/{self.config['training']['num_epochs']} ---")
            
            # Train epoch
            epoch_loss = self.train_epoch(epoch)
            self.current_loss = epoch_loss
            
            print(f"Epoch {epoch + 1} completed - Average Loss: {epoch_loss:.4f}")
            
            # Save checkpoint after each epoch
            self.save_checkpoint(epoch)
            
            # Early stopping check (optional)
            if epoch_loss < 0.001:  # Very low loss
                print(f"Loss is very low ({epoch_loss:.6f}), consider stopping training")
        
        print("\nTraining completed!")
        
        # Save final training stats
        with open(self.log_file, 'w') as f:
            json.dump(self.training_stats, f, indent=2)
        
        print(f"Training logs saved to: {self.log_file}")
        print(f"Final model saved to: {self.checkpoint_dir}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train EmotionXTTS model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    # Initialize trainer
    trainer = EmotionXTTSTrainer(args.config)
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming training from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=trainer.device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.global_step = checkpoint['global_step']
        trainer.best_loss = checkpoint['best_loss']
        print(f"Resumed from step {trainer.global_step}")
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()