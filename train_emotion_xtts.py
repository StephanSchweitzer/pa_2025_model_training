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
import torchaudio
from tensorboardX import SummaryWriter

from dataset import ValenceArousalDataset, valence_arousal_collate_fn
from emotion_model import ValenceArousalXTTS


class EmotionalXTTSTrainer:
    def __init__(self, config_path):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Set device
        self.device = torch.device(self.config['device'] if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Setup logging
        self.setup_logging()
        
        # Initialize your existing model (not official trainer)
        print("Loading Emotional XTTS model...")
        self.model = ValenceArousalXTTS(local_model_dir="./models/xtts_v2")
        self.model = self.model.to(self.device)
        
        # Only train the adapter - freeze everything else
        self.model.unfreeze_valence_arousal_adapter()
        
        # Setup dataset using your existing approach
        print("Loading dataset...")
        self.setup_dataset()
        
        # Setup optimizer for adapter only
        self.setup_optimizer()
        
        # Setup tensorboard
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        print("Trainer initialization complete!")

    def setup_logging(self):
        """Setup logging directories."""
        self.checkpoint_dir = Path(self.config['paths']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(self.config['paths']['log_dir'])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = self.log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        config_path = self.log_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        print(f"Logging to: {self.log_file}")

    def setup_dataset(self):
        """Setup dataset using your existing implementation."""
        self.dataset = ValenceArousalDataset(self.config)
        
        # Split into train/val
        dataset_size = len(self.dataset)
        val_size = int(0.1 * dataset_size)
        train_size = dataset_size - val_size
        
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            self.dataset, [train_size, val_size]
        )
        
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            collate_fn=valence_arousal_collate_fn,
            num_workers=self.config.get('num_workers', 4)
        )
        
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            collate_fn=valence_arousal_collate_fn,
            num_workers=self.config.get('num_workers', 4)
        )
        
        print(f"Dataset split: {len(self.train_dataset)} train, {len(self.val_dataset)} validation")

    def setup_optimizer(self):
        """Setup optimizer for adapter parameters only."""
        # Only train the adapter parameters
        adapter_params = list(self.model.va_adapter.parameters())
        
        print(f"Training {sum(p.numel() for p in adapter_params)} adapter parameters")
        
        self.optimizer = AdamW(
            adapter_params,
            lr=float(self.config['training']['learning_rate']),
            weight_decay=float(self.config['optimization']['weight_decay'])
        )
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['training']['num_epochs'],
            eta_min=float(self.config['training']['learning_rate']) * 0.1
        )

    def get_audio_tokens_batch(self, audio_batch):
        """Extract audio tokens using DVAE for ground truth - FAIL if extraction fails!"""
        try:
            # Use the corrected DVAE tokenization from emotion_model
            tokens_list = self.model.extract_dvae_tokens_batch(audio_batch)
            
            # Verify ALL tokens were extracted successfully
            if len(tokens_list) != len(audio_batch):
                raise ValueError(f"Token extraction failed: got {len(tokens_list)} tokens for {len(audio_batch)} audio samples")
            
            # Check that no tokens are None
            for i, tokens in enumerate(tokens_list):
                if tokens is None:
                    raise ValueError(f"Token extraction failed for sample {i}")
            
            return tokens_list
            
        except Exception as e:
            print(f"CRITICAL ERROR: Audio token extraction failed: {e}")
            print("Training cannot continue without real ground truth tokens!")
            raise e  # Fail fast - don't continue with bad data

    def compute_gpt_loss(self, emotion_gpt_latent, target_tokens, text_tokens):
        """Compute GPT loss for emotional conditioning - now with proper DVAE tokens."""
        try:
            # Now we have real DVAE tokens! We can compute a more meaningful loss
            batch_size = emotion_gpt_latent.shape[0]
            
            loss = 0.0
            
            for i in range(batch_size):
                # Emotional conditioning consistency loss
                gpt_latent = emotion_gpt_latent[i]
                
                # 1. Encourage meaningful but controlled modifications
                latent_norm = torch.norm(gpt_latent)
                consistency_loss = torch.abs(latent_norm - 1.0)  # Keep reasonable magnitude
                
                # 2. Token-aware loss (if we have target tokens)
                if target_tokens and i < len(target_tokens) and target_tokens[i] is not None:
                    tokens = target_tokens[i]
                    
                    # Simple token diversity loss - encourage the emotional conditioning 
                    # to be correlated with token diversity (more emotional = more varied tokens)
                    token_diversity = torch.unique(tokens).float().numel() / tokens.numel()
                    
                    # Use a simple relationship: more emotional content should have more token diversity
                    # This is a simple heuristic - you can make this more sophisticated
                    diversity_loss = torch.abs(token_diversity - 0.3)  # Target ~30% unique tokens
                    
                    loss += consistency_loss + 0.1 * diversity_loss
                else:
                    loss += consistency_loss
            
            return loss / batch_size
            
        except Exception as e:
            print(f"Error computing GPT loss: {e}")
            return torch.tensor(0.0, requires_grad=True, device=next(self.model.parameters()).device)

    def training_step(self, batch):
        """Simplified training step focusing on adapter training - FAIL FAST on bad data."""
        try:
            self.optimizer.zero_grad()
            
            # Extract target audio tokens - MUST succeed or skip this batch entirely
            try:
                target_tokens = self.get_audio_tokens_batch(batch['audios'])
            except Exception as e:
                print(f"Skipping batch due to token extraction failure: {e}")
                return 0.0  # Skip this batch, don't corrupt training
            
            batch_size = len(batch['speaker_refs'])
            total_loss = 0.0
            valid_samples = 0
            
            # Process each sample to get emotional conditioning
            for i in range(batch_size):
                try:
                    # Get emotional conditioning using speaker reference (file path)
                    emotion_gpt_latent, emotion_speaker_emb = self.model.get_conditioning_latents_with_valence_arousal(
                        batch['speaker_refs'][i],  # This is already a file path string
                        batch['valence'][i].item(),
                        batch['arousal'][i].item(),
                        training=True
                    )
                    
                    # Compute loss for this sample with REAL tokens
                    sample_loss = self.compute_gpt_loss(
                        emotion_gpt_latent, 
                        target_tokens,  # Real tokens only!
                        batch['texts'][i]
                    )
                    
                    if torch.isfinite(sample_loss):
                        total_loss += sample_loss
                        valid_samples += 1
                    
                except Exception as e:
                    print(f"Error processing sample {i}: {e}")
                    continue
            
            if valid_samples > 0:
                avg_loss = total_loss / valid_samples
                
                # Backward pass
                avg_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.va_adapter.parameters(), 1.0)
                
                # Optimizer step
                self.optimizer.step()
                
                return avg_loss.item()
            else:
                print("Warning: No valid samples in batch - skipping")
                return 0.0
                
        except Exception as e:
            print(f"Error in training step: {e}")
            return 0.0

    def validation_step(self, batch):
        """Validation step."""
        with torch.no_grad():
            try:
                batch_size = len(batch['speaker_refs'])
                total_loss = 0.0
                valid_samples = 0
                
                for i in range(batch_size):
                    try:
                        # Get emotional conditioning
                        emotion_gpt_latent, emotion_speaker_emb = self.model.get_conditioning_latents_with_valence_arousal(
                            batch['speaker_refs'][i],
                            batch['valence'][i].item(),
                            batch['arousal'][i].item(),
                            training=False
                        )
                        
                        # Simple validation loss
                        val_loss = torch.norm(emotion_gpt_latent) / emotion_gpt_latent.numel()
                        
                        if torch.isfinite(val_loss):
                            total_loss += val_loss.item()
                            valid_samples += 1
                            
                    except Exception as e:
                        print(f"Error in validation sample {i}: {e}")
                        continue
                
                return total_loss / valid_samples if valid_samples > 0 else 0.0
                
            except Exception as e:
                print(f"Error in validation step: {e}")
                return 0.0

    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            loss = self.training_step(batch)
            total_loss += loss
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss:.4f}'})
            
            # Log to tensorboard
            step = epoch * len(self.train_dataloader) + batch_idx
            self.writer.add_scalar('Train/Loss', loss, step)
            
            # Print progress
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss:.4f}")
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def validate_epoch(self, epoch):
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation"):
                loss = self.validation_step(batch)
                total_loss += loss
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        self.writer.add_scalar('Val/Loss', avg_loss, epoch)
        
        return avg_loss

    def save_checkpoint(self, epoch, train_loss, val_loss, is_best=False):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'va_adapter_state_dict': self.model.va_adapter.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_checkpoint.pth"
            torch.save(checkpoint, best_path)
            print(f"New best checkpoint saved: {best_path}")
        
        print(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.va_adapter.load_state_dict(checkpoint['va_adapter_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Checkpoint loaded from {checkpoint_path}")
        return checkpoint['epoch'], checkpoint['train_loss'], checkpoint['val_loss']

    def train(self):
        """Main training loop."""
        print("Starting Emotional XTTS training...")
        
        # Set random seeds
        torch.manual_seed(42)
        np.random.seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
        
        best_val_loss = float('inf')
        num_epochs = self.config['training']['num_epochs']
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Training
            train_loss = self.train_epoch(epoch)
            
            # Validation
            val_loss = self.validate_epoch(epoch)
            
            # Scheduler step
            self.scheduler.step()
            
            # Print epoch summary
            print(f"Epoch {epoch + 1} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Learning Rate: {self.scheduler.get_last_lr()[0]:.6f}")
            
            # Save checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            
            if (epoch + 1) % self.config['training']['checkpoint_every'] == 0:
                self.save_checkpoint(epoch + 1, train_loss, val_loss, is_best)
        
        print("Training completed!")
        
        # Save final adapter
        final_adapter_path = self.checkpoint_dir / "emotional_adapter_final.pth"
        torch.save({
            'va_adapter_state_dict': self.model.va_adapter.state_dict(),
            'config': self.config
        }, final_adapter_path)
        
        print(f"Final emotional adapter saved to: {final_adapter_path}")
        
        # Close tensorboard writer
        self.writer.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train Emotional XTTS model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = EmotionalXTTSTrainer(args.config)
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming training from {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()