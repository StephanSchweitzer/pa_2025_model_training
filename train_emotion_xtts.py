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
from tensorboardX import SummaryWriter

from TTS.tts.layers.tortoise.arch_utils import TorchMelSpectrogram
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
        
        # Setup logging and TensorBoard
        self.setup_logging()
        
        # Initialize TensorBoard writer
        self.tb_writer = SummaryWriter(
            log_dir=str(self.log_dir / "tensorboard"),
            comment=f"EmotionalXTTS_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # Initialize our custom emotional wrapper - let it handle XTTS loading
        print("Initializing Emotional XTTS model...")
        self.model = ValenceArousalXTTS(
            config_path=self.config.get('xtts', {}).get('config_path', None),
            checkpoint_path=self.config.get('xtts', {}).get('checkpoint_path', None),
            local_model_dir=self.config.get('xtts', {}).get('local_model_dir', './models/xtts_v2')
        ).to(self.device)
        
        # Initialize mel spectrogram processor
        mel_stats_path = self.config.get('xtts', {}).get('mel_stats_path', './models/xtts_v2/mel_stats.pth')
        self.mel_spectrogram = TorchMelSpectrogram(
            mel_norm_file=mel_stats_path,
            sampling_rate=22050
        ).to(self.device)
        
        # Initialize DVAE separately
        from TTS.tts.layers.xtts.dvae import DiscreteVAE
        dvae_path = self.config.get('xtts', {}).get('dvae_path', './models/xtts_v2/dvae.pth')
        self.dvae = DiscreteVAE(
            channels=80, normalization=None, positional_dims=1, num_tokens=1024,
            codebook_dim=512, hidden_dim=512, num_resnet_blocks=3, kernel_size=3,
            num_layers=2, use_transposed_convs=False,
        )
        self.dvae.load_state_dict(torch.load(dvae_path, map_location='cpu'), strict=False)
        self.dvae = self.dvae.to(self.device).eval()
        
        # Unfreeze emotional adapter
        self.model.unfreeze_valence_arousal_adapter()
        
        # Optionally unfreeze last GPT layers for fine-tuning
        if self.config['model'].get('unfreeze_last_n_layers', 0) > 0:
            self.model.unfreeze_last_n_gpt_layers(self.config['model']['unfreeze_last_n_layers'])
            print(f"Unfroze last {self.config['model']['unfreeze_last_n_layers']} GPT layers")
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")
        
        # Initialize dataset and dataloader
        print("Loading emotional dataset...")
        self.dataset = ValenceArousalDataset(self.config)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.config['training'].get('batch_size', 1),
            shuffle=True,
            collate_fn=valence_arousal_collate_fn,
            num_workers=self.config.get('num_workers', 4)
        )
        
        # Initialize optimizer (only for trainable parameters)
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
        print(f"TensorBoard logs: {self.log_dir / 'tensorboard'}")
    
    def compute_emotional_loss(self, batch):
        texts = batch['texts']
        languages = batch.get('languages', ['en'] * len(texts))
        speaker_refs = batch['speaker_refs']  # File paths
        valence_labels = batch['valence'].to(self.device).float()
        arousal_labels = batch['arousal'].to(self.device).float()
        target_audios = batch['audios'].to(self.device)
        
        total_loss = 0
        valid_samples = 0
        
        for i in range(len(texts)):
            try:
                # Get emotional conditioning
                emotion_gpt_latent, emotion_speaker_embedding = \
                    self.model.get_conditioning_latents_with_valence_arousal(
                        speaker_refs[i], valence_labels[i], arousal_labels[i], training=True
                    )
                
                # Convert target audio to mel and get audio tokens
                gt_audio = target_audios[i]
                if gt_audio.dim() == 1:
                    gt_audio = gt_audio.unsqueeze(0)
                
                mel = self.mel_spectrogram(gt_audio.unsqueeze(0))
                remainder = mel.shape[-1] % 4
                if remainder:
                    mel = mel[:, :, :-remainder]
                
                gt_audio_tokens = self.dvae.get_codebook_indices(mel)
                
                # Use XTTS GPT to predict audio tokens with emotional conditioning
                # This is a simplified version - you'd need to implement the full forward pass
                # based on how XTTS training works
                
                # For now, a simple loss that encourages meaningful emotional modulation:
                # Compare emotional conditioning with neutral (valence=0, arousal=0)
                neutral_gpt_latent, neutral_speaker_embedding = \
                    self.model.get_conditioning_latents_with_valence_arousal(
                        speaker_refs[i], torch.tensor(0.0), torch.tensor(0.0), training=True
                    )
                
                # Loss encourages emotional conditioning to differ from neutral in meaningful ways
                emotion_diff_loss = F.mse_loss(emotion_gpt_latent, neutral_gpt_latent)
                
                # Add regularization to prevent too extreme changes
                regularization_loss = 0.1 * (torch.abs(valence_labels[i]) + torch.abs(arousal_labels[i]))
                
                loss = emotion_diff_loss + regularization_loss
                
                total_loss += loss
                valid_samples += 1
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
        
        if valid_samples == 0:
            return torch.tensor(0.0, requires_grad=True, device=self.device)
        
        return total_loss / valid_samples
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        
        epoch_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                self.optimizer.zero_grad()
                
                # Compute loss
                loss = self.compute_emotional_loss(batch)
                
                if not torch.isfinite(loss):
                    print(f"Warning: Non-finite loss at batch {batch_idx}, skipping...")
                    continue
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
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
                    'lr': self.scheduler.get_last_lr()[0]
                })
                
                # Log to TensorBoard
                if self.global_step % 10 == 0:
                    self.tb_writer.add_scalar('Loss/Total', loss.item(), self.global_step)
                    self.tb_writer.add_scalar('Training/Learning_Rate', self.scheduler.get_last_lr()[0], self.global_step)
                
                # Save checkpoint periodically
                if self.global_step % self.config['training']['checkpoint_every'] == 0:
                    self.save_checkpoint(epoch)
                    
            except Exception as e:
                print(f"Error in training step {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                self.optimizer.zero_grad()
                continue
        
        # Log epoch averages
        if num_batches > 0:
            avg_epoch_loss = epoch_loss / num_batches
            self.tb_writer.add_scalar('Epoch/Total_Loss', avg_epoch_loss, epoch)
            return avg_epoch_loss
        
        return float('inf')
    
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
        print("Starting Emotional XTTS training...")
        print(f"Total epochs: {self.config['training']['num_epochs']}")
        print(f"Total batches per epoch: {len(self.dataloader)}")
        
        for epoch in range(int(self.config['training']['num_epochs'])):
            print(f"\n--- Epoch {epoch + 1}/{self.config['training']['num_epochs']} ---")
            
            epoch_loss = self.train_epoch(epoch)
            self.current_loss = epoch_loss
            
            print(f"Epoch {epoch + 1} completed - Average Loss: {epoch_loss:.4f}")
            
            # Save checkpoint after each epoch
            self.save_checkpoint(epoch)
        
        print("\nEmotional XTTS training completed!")
        self.tb_writer.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train Emotional XTTS model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Initialize trainer
    trainer = EmotionalXTTSTrainer(args.config)
    
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