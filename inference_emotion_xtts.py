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
import wandb
from pathlib import Path

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
        self.model = EmotionXTTS(
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
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['optimization']['weight_decay']
        )
        
        # Initialize scheduler
        total_steps = len(self.dataloader) * self.config['training']['num_epochs']
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=self.config['training']['learning_rate'] * 0.1
        )
        
        # Setup logging
        self.setup_logging()
        
        # Training state
        self.global_step = 0
        self.best_loss = float('inf')
        
    def setup_logging(self):
        """Setup logging directories and wandb."""
        # Create directories
        self.checkpoint_dir = Path(self.config['paths']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(self.config['paths']['log_dir'])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize wandb
        wandb.init(
            project="emotion-xtts",
            config=self.config,
            name=f"emotion_xtts_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
    def compute_loss(self, batch):
        """Compute training loss with emotion conditioning."""
        texts = batch['texts']
        target_audios = batch['audios'].to(self.device)
        speaker_refs = batch['speaker_refs'].to(self.device)
        emotion_ids = batch['emotion_ids'].to(self.device)
        
        # For each sample in batch
        total_loss = 0
        for i in range(len(texts)):
            try:
                # Get emotion-conditioned generation
                with torch.cuda.amp.autocast(enabled=self.config['mixed_precision']):
                    # Save speaker reference to temp file (XTTS requires file path)
                    temp_ref_path = f"/tmp/speaker_ref_{i}.wav"
                    import torchaudio
                    torchaudio.save(
                        temp_ref_path,
                        speaker_refs[i].unsqueeze(0).cpu(),
                        self.config['data']['sample_rate']
                    )
                    
                    # Generate audio with emotion
                    output = self.model.inference_with_emotion(
                        text=texts[i],
                        language="en",
                        audio_path=temp_ref_path,
                        emotion_id=emotion_ids[i],
                        enable_text_splitting=False
                    )
                    
                    # Convert output to tensor
                    if isinstance(output, dict):
                        generated_audio = torch.tensor(output['wav']).to(self.device)
                    else:
                        generated_audio = torch.tensor(output).to(self.device)
                    
                    # Ensure same length for loss computation
                    target_audio = target_audios[i]
                    min_len = min(generated_audio.shape[0], target_audio.shape[0])
                    generated_audio = generated_audio[:min_len]
                    target_audio = target_audio[:min_len]
                    
                    # Reconstruction loss
                    recon_loss = F.l1_loss(generated_audio, target_audio)
                    
                    # Clean up temp file
                    os.remove(temp_ref_path)
                    
                    total_loss += recon_loss
                    
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
        
        return total_loss / len(texts)
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0
        progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Compute loss
            loss = self.compute_loss(batch)
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config['training']['gradient_accumulation_steps'] == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['optimization']['gradient_clip_norm']
                )
                
                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                # Logging
                self.global_step += 1
                epoch_loss += loss.item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': loss.item(),
                    'lr': self.scheduler.get_last_lr()[0]
                })
                
                # Log to wandb
                wandb.log({
                    'train/loss': loss.item(),
                    'train/lr': self.scheduler.get_last_lr()[0],
                    'train/epoch': epoch,
                    'train/step': self.global_step
                })
                
                # Save checkpoint
                if self.global_step % self.config['training']['checkpoint_every'] == 0:
                    self.save_checkpoint(epoch)
        
        return epoch_loss / len(self.dataloader)
    
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
        
        # Save best model
        if hasattr(self, 'current_loss') and self.current_loss < self.best_loss:
            self.best_loss = self.current_loss
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"Saved best model with loss: {self.best_loss:.4f}")
    
    def train(self):
        """Main training loop."""
        print("Starting training...")
        
        for epoch in range(self.config['training']['num_epochs']):
            # Train epoch
            epoch_loss = self.train_epoch(epoch)
            self.current_loss = epoch_loss
            
            print(f"Epoch {epoch} - Average Loss: {epoch_loss:.4f}")
            
            # Save checkpoint after each epoch
            self.save_checkpoint(epoch)
        
        print("Training completed!")
        wandb.finish()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train EmotionXTTS model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Initialize trainer
    trainer = EmotionXTTSTrainer(args.config)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()