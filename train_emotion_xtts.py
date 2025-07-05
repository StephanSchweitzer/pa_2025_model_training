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
import tempfile
from tensorboardX import SummaryWriter

from dataset import ValenceArousalDataset, cross_emotional_collate_fn
from emotion_model import ValenceArousalXTTS
from vad_analyzer import VADAnalyzer


class EmotionalXTTSTrainer:
    def __init__(self, config_path):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Set device consistently
        self.device = torch.device(self.config['device'] if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Setup logging
        self.setup_logging()
        
        # Initialize VAD analyzer
        print("Loading VAD analyzer...")
        self.vad_analyzer = VADAnalyzer(
            model_dir=self.config.get('vad_model_dir', 'vad_model'),
            auto_download=True,
            verbose=True
        )
        
        if not self.vad_analyzer.model_available:
            raise RuntimeError("VAD analyzer failed to initialize. Cannot train without emotional evaluation.")
        
        # Initialize emotional XTTS model and move to device in one step
        print("Loading Emotional XTTS model...")
        self.model = ValenceArousalXTTS(local_model_dir="./models/xtts_v2")
        self.model = self.model.to(self.device)  # This should handle all submodules
        
        # Verify all components are on the same device
        print(f"Model device: {next(self.model.parameters()).device}")
        print(f"Adapter device: {next(self.model.va_adapter.parameters()).device}")
        
        # Only train the adapter - freeze everything else
        self.model.unfreeze_valence_arousal_adapter()
        
        # Setup dataset using cross-emotional pairs
        print("Loading dataset...")
        self.setup_dataset()
        
        # Setup optimizer for adapter only
        self.setup_optimizer()
        
        # Setup tensorboard
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        # Generation parameters
        self.inference_kwargs = {
            "temperature": 0.75,
            "length_penalty": 1.0,
            "repetition_penalty": 1.1,
            "top_k": 50,
            "top_p": 0.8
        }
        
        # Training step counter for periodic VAD evaluation
        self.training_step_count = 0
        
        print("Trainer initialization complete!")

    def debug_tensor_devices(self, *tensors, names=None):
        """Debug helper to check tensor devices"""
        if names is None:
            names = [f"tensor_{i}" for i in range(len(tensors))]
        
        for tensor, name in zip(tensors, names):
            if torch.is_tensor(tensor):
                print(f"{name}: device={tensor.device}, shape={tensor.shape}")
            else:
                print(f"{name}: not a tensor, type={type(tensor)}")

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
        """Setup dataset using cross-emotional pairs."""
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
            collate_fn=cross_emotional_collate_fn,
            num_workers=self.config.get('num_workers', 2)  # Reduced for stability
        )
        
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            collate_fn=cross_emotional_collate_fn,
            num_workers=self.config.get('num_workers', 2)
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

    def save_temp_audio(self, audio_tensor, sample_rate=22050):
        """Save audio tensor to temporary file for VAD analysis."""
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_path = temp_file.name
        temp_file.close()
        
        try:
            # Convert numpy array to torch tensor if needed
            if isinstance(audio_tensor, np.ndarray):
                audio_tensor = torch.from_numpy(audio_tensor)
            
            # Handle failed generation (zeros)
            if torch.all(audio_tensor == 0) or torch.max(torch.abs(audio_tensor)) == 0:
                print("Warning: Generated audio is silence - inference likely failed")
                # Create minimal test tone instead of silence for VAD
                audio_tensor = 0.1 * torch.sin(2 * 3.14159 * 440 * torch.linspace(0, 1, sample_rate))
            
            # Ensure audio is properly shaped and on CPU for saving
            audio_tensor = audio_tensor.detach().cpu() if hasattr(audio_tensor, 'detach') else audio_tensor.cpu()
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            elif audio_tensor.dim() == 3:
                audio_tensor = audio_tensor.squeeze(0)
            elif audio_tensor.dim() == 2 and audio_tensor.shape[0] > 1:
                audio_tensor = audio_tensor.mean(dim=0, keepdim=True)
            
            # Ensure 2D shape [channels, samples]
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            
            # Save audio
            torchaudio.save(temp_path, audio_tensor, sample_rate)
            return temp_path
        except Exception as e:
            # Clean up on failure
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
            print(f"Warning: Could not delete temporary file {temp_path}: {e}")

    def generate_audio_sample(self, text, speaker_ref, target_valence, target_arousal):
        """Generate audio using emotional conditioning."""
        try:
            print(f"Generating: '{text[:50]}...' with V={target_valence.item():.3f}, A={target_arousal.item():.3f}")
            
            # Ensure target values are on correct device
            target_valence = target_valence.to(self.device)
            target_arousal = target_arousal.to(self.device)
            
            # Generate audio with emotional conditioning
            audio_output = self.model.inference_with_valence_arousal(
                text=text,
                language="en",
                audio_path=speaker_ref,
                valence=target_valence.item(),
                arousal=target_arousal.item(),
                **self.inference_kwargs
            )
            
            # Extract the actual audio tensor and handle both numpy and torch formats
            if isinstance(audio_output, dict) and 'wav' in audio_output:
                generated_audio = audio_output['wav']
            elif isinstance(audio_output, (torch.Tensor, np.ndarray)):
                generated_audio = audio_output
            else:
                raise ValueError(f"Unexpected audio output format: {type(audio_output)}")
            
            # Convert numpy array to torch tensor if needed
            if isinstance(generated_audio, np.ndarray):
                generated_audio = torch.from_numpy(generated_audio).to(self.device)
            elif isinstance(generated_audio, torch.Tensor):
                generated_audio = generated_audio.to(self.device)
            
            # Ensure proper tensor format
            if generated_audio.dim() == 3:
                generated_audio = generated_audio.squeeze(0)
            if generated_audio.dim() == 2:
                generated_audio = generated_audio.squeeze(0)
            
            # Check if generation was successful
            audio_max = torch.max(torch.abs(generated_audio))
            if audio_max == 0 or torch.isnan(audio_max):
                print("Warning: Generated audio is silence or contains NaN")
                # Return minimal test tone for VAD analysis
                sample_rate = 22050
                duration = 1.0
                t = torch.linspace(0, duration, int(sample_rate * duration), device=self.device)
                generated_audio = 0.1 * torch.sin(2 * 3.14159 * 440 * t)
            
            print(f"Generated audio: shape={generated_audio.shape}, max={audio_max:.4f}, device={generated_audio.device}")
            return generated_audio
            
        except Exception as e:
            print(f"Error generating audio: {e}")
            # Return test tone as fallback (on correct device)
            sample_rate = 22050
            duration = 1.0
            t = torch.linspace(0, duration, int(sample_rate * duration), device=self.device)
            return 0.1 * torch.sin(2 * 3.14159 * 440 * t)

    def compute_conditioning_loss(self, speaker_ref_path, target_valence, target_arousal):
        """Compute loss based on conditioning latents (differentiable training loss)."""
        try:
            device = self.device  # Use consistent device reference
            
            # Ensure targets are on correct device first
            target_valence = target_valence.to(device)
            target_arousal = target_arousal.to(device)
            
            # Get original conditioning latents and IMMEDIATELY move to device
            with torch.no_grad():  # Don't need gradients for original latents
                original_gpt_latent, original_speaker_emb = self.model.xtts.get_conditioning_latents([speaker_ref_path])
                original_gpt_latent = original_gpt_latent.to(device)
                original_speaker_emb = original_speaker_emb.to(device)
            
            # Get emotion-modified conditioning latents (this has gradients!)
            emotion_gpt_latent, emotion_speaker_emb = self.model.get_conditioning_latents_with_valence_arousal(
                speaker_ref_path, target_valence, target_arousal, training=True
            )
            
            # Ensure emotion tensors are on correct device (should already be, but double-check)
            emotion_gpt_latent = emotion_gpt_latent.to(device)
            emotion_speaker_emb = emotion_speaker_emb.to(device)
            
            # Debug tensor devices if needed
            # self.debug_tensor_devices(
            #     original_gpt_latent, emotion_gpt_latent, original_speaker_emb, emotion_speaker_emb,
            #     names=['orig_gpt', 'emo_gpt', 'orig_spk', 'emo_spk']
            # )
            
            # Now compute differences (all tensors guaranteed to be on same device)
            gpt_diff = torch.norm(emotion_gpt_latent - original_gpt_latent)
            speaker_diff = torch.norm(emotion_speaker_emb - original_speaker_emb)
            
            # 2. Emotional strength should correlate with modification magnitude
            emotion_strength = torch.sqrt(target_valence**2 + target_arousal**2)
            target_modification = 0.1 + 0.3 * emotion_strength  # Target modification range [0.1, 0.4]
            
            gpt_loss = F.smooth_l1_loss(gpt_diff, target_modification)
            speaker_loss = F.smooth_l1_loss(speaker_diff, target_modification * 0.5)  # Speaker should change less
            
            # 3. Regularization to prevent extreme values
            reg_loss = 0.01 * (torch.norm(emotion_gpt_latent) + torch.norm(emotion_speaker_emb))
            
            total_loss = gpt_loss + speaker_loss + reg_loss
            
            return total_loss, {
                'gpt_diff': gpt_diff.item(),
                'speaker_diff': speaker_diff.item(),
                'emotion_strength': emotion_strength.item(),
                'gpt_loss': gpt_loss.item(),
                'speaker_loss': speaker_loss.item(),
                'reg_loss': reg_loss.item()
            }
            
        except Exception as e:
            print(f"Error computing conditioning loss: {e}")
            print(f"Device being used: {device}")
            print(f"Original GPT device: {original_gpt_latent.device if 'original_gpt_latent' in locals() else 'undefined'}")
            print(f"Emotion GPT device: {emotion_gpt_latent.device if 'emotion_gpt_latent' in locals() else 'undefined'}")
            print(f"Target valence device: {target_valence.device}")
            print(f"Target arousal device: {target_arousal.device}")
            raise e

    def compute_vad_loss_for_validation(self, generated_audio, target_valence, target_arousal):
        """Compute VAD loss for validation (non-differentiable evaluation)."""
        temp_path = None
        try:
            # Ensure generated_audio is on CPU for saving
            generated_audio_cpu = generated_audio.detach().cpu()
            
            # Save generated audio to temporary file
            temp_path = self.save_temp_audio(generated_audio_cpu)
            
            # Analyze generated audio with VAD
            vad_result, status = self.vad_analyzer.extract(temp_path)
            
            if status != "success" or vad_result is None:
                print(f"VAD analysis failed: {status}")
                return None
            
            return {
                'pred_valence': vad_result['valence'],
                'pred_arousal': vad_result['arousal'],
                'target_valence': target_valence.item(),
                'target_arousal': target_arousal.item()
            }
            
        except Exception as e:
            print(f"Error computing VAD loss: {e}")
            return None
        
        finally:
            if temp_path:
                self.cleanup_temp_file(temp_path)

    def compute_speaker_similarity_loss(self, speaker_ref_path, generated_audio):
        """Compute speaker similarity between reference and generated audio."""
        temp_path = None
        try:
            device = self.device
            
            # Get reference speaker embedding
            ref_gpt_latent, ref_speaker_emb = self.model.xtts.get_conditioning_latents([speaker_ref_path])
            ref_speaker_emb = ref_speaker_emb.to(device)
            
            # Ensure generated_audio is on CPU for saving
            generated_audio_cpu = generated_audio.detach().cpu()
            
            # Save generated audio and get its speaker embedding
            temp_path = self.save_temp_audio(generated_audio_cpu)
            gen_gpt_latent, gen_speaker_emb = self.model.xtts.get_conditioning_latents([temp_path])
            gen_speaker_emb = gen_speaker_emb.to(device)
            
            # Compute cosine similarity between speaker embeddings
            ref_speaker_emb = ref_speaker_emb.flatten()
            gen_speaker_emb = gen_speaker_emb.flatten()
            
            similarity = F.cosine_similarity(ref_speaker_emb, gen_speaker_emb, dim=0)
            
            # Convert to loss (higher similarity = lower loss)
            speaker_loss = 1.0 - similarity
            
            return speaker_loss, similarity.item()
            
        except Exception as e:
            print(f"Error computing speaker similarity: {e}")
            return torch.tensor(0.5, requires_grad=True, device=self.device), 0.0
        
        finally:
            if temp_path:
                self.cleanup_temp_file(temp_path)

    def training_step(self, batch):
        """Training step with conditioning loss (differentiable) + periodic VAD evaluation."""
        try:
            self.optimizer.zero_grad()
            
            batch_size = len(batch['texts'])
            total_loss = 0.0
            valid_samples = 0
            
            conditioning_metrics = {
                'gpt_diff': 0.0,
                'speaker_diff': 0.0,
                'emotion_strength': 0.0
            }
            
            # VAD evaluation (only every N samples for monitoring)
            do_vad_eval = (self.training_step_count % 10 == 0)  # Every 10th step
            vad_metrics = {'valence_mae': 0.0, 'arousal_mae': 0.0} if do_vad_eval else {}
            
            # Process each sample in the batch
            for i in range(batch_size):
                try:
                    # Ensure target values are on correct device
                    target_valence = batch['target_valences'][i].to(self.device)
                    target_arousal = batch['target_arousals'][i].to(self.device)
                    
                    # Main training loss: differentiable conditioning loss
                    conditioning_loss, cond_info = self.compute_conditioning_loss(
                        batch['speaker_refs'][i],
                        target_valence,
                        target_arousal
                    )
                    
                    if torch.isfinite(conditioning_loss):
                        total_loss += conditioning_loss
                        valid_samples += 1
                        
                        # Accumulate conditioning metrics
                        for key in conditioning_metrics:
                            if key in cond_info:
                                conditioning_metrics[key] += cond_info[key]
                    
                    # Optional VAD evaluation for monitoring (no gradients)
                    if do_vad_eval:
                        with torch.no_grad():
                            try:
                                generated_audio = self.generate_audio_sample(
                                    text=batch['texts'][i],
                                    speaker_ref=batch['speaker_refs'][i],
                                    target_valence=target_valence,
                                    target_arousal=target_arousal
                                )
                                
                                # VAD evaluation (for monitoring only)
                                temp_path = self.save_temp_audio(generated_audio.detach().cpu())
                                vad_result, status = self.vad_analyzer.extract(temp_path)
                                self.cleanup_temp_file(temp_path)
                                
                                if status == "success" and vad_result:
                                    vad_metrics['valence_mae'] += abs(vad_result['valence'] - target_valence.item())
                                    vad_metrics['arousal_mae'] += abs(vad_result['arousal'] - target_arousal.item())
                                    
                            except Exception as e:
                                print(f"VAD evaluation failed for sample {i}: {e}")
                    
                except Exception as e:
                    print(f"Error processing sample {i}: {e}")
                    continue
            
            if valid_samples > 0:
                avg_loss = total_loss / valid_samples
                
                # Backward pass (only on conditioning loss)
                avg_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.va_adapter.parameters(), 1.0)
                
                # Optimizer step
                self.optimizer.step()
                
                # Average metrics
                for key in conditioning_metrics:
                    conditioning_metrics[key] /= valid_samples
                
                if do_vad_eval and vad_metrics:
                    for key in vad_metrics:
                        vad_metrics[key] /= valid_samples
                
                # Track training step count
                self.training_step_count = getattr(self, 'training_step_count', 0) + 1
                
                return {
                    'total_loss': avg_loss.item(),
                    'conditioning_loss': avg_loss.item(),
                    'gpt_diff': conditioning_metrics['gpt_diff'],
                    'speaker_diff': conditioning_metrics['speaker_diff'], 
                    'emotion_strength': conditioning_metrics['emotion_strength'],
                    'valence_mae': vad_metrics.get('valence_mae', 0.0),
                    'arousal_mae': vad_metrics.get('arousal_mae', 0.0),
                    'valid_samples': valid_samples,
                    'vad_evaluated': do_vad_eval
                }
            else:
                print("Warning: No valid samples in batch - skipping")
                return {
                    'total_loss': 0.0,
                    'conditioning_loss': 0.0,
                    'gpt_diff': 0.0,
                    'speaker_diff': 0.0,
                    'emotion_strength': 0.0,
                    'valence_mae': 0.0,
                    'arousal_mae': 0.0,
                    'valid_samples': 0,
                    'vad_evaluated': False
                }
                
        except Exception as e:
            print(f"Error in training step: {e}")
            return {
                'total_loss': 0.0,
                'conditioning_loss': 0.0,
                'gpt_diff': 0.0,
                'speaker_diff': 0.0, 
                'emotion_strength': 0.0,
                'valence_mae': 0.0,
                'arousal_mae': 0.0,
                'valid_samples': 0,
                'vad_evaluated': False
            }

    def validation_step(self, batch):
        """Validation step with full VAD evaluation."""
        with torch.no_grad():
            try:
                batch_size = len(batch['texts'])
                total_conditioning_loss = 0.0
                valid_samples = 0
                
                vad_metrics = {
                    'valence_mae': 0.0,
                    'arousal_mae': 0.0
                }
                
                for i in range(batch_size):
                    try:
                        # Ensure target values are on correct device
                        target_valence = batch['target_valences'][i].to(self.device)
                        target_arousal = batch['target_arousals'][i].to(self.device)
                        
                        # Conditioning loss evaluation
                        conditioning_loss, cond_info = self.compute_conditioning_loss(
                            batch['speaker_refs'][i],
                            target_valence,
                            target_arousal
                        )
                        
                        # Generate audio for VAD evaluation
                        generated_audio = self.generate_audio_sample(
                            text=batch['texts'][i],
                            speaker_ref=batch['speaker_refs'][i],
                            target_valence=target_valence,
                            target_arousal=target_arousal
                        )
                        
                        # VAD evaluation
                        temp_path = self.save_temp_audio(generated_audio.detach().cpu())
                        vad_result, status = self.vad_analyzer.extract(temp_path)
                        self.cleanup_temp_file(temp_path)
                        
                        if torch.isfinite(conditioning_loss) and status == "success" and vad_result:
                            total_conditioning_loss += conditioning_loss.item()
                            valid_samples += 1
                            
                            vad_metrics['valence_mae'] += abs(vad_result['valence'] - target_valence.item())
                            vad_metrics['arousal_mae'] += abs(vad_result['arousal'] - target_arousal.item())
                        
                    except Exception as e:
                        print(f"Error in validation sample {i}: {e}")
                        continue
                
                if valid_samples > 0:
                    return {
                        'conditioning_loss': total_conditioning_loss / valid_samples,
                        'valence_mae': vad_metrics['valence_mae'] / valid_samples,
                        'arousal_mae': vad_metrics['arousal_mae'] / valid_samples,
                        'valid_samples': valid_samples
                    }
                else:
                    return {
                        'conditioning_loss': 0.0,
                        'valence_mae': 0.0,
                        'arousal_mae': 0.0,
                        'valid_samples': 0
                    }
                
            except Exception as e:
                print(f"Error in validation step: {e}")
                return {
                    'conditioning_loss': 0.0,
                    'valence_mae': 0.0,
                    'arousal_mae': 0.0,
                    'valid_samples': 0
                }

    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        epoch_metrics = {
            'total_loss': 0.0,
            'conditioning_loss': 0.0,
            'gpt_diff': 0.0,
            'speaker_diff': 0.0,
            'emotion_strength': 0.0,
            'valence_mae': 0.0,
            'arousal_mae': 0.0,
            'valid_samples': 0
        }
        num_batches = 0
        
        pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Quick test mode support
            if self.config.get('quick_test', {}).get('enabled', False):
                max_steps = self.config['quick_test'].get('max_steps', float('inf'))
                if batch_idx >= max_steps:
                    print(f"Quick test mode: Stopping at step {batch_idx}")
                    break
            
            metrics = self.training_step(batch)
            
            # Accumulate metrics
            for key in epoch_metrics:
                epoch_metrics[key] += metrics[key]
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{metrics["total_loss"]:.4f}',
                'gpt_diff': f'{metrics["gpt_diff"]:.3f}',
                'spk_diff': f'{metrics["speaker_diff"]:.3f}',
                'v_mae': f'{metrics["valence_mae"]:.3f}' if metrics['vad_evaluated'] else 'N/A',
                'a_mae': f'{metrics["arousal_mae"]:.3f}' if metrics['vad_evaluated'] else 'N/A'
            })
            
            # Log to tensorboard
            step = epoch * len(self.train_dataloader) + batch_idx
            self.writer.add_scalar('Train/TotalLoss', metrics['total_loss'], step)
            self.writer.add_scalar('Train/ConditioningLoss', metrics['conditioning_loss'], step)
            self.writer.add_scalar('Train/GPTDiff', metrics['gpt_diff'], step)
            self.writer.add_scalar('Train/SpeakerDiff', metrics['speaker_diff'], step)
            
            if metrics['vad_evaluated']:
                self.writer.add_scalar('Train/ValenceMAE', metrics['valence_mae'], step)
                self.writer.add_scalar('Train/ArousalMAE', metrics['arousal_mae'], step)
            
            # Print progress
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}")
                print(f"  Total Loss: {metrics['total_loss']:.4f}")
                print(f"  Conditioning Loss: {metrics['conditioning_loss']:.4f}")
                print(f"  GPT Diff: {metrics['gpt_diff']:.3f}")
                print(f"  Speaker Diff: {metrics['speaker_diff']:.3f}")
                if metrics['vad_evaluated']:
                    print(f"  Valence MAE: {metrics['valence_mae']:.3f}")
                    print(f"  Arousal MAE: {metrics['arousal_mae']:.3f}")
        
        # Average metrics over epoch
        if num_batches > 0:
            for key in epoch_metrics:
                epoch_metrics[key] /= num_batches
        
        return epoch_metrics

    def validate_epoch(self, epoch):
        """Validate for one epoch."""
        self.model.eval()
        epoch_metrics = {
            'conditioning_loss': 0.0,
            'valence_mae': 0.0,
            'arousal_mae': 0.0,
            'valid_samples': 0
        }
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation"):
                metrics = self.validation_step(batch)
                
                # Accumulate metrics
                for key in epoch_metrics:
                    epoch_metrics[key] += metrics[key]
                num_batches += 1
        
        # Average metrics
        if num_batches > 0:
            for key in epoch_metrics:
                epoch_metrics[key] /= num_batches
        
        # Log to tensorboard
        self.writer.add_scalar('Val/ConditioningLoss', epoch_metrics['conditioning_loss'], epoch)
        self.writer.add_scalar('Val/ValenceMAE', epoch_metrics['valence_mae'], epoch)
        self.writer.add_scalar('Val/ArousalMAE', epoch_metrics['arousal_mae'], epoch)
        
        return epoch_metrics

    def save_checkpoint(self, epoch, train_metrics, val_metrics, is_best=False):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'va_adapter_state_dict': self.model.va_adapter.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
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
        return checkpoint['epoch'], checkpoint.get('train_metrics', {}), checkpoint.get('val_metrics', {})

    def train(self):
        """Main training loop."""
        print("Starting Emotional XTTS training with VAD evaluation...")
        
        # Set random seeds
        torch.manual_seed(42)
        np.random.seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
        
        best_val_vad_loss = float('inf')  # Now tracks valence_mae + arousal_mae
        num_epochs = self.config['training']['num_epochs']
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Training
            train_metrics = self.train_epoch(epoch)
            
            # Validation
            val_metrics = self.validate_epoch(epoch)
            
            # Scheduler step
            self.scheduler.step()
            
            # Print epoch summary
            print(f"Epoch {epoch + 1} Summary:")
            print(f"  Train - Total Loss: {train_metrics['total_loss']:.4f}")
            print(f"  Train - Conditioning Loss: {train_metrics['conditioning_loss']:.4f}")
            print(f"  Train - GPT Diff: {train_metrics['gpt_diff']:.3f}")
            print(f"  Train - Speaker Diff: {train_metrics['speaker_diff']:.3f}")
            print(f"  Train - Valence MAE: {train_metrics['valence_mae']:.3f}")
            print(f"  Train - Arousal MAE: {train_metrics['arousal_mae']:.3f}")
            print(f"  Val - Conditioning Loss: {val_metrics['conditioning_loss']:.4f}")
            print(f"  Val - Valence MAE: {val_metrics['valence_mae']:.3f}")
            print(f"  Val - Arousal MAE: {val_metrics['arousal_mae']:.3f}")
            print(f"  Learning Rate: {self.scheduler.get_last_lr()[0]:.6f}")
            
            # Save checkpoint (use validation VAD performance for best model)
            val_score = val_metrics['valence_mae'] + val_metrics['arousal_mae']  # Lower is better
            is_best = val_score < best_val_vad_loss
            if is_best:
                best_val_vad_loss = val_score
            
            if (epoch + 1) % self.config['training']['checkpoint_every'] == 0:
                self.save_checkpoint(epoch + 1, train_metrics, val_metrics, is_best)
        
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
    parser = argparse.ArgumentParser(description="Train Emotional XTTS model with VAD evaluation")
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