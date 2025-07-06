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
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.device = torch.device(self.config['device'] if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.setup_logging()
        
        print("Loading VAD analyzer...")
        self.vad_analyzer = VADAnalyzer(
            model_dir=self.config.get('vad_model_dir', 'vad_model'),
            verbose=True
        )
        
        if not self.vad_analyzer.model_available:
            raise RuntimeError("VAD analyzer failed to initialize. Cannot train without emotional evaluation.")
        
        print("Loading Emotional XTTS model...")
        self.model = ValenceArousalXTTS(local_model_dir="./models/xtts_v2")
        self.model = self.model.to(self.device)  # This should handle all submodules
        
        print(f"Model device: {next(self.model.parameters()).device}")
        print(f"Adapter device: {next(self.model.va_adapter.parameters()).device}")
        
        self.model.unfreeze_valence_arousal_adapter()
        
        print("Loading dataset...")
        self.setup_dataset()
        
        self.setup_optimizer()
        
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        self.inference_kwargs = {
            "temperature": 0.75,
            "length_penalty": 1.0,
            "repetition_penalty": 1.1,
            "top_k": 50,
            "top_p": 0.8
        }
        
        self.training_step_count = 0
        
        vad_config = self.config.get('vad_training', {})
        adaptation_config = vad_config.get('adaptation', {})
        
        self.adaptive_gpt_strength = adaptation_config.get('initial_gpt_strength', 0.2)
        self.adaptive_speaker_strength = adaptation_config.get('initial_speaker_strength', 0.1)
        self.vad_feedback_history = []
        
        self.vad_eval_frequency = vad_config.get('vad_eval_frequency', 10)
        self.low_accuracy_threshold = adaptation_config.get('low_accuracy_threshold', 0.7)
        self.high_accuracy_threshold = adaptation_config.get('high_accuracy_threshold', 0.9)
        self.increase_rate_gpt = adaptation_config.get('increase_rate_gpt', 1.05)
        self.increase_rate_speaker = adaptation_config.get('increase_rate_speaker', 1.03)
        self.decrease_rate_gpt = adaptation_config.get('decrease_rate_gpt', 0.98)
        self.decrease_rate_speaker = adaptation_config.get('decrease_rate_speaker', 0.99)
        self.min_gpt_strength = adaptation_config.get('min_gpt_strength', 0.05)
        self.max_gpt_strength = adaptation_config.get('max_gpt_strength', 0.8)
        self.min_speaker_strength = adaptation_config.get('min_speaker_strength', 0.02)
        self.max_speaker_strength = adaptation_config.get('max_speaker_strength', 0.4)
        self.max_feedback_history = adaptation_config.get('max_feedback_history', 50)
        self.recent_history_window = adaptation_config.get('recent_history_window', 10)
        
        modes_config = vad_config.get('modes', {})
        self.vad_training_enabled = modes_config.get('training', True)
        self.vad_validation_enabled = modes_config.get('validation', True)
        self.vad_disabled = modes_config.get('disable_vad_eval', False)
        self.vad_validation_only = modes_config.get('validation_only', False)
        
        print(f"VAD Training Configuration:")
        print(f"  Evaluation frequency: every {self.vad_eval_frequency} steps")
        print(f"  Training VAD: {'enabled' if self.vad_training_enabled and not self.vad_disabled else 'disabled'}")
        print(f"  Validation VAD: {'enabled' if self.vad_validation_enabled and not self.vad_disabled else 'disabled'}")
        print(f"  Initial strengths: GPT={self.adaptive_gpt_strength:.3f}, Speaker={self.adaptive_speaker_strength:.3f}")
        print(f"  Adaptation thresholds: low={self.low_accuracy_threshold:.2f}, high={self.high_accuracy_threshold:.2f}")
        
        print("Trainer initialization complete!")

    def debug_tensor_devices(self, *tensors, names=None):
        if names is None:
            names = [f"tensor_{i}" for i in range(len(tensors))]
        
        for tensor, name in zip(tensors, names):
            if torch.is_tensor(tensor):
                print(f"{name}: device={tensor.device}, shape={tensor.shape}")
            else:
                print(f"{name}: not a tensor, type={type(tensor)}")

    def setup_logging(self):
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
        self.dataset = ValenceArousalDataset(self.config)
        
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
            if isinstance(audio_tensor, np.ndarray):
                audio_tensor = torch.from_numpy(audio_tensor)
            
            if torch.all(audio_tensor == 0) or torch.max(torch.abs(audio_tensor)) == 0:
                print("Warning: Generated audio is silence - inference likely failed")
                audio_tensor = 0.1 * torch.sin(2 * 3.14159 * 440 * torch.linspace(0, 1, sample_rate))
            
            audio_tensor = audio_tensor.detach().cpu() if hasattr(audio_tensor, 'detach') else audio_tensor.cpu()
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            elif audio_tensor.dim() == 3:
                audio_tensor = audio_tensor.squeeze(0)
            elif audio_tensor.dim() == 2 and audio_tensor.shape[0] > 1:
                audio_tensor = audio_tensor.mean(dim=0, keepdim=True)
            
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            
            torchaudio.save(temp_path, audio_tensor, sample_rate)
            return temp_path
        except Exception as e:
            try:
                os.unlink(temp_path)
            except:
                pass
            raise e

    def cleanup_temp_file(self, temp_path):
        try:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        except Exception as e:
            print(f"Warning: Could not delete temporary file {temp_path}: {e}")

    def generate_audio_sample(self, text, speaker_ref, target_valence, target_arousal):
        try:
            print(f"Generating: '{text[:50]}...' with V={target_valence.item():.3f}, A={target_arousal.item():.3f}")
            
            target_valence = target_valence.to(self.device)
            target_arousal = target_arousal.to(self.device)
            
            audio_output = self.model.inference_with_valence_arousal(
                text=text,
                language="en",
                audio_path=speaker_ref,
                valence=target_valence.item(),
                arousal=target_arousal.item(),
                **self.inference_kwargs
            )
            
            if isinstance(audio_output, dict) and 'wav' in audio_output:
                generated_audio = audio_output['wav']
            elif isinstance(audio_output, (torch.Tensor, np.ndarray)):
                generated_audio = audio_output
            else:
                raise ValueError(f"Unexpected audio output format: {type(audio_output)}")
            
            if isinstance(generated_audio, np.ndarray):
                generated_audio = torch.from_numpy(generated_audio).to(self.device)
            elif isinstance(generated_audio, torch.Tensor):
                generated_audio = generated_audio.to(self.device)
            
            if generated_audio.dim() == 3:
                generated_audio = generated_audio.squeeze(0)
            if generated_audio.dim() == 2:
                generated_audio = generated_audio.squeeze(0)
            
            audio_max = torch.max(torch.abs(generated_audio))
            if audio_max == 0 or torch.isnan(audio_max):
                print("Warning: Generated audio is silence or contains NaN")
                sample_rate = 22050
                duration = 1.0
                t = torch.linspace(0, duration, int(sample_rate * duration), device=self.device)
                generated_audio = 0.1 * torch.sin(2 * 3.14159 * 440 * t)
            
            print(f"Generated audio: shape={generated_audio.shape}, max={audio_max:.4f}, device={generated_audio.device}")
            return generated_audio
            
        except Exception as e:
            print(f"Error generating audio: {e}")
            sample_rate = 22050
            duration = 1.0
            t = torch.linspace(0, duration, int(sample_rate * duration), device=self.device)
            return 0.1 * torch.sin(2 * 3.14159 * 440 * t)

    def get_vad_guided_targets(self, target_valence, target_arousal):        
        emotion_magnitude = torch.sqrt(target_valence**2 + target_arousal**2)
        
        target_gpt_modification = torch.tensor(
            self.adaptive_gpt_strength * (0.5 + emotion_magnitude.item()), 
            requires_grad=False, 
            device=target_valence.device
        )
        
        target_speaker_modification = torch.tensor(
            self.adaptive_speaker_strength * (0.5 + emotion_magnitude.item()), 
            requires_grad=False, 
            device=target_valence.device
        )
        
        return target_gpt_modification, target_speaker_modification

    def update_vad_guided_targets(self, vad_accuracy):        
        self.vad_feedback_history.append(vad_accuracy)
        if len(self.vad_feedback_history) > self.max_feedback_history:
            self.vad_feedback_history.pop(0)
        
        recent_accuracy = sum(self.vad_feedback_history[-self.recent_history_window:]) / min(self.recent_history_window, len(self.vad_feedback_history))
        
        if recent_accuracy < self.low_accuracy_threshold:  # VAD shows we're not hitting targets
            self.adaptive_gpt_strength *= self.increase_rate_gpt
            self.adaptive_speaker_strength *= self.increase_rate_speaker
            print(f"Increasing conditioning strength: GPT={self.adaptive_gpt_strength:.3f} (accuracy: {recent_accuracy:.3f})")
        elif recent_accuracy > self.high_accuracy_threshold:  # VAD shows we're very accurate
            self.adaptive_gpt_strength *= self.decrease_rate_gpt
            self.adaptive_speaker_strength *= self.decrease_rate_speaker
            print(f"Decreasing conditioning strength: GPT={self.adaptive_gpt_strength:.3f} (accuracy: {recent_accuracy:.3f})")
        
        self.adaptive_gpt_strength = max(self.min_gpt_strength, min(self.max_gpt_strength, self.adaptive_gpt_strength))
        self.adaptive_speaker_strength = max(self.min_speaker_strength, min(self.max_speaker_strength, self.adaptive_speaker_strength))

    def compute_conditioning_loss(self, speaker_ref_path, target_valence, target_arousal):
        try:
            device = self.device
            
            target_valence = target_valence.to(device)
            target_arousal = target_arousal.to(device)
            
            with torch.no_grad():
                original_gpt_latent, original_speaker_emb = self.model.xtts.get_conditioning_latents([speaker_ref_path])
                original_gpt_latent = original_gpt_latent.to(device)
                original_speaker_emb = original_speaker_emb.to(device)
            
            emotion_gpt_latent, emotion_speaker_emb = self.model.get_conditioning_latents_with_valence_arousal(
                speaker_ref_path, target_valence, target_arousal, training=True
            )
            
            emotion_gpt_latent = emotion_gpt_latent.to(device)
            emotion_speaker_emb = emotion_speaker_emb.to(device)
            
            gpt_diff = torch.norm(emotion_gpt_latent - original_gpt_latent)
            speaker_diff = torch.norm(emotion_speaker_emb - original_speaker_emb)
            
            target_gpt_modification, target_speaker_modification = self.get_vad_guided_targets(
                target_valence, target_arousal
            )
            
            gpt_loss = F.smooth_l1_loss(gpt_diff, target_gpt_modification)
            speaker_loss = F.smooth_l1_loss(speaker_diff, target_speaker_modification)
            
            reg_loss = 0.01 * (torch.norm(emotion_gpt_latent) + torch.norm(emotion_speaker_emb))
            
            total_loss = gpt_loss + speaker_loss + reg_loss
            
            return total_loss, {
                'gpt_diff': gpt_diff.item(),
                'speaker_diff': speaker_diff.item(),
                'target_gpt_mod': target_gpt_modification.item(),
                'target_speaker_mod': target_speaker_modification.item(),
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
        temp_path = None
        try:
            generated_audio_cpu = generated_audio.detach().cpu()
            temp_path = self.save_temp_audio(generated_audio_cpu)
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
        temp_path = None
        try:
            device = self.device
            
            ref_gpt_latent, ref_speaker_emb = self.model.xtts.get_conditioning_latents([speaker_ref_path])
            ref_speaker_emb = ref_speaker_emb.to(device)
            
            generated_audio_cpu = generated_audio.detach().cpu()
            
            temp_path = self.save_temp_audio(generated_audio_cpu)
            gen_gpt_latent, gen_speaker_emb = self.model.xtts.get_conditioning_latents([temp_path])
            gen_speaker_emb = gen_speaker_emb.to(device)
            
            ref_speaker_emb = ref_speaker_emb.flatten()
            gen_speaker_emb = gen_speaker_emb.flatten()
            
            similarity = F.cosine_similarity(ref_speaker_emb, gen_speaker_emb, dim=0)
            
            speaker_loss = 1.0 - similarity
            
            return speaker_loss, similarity.item()
            
        except Exception as e:
            print(f"Error computing speaker similarity: {e}")
            return torch.tensor(0.5, requires_grad=True, device=self.device), 0.0
        
        finally:
            if temp_path:
                self.cleanup_temp_file(temp_path)

    def training_step(self, batch):
        try:
            self.optimizer.zero_grad()
            
            batch_size = len(batch['texts'])
            total_loss = 0.0
            valid_samples = 0
            
            conditioning_metrics = {
                'gpt_diff': 0.0,
                'speaker_diff': 0.0,
                'target_gpt_mod': 0.0,
                'target_speaker_mod': 0.0
            }
            
            do_vad_eval = (
                not self.vad_disabled and
                self.vad_training_enabled and
                not self.vad_validation_only and
                (self.training_step_count % self.vad_eval_frequency == 0)
            )
            vad_metrics = {'valence_mae': 0.0, 'arousal_mae': 0.0} if do_vad_eval else {}
            vad_samples = 0
            
            for i in range(batch_size):
                try:
                    target_valence = batch['target_valences'][i].to(self.device)
                    target_arousal = batch['target_arousals'][i].to(self.device)
                    
                    conditioning_loss, cond_info = self.compute_conditioning_loss(
                        batch['speaker_refs'][i],
                        target_valence,
                        target_arousal
                    )
                    
                    if torch.isfinite(conditioning_loss):
                        total_loss += conditioning_loss
                        valid_samples += 1
                        
                        for key in conditioning_metrics:
                            if key in cond_info:
                                conditioning_metrics[key] += cond_info[key]
                    
                    if do_vad_eval:
                        with torch.no_grad():
                            try:
                                generated_audio = self.generate_audio_sample(
                                    text=batch['texts'][i],
                                    speaker_ref=batch['speaker_refs'][i],
                                    target_valence=target_valence,
                                    target_arousal=target_arousal
                                )
                                
                                temp_path = self.save_temp_audio(generated_audio.detach().cpu())
                                vad_result, status = self.vad_analyzer.extract(temp_path)
                                self.cleanup_temp_file(temp_path)
                                
                                if status == "success" and vad_result:
                                    vad_metrics['valence_mae'] += abs(vad_result['valence'] - target_valence.item())
                                    vad_metrics['arousal_mae'] += abs(vad_result['arousal'] - target_arousal.item())
                                    vad_samples += 1
                                    
                            except Exception as e:
                                print(f"VAD evaluation failed for sample {i}: {e}")
                    
                except Exception as e:
                    print(f"Error processing sample {i}: {e}")
                    continue
            
            if valid_samples > 0:
                avg_loss = total_loss / valid_samples
                
                avg_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.va_adapter.parameters(), 1.0)
                
                self.optimizer.step()
                
                for key in conditioning_metrics:
                    conditioning_metrics[key] /= valid_samples
                
                if do_vad_eval and vad_samples > 0:
                    for key in vad_metrics:
                        vad_metrics[key] /= vad_samples
                    
                    vad_accuracy = max(0.0, 1.0 - (vad_metrics['valence_mae'] + vad_metrics['arousal_mae']) / 2.0)
                    self.update_vad_guided_targets(vad_accuracy)
                
                self.training_step_count = getattr(self, 'training_step_count', 0) + 1
                
                return {
                    'total_loss': avg_loss.item(),
                    'conditioning_loss': avg_loss.item(),
                    'gpt_diff': conditioning_metrics['gpt_diff'],
                    'speaker_diff': conditioning_metrics['speaker_diff'], 
                    'target_gpt_mod': conditioning_metrics['target_gpt_mod'],
                    'target_speaker_mod': conditioning_metrics['target_speaker_mod'],
                    'valence_mae': vad_metrics.get('valence_mae', 0.0),
                    'arousal_mae': vad_metrics.get('arousal_mae', 0.0),
                    'valid_samples': valid_samples,
                    'vad_evaluated': do_vad_eval and vad_samples > 0,
                    'adaptive_gpt_strength': self.adaptive_gpt_strength,
                    'adaptive_speaker_strength': self.adaptive_speaker_strength
                }
            else:
                print("Warning: No valid samples in batch - skipping")
                return {
                    'total_loss': 0.0,
                    'conditioning_loss': 0.0,
                    'gpt_diff': 0.0,
                    'speaker_diff': 0.0,
                    'target_gpt_mod': 0.0,
                    'target_speaker_mod': 0.0,
                    'valence_mae': 0.0,
                    'arousal_mae': 0.0,
                    'valid_samples': 0,
                    'vad_evaluated': False,
                    'adaptive_gpt_strength': self.adaptive_gpt_strength,
                    'adaptive_speaker_strength': self.adaptive_speaker_strength
                }
                
        except Exception as e:
            print(f"Error in training step: {e}")
            return {
                'total_loss': 0.0,
                'conditioning_loss': 0.0,
                'gpt_diff': 0.0,
                'speaker_diff': 0.0, 
                'target_gpt_mod': 0.0,
                'target_speaker_mod': 0.0,
                'valence_mae': 0.0,
                'arousal_mae': 0.0,
                'valid_samples': 0,
                'vad_evaluated': False,
                'adaptive_gpt_strength': self.adaptive_gpt_strength,
                'adaptive_speaker_strength': self.adaptive_speaker_strength
            }

    def validation_step(self, batch):
        with torch.no_grad():
            try:
                batch_size = len(batch['texts'])
                total_conditioning_loss = 0.0
                valid_samples = 0
                
                vad_metrics = {
                    'valence_mae': 0.0,
                    'arousal_mae': 0.0
                }
                
                vad_eval_enabled = (
                    not self.vad_disabled and 
                    self.vad_validation_enabled
                )
                
                for i in range(batch_size):
                    try:
                        target_valence = batch['target_valences'][i].to(self.device)
                        target_arousal = batch['target_arousals'][i].to(self.device)
                        
                        conditioning_loss, cond_info = self.compute_conditioning_loss(
                            batch['speaker_refs'][i],
                            target_valence,
                            target_arousal
                        )
                        
                        if vad_eval_enabled:
                            generated_audio = self.generate_audio_sample(
                                text=batch['texts'][i],
                                speaker_ref=batch['speaker_refs'][i],
                                target_valence=target_valence,
                                target_arousal=target_arousal
                            )
                            
                            temp_path = self.save_temp_audio(generated_audio.detach().cpu())
                            vad_result, status = self.vad_analyzer.extract(temp_path)
                            self.cleanup_temp_file(temp_path)
                            
                            if torch.isfinite(conditioning_loss) and status == "success" and vad_result:
                                total_conditioning_loss += conditioning_loss.item()
                                valid_samples += 1
                                
                                vad_metrics['valence_mae'] += abs(vad_result['valence'] - target_valence.item())
                                vad_metrics['arousal_mae'] += abs(vad_result['arousal'] - target_arousal.item())
                        else:
                            if torch.isfinite(conditioning_loss):
                                total_conditioning_loss += conditioning_loss.item()
                                valid_samples += 1
                        
                    except Exception as e:
                        print(f"Error in validation sample {i}: {e}")
                        continue
                
                if valid_samples > 0:
                    result = {
                        'conditioning_loss': total_conditioning_loss / valid_samples,
                        'valid_samples': valid_samples
                    }
                    
                    if vad_eval_enabled:
                        result.update({
                            'valence_mae': vad_metrics['valence_mae'] / valid_samples,
                            'arousal_mae': vad_metrics['arousal_mae'] / valid_samples,
                        })
                    else:
                        result.update({
                            'valence_mae': 0.0,
                            'arousal_mae': 0.0,
                        })
                    
                    return result
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
        self.model.train()
        epoch_metrics = {
            'total_loss': 0.0,
            'conditioning_loss': 0.0,
            'gpt_diff': 0.0,
            'speaker_diff': 0.0,
            'target_gpt_mod': 0.0,
            'target_speaker_mod': 0.0,
            'valence_mae': 0.0,
            'arousal_mae': 0.0,
            'valid_samples': 0,
            'adaptive_gpt_strength': 0.0,
            'adaptive_speaker_strength': 0.0
        }
        num_batches = 0
        
        pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            if self.config.get('quick_test', {}).get('enabled', False):
                max_steps = self.config['quick_test'].get('max_steps', float('inf'))
                if batch_idx >= max_steps:
                    print(f"Quick test mode: Stopping at step {batch_idx}")
                    break
            
            metrics = self.training_step(batch)
            
            for key in epoch_metrics:
                epoch_metrics[key] += metrics[key]
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f'{metrics["total_loss"]:.4f}',
                'gpt_str': f'{metrics["adaptive_gpt_strength"]:.3f}',
                'spk_str': f'{metrics["adaptive_speaker_strength"]:.3f}',
                'v_mae': f'{metrics["valence_mae"]:.3f}' if metrics['vad_evaluated'] else 'N/A',
                'a_mae': f'{metrics["arousal_mae"]:.3f}' if metrics['vad_evaluated'] else 'N/A'
            })
            
            step = epoch * len(self.train_dataloader) + batch_idx
            self.writer.add_scalar('Train/TotalLoss', metrics['total_loss'], step)
            self.writer.add_scalar('Train/ConditioningLoss', metrics['conditioning_loss'], step)
            self.writer.add_scalar('Train/GPTDiff', metrics['gpt_diff'], step)
            self.writer.add_scalar('Train/SpeakerDiff', metrics['speaker_diff'], step)
            self.writer.add_scalar('Train/AdaptiveGPTStrength', metrics['adaptive_gpt_strength'], step)
            self.writer.add_scalar('Train/AdaptiveSpeakerStrength', metrics['adaptive_speaker_strength'], step)
            
            if metrics['vad_evaluated']:
                self.writer.add_scalar('Train/ValenceMAE', metrics['valence_mae'], step)
                self.writer.add_scalar('Train/ArousalMAE', metrics['arousal_mae'], step)
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}")
                print(f"  Total Loss: {metrics['total_loss']:.4f}")
                print(f"  Conditioning Loss: {metrics['conditioning_loss']:.4f}")
                print(f"  GPT Diff: {metrics['gpt_diff']:.3f} (target: {metrics['target_gpt_mod']:.3f})")
                print(f"  Speaker Diff: {metrics['speaker_diff']:.3f} (target: {metrics['target_speaker_mod']:.3f})")
                print(f"  Adaptive Strengths: GPT={metrics['adaptive_gpt_strength']:.3f}, Speaker={metrics['adaptive_speaker_strength']:.3f}")
                if metrics['vad_evaluated']:
                    print(f"  Valence MAE: {metrics['valence_mae']:.3f}")
                    print(f"  Arousal MAE: {metrics['arousal_mae']:.3f}")
        
        if num_batches > 0:
            for key in epoch_metrics:
                epoch_metrics[key] /= num_batches
        
        return epoch_metrics

    def validate_epoch(self, epoch):
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
                
                for key in epoch_metrics:
                    epoch_metrics[key] += metrics[key]
                num_batches += 1
        
        if num_batches > 0:
            for key in epoch_metrics:
                epoch_metrics[key] /= num_batches
        
        self.writer.add_scalar('Val/ConditioningLoss', epoch_metrics['conditioning_loss'], epoch)
        self.writer.add_scalar('Val/ValenceMAE', epoch_metrics['valence_mae'], epoch)
        self.writer.add_scalar('Val/ArousalMAE', epoch_metrics['arousal_mae'], epoch)
        
        return epoch_metrics

    def save_checkpoint(self, epoch, train_metrics, val_metrics, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'va_adapter_state_dict': self.model.va_adapter.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'config': self.config,
            'adaptive_gpt_strength': self.adaptive_gpt_strength,
            'adaptive_speaker_strength': self.adaptive_speaker_strength,
            'vad_feedback_history': self.vad_feedback_history
        }
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.checkpoint_dir / "best_checkpoint.pth"
            torch.save(checkpoint, best_path)
            print(f"New best checkpoint saved: {best_path}")
        
        print(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.va_adapter.load_state_dict(checkpoint['va_adapter_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'adaptive_gpt_strength' in checkpoint:
            self.adaptive_gpt_strength = checkpoint['adaptive_gpt_strength']
        if 'adaptive_speaker_strength' in checkpoint:
            self.adaptive_speaker_strength = checkpoint['adaptive_speaker_strength']
        if 'vad_feedback_history' in checkpoint:
            self.vad_feedback_history = checkpoint['vad_feedback_history']
        
        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"Restored adaptive strengths: GPT={self.adaptive_gpt_strength:.3f}, Speaker={self.adaptive_speaker_strength:.3f}")
        print(f"VAD feedback history: {len(self.vad_feedback_history)} samples")
        return checkpoint['epoch'], checkpoint.get('train_metrics', {}), checkpoint.get('val_metrics', {})

    def train(self):
        print("Starting VAD-Guided Emotional XTTS training...")
        
        torch.manual_seed(42)
        np.random.seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
        
        best_val_vad_loss = float('inf')  # Now tracks valence_mae + arousal_mae
        num_epochs = self.config['training']['num_epochs']
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            train_metrics = self.train_epoch(epoch)
            
            val_metrics = self.validate_epoch(epoch)
            
            self.scheduler.step()
            
            print(f"Epoch {epoch + 1} Summary:")
            print(f"  Train - Total Loss: {train_metrics['total_loss']:.4f}")
            print(f"  Train - Conditioning Loss: {train_metrics['conditioning_loss']:.4f}")
            print(f"  Train - GPT Diff: {train_metrics['gpt_diff']:.3f} (target: {train_metrics['target_gpt_mod']:.3f})")
            print(f"  Train - Speaker Diff: {train_metrics['speaker_diff']:.3f} (target: {train_metrics['target_speaker_mod']:.3f})")
            print(f"  Train - Adaptive Strengths: GPT={train_metrics['adaptive_gpt_strength']:.3f}, Speaker={train_metrics['adaptive_speaker_strength']:.3f}")
            print(f"  Train - Valence MAE: {train_metrics['valence_mae']:.3f}")
            print(f"  Train - Arousal MAE: {train_metrics['arousal_mae']:.3f}")
            print(f"  Val - Conditioning Loss: {val_metrics['conditioning_loss']:.4f}")
            print(f"  Val - Valence MAE: {val_metrics['valence_mae']:.3f}")
            print(f"  Val - Arousal MAE: {val_metrics['arousal_mae']:.3f}")
            print(f"  Learning Rate: {self.scheduler.get_last_lr()[0]:.6f}")
            
            val_score = val_metrics['valence_mae'] + val_metrics['arousal_mae']  # Lower is better
            is_best = val_score < best_val_vad_loss
            if is_best:
                best_val_vad_loss = val_score
            
            if (epoch + 1) % self.config['training']['checkpoint_every'] == 0:
                self.save_checkpoint(epoch + 1, train_metrics, val_metrics, is_best)
        
        print("Training completed!")
        
        final_adapter_path = self.checkpoint_dir / "emotional_adapter_final.pth"
        torch.save({
            'va_adapter_state_dict': self.model.va_adapter.state_dict(),
            'config': self.config,
            'adaptive_gpt_strength': self.adaptive_gpt_strength,
            'adaptive_speaker_strength': self.adaptive_speaker_strength,
            'vad_feedback_history': self.vad_feedback_history
        }, final_adapter_path)
        
        print(f"Final emotional adapter saved to: {final_adapter_path}")
        print(f"Final adaptive strengths: GPT={self.adaptive_gpt_strength:.3f}, Speaker={self.adaptive_speaker_strength:.3f}")
        print(f"VAD evaluation frequency used: every {self.vad_eval_frequency} steps")
        print(f"Total VAD feedback samples collected: {len(self.vad_feedback_history)}")
        
        self.writer.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train VAD-Guided Emotional XTTS model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    trainer = EmotionalXTTSTrainer(args.config)
    
    if args.resume:
        print(f"Resuming training from {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    trainer.train()


if __name__ == "__main__":
    main()