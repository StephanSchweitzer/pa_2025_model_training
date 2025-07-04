import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tempfile
import os
import torchaudio

from model_utils import (
    load_xtts_model, 
    verify_xtts_components, 
    get_model_info,
    move_model_to_device,
    freeze_model_parameters,
    get_device_info,
    DEFAULT_XTTS_CONFIG,
    EMOTION_MAPPING
)


class EmotionConditioningAdapter(nn.Module):   
    def __init__(self, num_emotions=7, emotion_dim=256, latent_dim=1024):
        super().__init__()
        self.num_emotions = num_emotions
        
        self.emotion_embedding = nn.Embedding(num_emotions, emotion_dim)
        
        self.gpt_latent_transform = nn.Sequential(
            nn.Linear(latent_dim + emotion_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.Tanh()
        )
        
        self.speaker_embed_transform = nn.Sequential(
            nn.Linear(512 + emotion_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.Tanh()
        )
        
        self.emotion_gate = nn.Parameter(torch.tensor(0.3))
        
    def forward(self, gpt_cond_latent, speaker_embedding, emotion_id):
        print(f"Input shapes - GPT: {gpt_cond_latent.shape}, Speaker: {speaker_embedding.shape}, Emotion: {emotion_id.shape if hasattr(emotion_id, 'shape') else emotion_id}")
        
        if gpt_cond_latent.dim() == 2:
            gpt_cond_latent = gpt_cond_latent.unsqueeze(0)
        
        if speaker_embedding.dim() == 3:
            if speaker_embedding.shape[-1] == 1:
                speaker_embedding = speaker_embedding.squeeze(-1)
            else:
                if speaker_embedding.shape[0] == 1:
                    speaker_embedding = speaker_embedding.squeeze(0)
        elif speaker_embedding.dim() == 1:
            speaker_embedding = speaker_embedding.unsqueeze(0)
        
        if isinstance(emotion_id, torch.Tensor):
            if emotion_id.dim() == 0:
                emotion_id = emotion_id.unsqueeze(0)
        else:
            emotion_id = torch.tensor(emotion_id, dtype=torch.long, device=gpt_cond_latent.device)
            if emotion_id.dim() == 0:
                emotion_id = emotion_id.unsqueeze(0)
        
        batch_size = emotion_id.shape[0]
        
        print(f"Processed shapes - GPT: {gpt_cond_latent.shape}, Speaker: {speaker_embedding.shape}, Emotion: {emotion_id.shape}")
        
        if gpt_cond_latent.shape[0] != batch_size:
            if gpt_cond_latent.shape[0] == 1:
                gpt_cond_latent = gpt_cond_latent.expand(batch_size, -1, -1)
            else:
                raise ValueError(f"Batch size mismatch: GPT {gpt_cond_latent.shape[0]} vs emotion {batch_size}")
                
        if speaker_embedding.shape[0] != batch_size:
            if speaker_embedding.shape[0] == 1:
                speaker_embedding = speaker_embedding.expand(batch_size, -1)
            else:
                raise ValueError(f"Batch size mismatch: Speaker {speaker_embedding.shape[0]} vs emotion {batch_size}")
        
        emotion_emb = self.emotion_embedding(emotion_id)
        
        T = gpt_cond_latent.shape[1]
        emotion_emb_expanded = emotion_emb.unsqueeze(1).expand(-1, T, -1)
        
        gpt_input = torch.cat([gpt_cond_latent, emotion_emb_expanded], dim=-1)
        gpt_transform = self.gpt_latent_transform(gpt_input)
        
        emotion_gpt_latent = gpt_cond_latent + self.emotion_gate * gpt_transform
        
        speaker_input = torch.cat([speaker_embedding, emotion_emb], dim=-1)
        speaker_transform = self.speaker_embed_transform(speaker_input)
        
        emotion_speaker_embedding = speaker_embedding + self.emotion_gate * speaker_transform
        
        print(f"Output shapes - GPT: {emotion_gpt_latent.shape}, Speaker: {emotion_speaker_embedding.shape}")
        
        return emotion_gpt_latent, emotion_speaker_embedding


class EmotionXTTS(nn.Module):   
    def __init__(self, config_path=None, checkpoint_path=None, num_emotions=7, local_model_dir="./models/xtts_v2"):
        super().__init__()
        
        self.local_model_dir = local_model_dir
        self.num_emotions = num_emotions
        
        print("Loading XTTS model...")
        
        try:
            self.xtts, self.config = load_xtts_model(
                config_path=config_path,
                checkpoint_path=checkpoint_path, 
                local_model_dir=local_model_dir
            )
            
            print("XTTS model loaded successfully!")
            
            verify_xtts_components(self.xtts)
            
        except Exception as e:
            print(f"❌ Failed to load XTTS model: {e}")
            raise RuntimeError(f"XTTS model loading failed: {e}")
        
        latent_dim = self.config.model_args.gpt_n_model_channels
        self.emotion_adapter = EmotionConditioningAdapter(
            num_emotions=num_emotions,
            emotion_dim=256,
            latent_dim=latent_dim
        )
        
        freeze_model_parameters(self.xtts, freeze=True)
        
        print("EmotionXTTS initialization complete!")
        print(f"Emotion mapping: {EMOTION_MAPPING}")
    
    def to(self, device):
        super().to(device)
        
        if hasattr(self, 'xtts'):
            self.xtts = move_model_to_device(self.xtts, device)
        
        if hasattr(self, 'emotion_adapter'):
            self.emotion_adapter = self.emotion_adapter.to(device)
            
        return self
    
    def cuda(self, device=None):
        super().cuda(device)
        
        target_device = device if device is not None else torch.device('cuda')
        
        if hasattr(self, 'xtts'):
            self.xtts = move_model_to_device(self.xtts, target_device)
        
        if hasattr(self, 'emotion_adapter'):
            self.emotion_adapter = self.emotion_adapter.cuda(device)
            
        return self
    
    def cpu(self):
        super().cpu()
        
        if hasattr(self, 'xtts'):
            self.xtts = move_model_to_device(self.xtts, torch.device('cpu'))
            
        if hasattr(self, 'emotion_adapter'):
            self.emotion_adapter = self.emotion_adapter.cpu()
            
        return self
    
    def _prepare_audio_tensor(self, audio_tensor):
        """Prepare audio tensor for XTTS conditioning."""
        if audio_tensor.dim() == 3:
            audio_tensor = audio_tensor.squeeze(0)
        if audio_tensor.dim() == 2:
            audio_tensor = audio_tensor.squeeze(0)
        
        if audio_tensor.dim() != 1:
            raise ValueError(f"Expected 1D audio tensor, got {audio_tensor.dim()}D")
        
        audio_tensor = audio_tensor.cpu()
        
        return audio_tensor
            
    def get_conditioning_latents_with_emotion(self, audio_input, emotion_id, training=False):
        if isinstance(emotion_id, torch.Tensor):
            emotion_id = emotion_id.to(next(self.parameters()).device)
        else:
            emotion_id = torch.tensor(emotion_id, dtype=torch.long, device=next(self.parameters()).device)
        
        if training and isinstance(audio_input, torch.Tensor):
            print(f"Processing audio tensor with shape: {audio_input.shape}")
            
            audio_input = self._prepare_audio_tensor(audio_input)
            print(f"Prepared audio tensor shape: {audio_input.shape}")
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                temp_path = tmp_file.name
                
            try:
                audio_to_save = audio_input.unsqueeze(0) if audio_input.dim() == 1 else audio_input
                torchaudio.save(temp_path, audio_to_save, DEFAULT_XTTS_CONFIG["sample_rate"])
                
                if training:
                    gpt_cond_latent, speaker_embedding = self.xtts.get_conditioning_latents(
                        audio_path=[temp_path]
                    )
                else:
                    with torch.no_grad():
                        gpt_cond_latent, speaker_embedding = self.xtts.get_conditioning_latents(
                            audio_path=[temp_path]
                        )
            finally:
                try:
                    os.unlink(temp_path)
                except:
                    pass
                        
        elif isinstance(audio_input, str) or isinstance(audio_input, list):
            if training:
                gpt_cond_latent, speaker_embedding = self.xtts.get_conditioning_latents(
                    audio_path=audio_input if isinstance(audio_input, list) else [audio_input]
                )
            else:
                with torch.no_grad():
                    gpt_cond_latent, speaker_embedding = self.xtts.get_conditioning_latents(
                        audio_path=audio_input if isinstance(audio_input, list) else [audio_input]
                    )
        else:
            raise ValueError(f"Unsupported audio_input type: {type(audio_input)}")
        
        print(f"Raw conditioning shapes - GPT: {gpt_cond_latent.shape}, Speaker: {speaker_embedding.shape}")
        
        device = next(self.parameters()).device
        
        gpt_cond_latent = gpt_cond_latent.to(device)
        speaker_embedding = speaker_embedding.to(device)
        
        emotion_gpt_latent, emotion_speaker_embedding = self.emotion_adapter(
            gpt_cond_latent, speaker_embedding, emotion_id
        )
        
        print(f"Emotion conditioning shapes - GPT: {emotion_gpt_latent.shape}, Speaker: {emotion_speaker_embedding.shape}")
        
        return emotion_gpt_latent, emotion_speaker_embedding
    
    def forward_training(self, text, audio_input, emotion_id, language="en"):
        try:
            if language not in DEFAULT_XTTS_CONFIG["supported_languages"]:
                print(f"⚠️ Warning: Language '{language}' not in supported languages. Defaulting to 'en'")
                language = "en"
            
            print(f"Forward training - text: '{text[:50]}...', audio shape: {audio_input.shape if hasattr(audio_input, 'shape') else 'N/A'}, emotion: {emotion_id}")
            
            if not isinstance(emotion_id, torch.Tensor):
                emotion_id = torch.tensor(emotion_id, dtype=torch.long).to(next(self.parameters()).device)
            
            gpt_cond_latent, speaker_embedding = self.get_conditioning_latents_with_emotion(
                audio_input, emotion_id, training=True
            )
            
            if gpt_cond_latent.dim() == 3 and gpt_cond_latent.shape[0] == 1:
                gpt_cond_latent = gpt_cond_latent.squeeze(0)  # [1, T, D] -> [T, D]
            if speaker_embedding.dim() == 2 and speaker_embedding.shape[0] == 1:
                speaker_embedding = speaker_embedding.squeeze(0)  # [1, D] -> [D]
            
            print(f"Final conditioning shapes - GPT: {gpt_cond_latent.shape}, Speaker: {speaker_embedding.shape}")
            
            output = self.xtts.inference(
                text,
                language,
                gpt_cond_latent,
                speaker_embedding,
                temperature=0.7,
                length_penalty=1.0,
                repetition_penalty=2.0,
                top_k=50,
                top_p=0.8,
                speed=1.0,
                enable_text_splitting=False  # Important for training!
            )
            
            if isinstance(output, dict):
                generated_audio = output.get('wav', output)
            else:
                generated_audio = output
            
            if isinstance(generated_audio, np.ndarray):
                generated_audio = torch.from_numpy(generated_audio).to(next(self.parameters()).device)
            elif isinstance(generated_audio, list):
                generated_audio = torch.tensor(generated_audio).to(next(self.parameters()).device)
            
            generated_audio = generated_audio.to(next(self.parameters()).device)
            if not generated_audio.requires_grad:
                generated_audio = generated_audio.clone().detach().requires_grad_(True)
            
            print(f"Generated audio shape: {generated_audio.shape}")
            
            return generated_audio
            
        except Exception as e:
            print(f"Error in forward_training: {e}")
            import traceback
            traceback.print_exc()
            raise e

    def inference_with_emotion(self, text, language, audio_path, emotion_id, **kwargs):
        if isinstance(emotion_id, str):
            emotion_name_to_id = {v: k for k, v in EMOTION_MAPPING.items()}
            if emotion_id.lower() in emotion_name_to_id:
                emotion_id = emotion_name_to_id[emotion_id.lower()]
            else:
                raise ValueError(f"Unknown emotion '{emotion_id}'. Available: {list(emotion_name_to_id.keys())}")
        
        if language not in DEFAULT_XTTS_CONFIG["supported_languages"]:
            print(f"⚠️ Warning: Language '{language}' not in supported languages. Defaulting to 'en'")
            language = "en"
        
        if not isinstance(emotion_id, torch.Tensor):
            emotion_id = torch.tensor(emotion_id, dtype=torch.long).to(next(self.parameters()).device)
        
        print(f"🎭 Generating speech with emotion: {EMOTION_MAPPING.get(emotion_id.item(), 'unknown')}")
        
        gpt_cond_latent, speaker_embedding = self.get_conditioning_latents_with_emotion(
            audio_path, emotion_id, training=False
        )
        
        if gpt_cond_latent.dim() > 2:
            gpt_cond_latent = gpt_cond_latent.squeeze(0)
        if speaker_embedding.dim() > 1:
            speaker_embedding = speaker_embedding.squeeze(0)
        
        return self.xtts.inference(
            text,
            language,
            gpt_cond_latent,
            speaker_embedding,
            **kwargs
        )
    
    def unfreeze_emotion_adapter(self):
        freeze_model_parameters(self.emotion_adapter, freeze=False)
        print("Emotion adapter unfrozen for training")
        
        
            
    def unfreeze_last_n_gpt_layers(self, n=2):
        gpt_layers = self.xtts.gpt.gpt.layers
        
        for layer in gpt_layers[-n:]:
            freeze_model_parameters(layer, freeze=False)
            
        print(f"Last {n} GPT layers unfrozen for fine-tuning")
        
        
    
    def get_model_info(self):
        base_info = get_model_info(self.xtts, self.config, self.local_model_dir)
        
        emotion_info = {
            "num_emotions": self.num_emotions,
            "emotion_mapping": EMOTION_MAPPING,
            "emotion_adapter_params": sum(p.numel() for p in self.emotion_adapter.parameters()),
            "emotion_adapter_trainable": sum(p.numel() for p in self.emotion_adapter.parameters() if p.requires_grad),
            "supported_languages": DEFAULT_XTTS_CONFIG["supported_languages"]
        }
        
        device_info = get_device_info()
        
        return {**base_info, **emotion_info, "device_info": device_info}
    
    def save_emotion_adapter(self, save_path):
        torch.save({
            'emotion_adapter_state_dict': self.emotion_adapter.state_dict(),
            'num_emotions': self.num_emotions,
            'emotion_mapping': EMOTION_MAPPING
        }, save_path)
        print(f"Emotion adapter saved to {save_path}")
    
    def load_emotion_adapter(self, load_path):
        checkpoint = torch.load(load_path, map_location=next(self.parameters()).device)
        self.emotion_adapter.load_state_dict(checkpoint['emotion_adapter_state_dict'])
        print(f"Emotion adapter loaded from {load_path}")
        
        # Verify emotion mapping compatibility
        saved_mapping = checkpoint.get('emotion_mapping', {})
        if saved_mapping != EMOTION_MAPPING:
            print(f"Warning: Emotion mapping mismatch!")
            print(f"Saved: {saved_mapping}")
            print(f"Current: {EMOTION_MAPPING}")


if __name__ == "__main__":
    model = EmotionXTTS(local_model_dir="./models/xtts_v2")
    
    if torch.cuda.is_available():
        model = model.cuda()
        print("Model moved to GPU")
    
    info = model.get_model_info()
    print("📊 Model Info:")
    for key, value in info.items():
        if key != "model_files":
            print(f"  {key}: {value}")
    
    # Example inference (you'll need actual audio file)
    # audio_output = model.inference_with_emotion(
    #     text="Hello, how are you today?",
    #     language="en", 
    #     audio_path="path/to/reference.wav",
    #     emotion_id="happy"  # Can use emotion name or ID
    # )