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
    DEFAULT_XTTS_CONFIG
)


class ValenceArousalAdapter(nn.Module):   
    def __init__(self, emotion_dim=256, latent_dim=1024):
        super().__init__()
        
        # Valence-arousal input layer (2 inputs: valence, arousal)
        self.va_encoder = nn.Sequential(
            nn.Linear(2, emotion_dim),
            nn.ReLU(),
            nn.Linear(emotion_dim, emotion_dim)
        )
        
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
        
    def forward(self, gpt_cond_latent, speaker_embedding, valence, arousal):
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
        
        # Ensure valence and arousal are tensors
        if not isinstance(valence, torch.Tensor):
            valence = torch.tensor(valence, dtype=torch.float32, device=gpt_cond_latent.device)
        if not isinstance(arousal, torch.Tensor):
            arousal = torch.tensor(arousal, dtype=torch.float32, device=gpt_cond_latent.device)
        
        if valence.dim() == 0:
            valence = valence.unsqueeze(0)
        if arousal.dim() == 0:
            arousal = arousal.unsqueeze(0)
        
        batch_size = valence.shape[0]
        
        # Combine valence and arousal
        va_input = torch.stack([valence, arousal], dim=1)  # [batch_size, 2]
        emotion_emb = self.va_encoder(va_input)  # [batch_size, emotion_dim]
        
        # Ensure batch sizes match
        if gpt_cond_latent.shape[0] != batch_size:
            if gpt_cond_latent.shape[0] == 1:
                gpt_cond_latent = gpt_cond_latent.expand(batch_size, -1, -1)
        
        if speaker_embedding.shape[0] != batch_size:
            if speaker_embedding.shape[0] == 1:
                speaker_embedding = speaker_embedding.expand(batch_size, -1)
        
        # Transform GPT latents
        T = gpt_cond_latent.shape[1]
        emotion_emb_expanded = emotion_emb.unsqueeze(1).expand(-1, T, -1)
        
        gpt_input = torch.cat([gpt_cond_latent, emotion_emb_expanded], dim=-1)
        gpt_transform = self.gpt_latent_transform(gpt_input)
        
        emotion_gpt_latent = gpt_cond_latent + self.emotion_gate * gpt_transform
        
        # Transform speaker embedding
        speaker_input = torch.cat([speaker_embedding, emotion_emb], dim=-1)
        speaker_transform = self.speaker_embed_transform(speaker_input)
        
        emotion_speaker_embedding = speaker_embedding + self.emotion_gate * speaker_transform
        
        return emotion_gpt_latent, emotion_speaker_embedding


class ValenceArousalXTTS(nn.Module):   
    def __init__(self, config_path=None, checkpoint_path=None, local_model_dir="./models/xtts_v2"):
        super().__init__()
        
        self.local_model_dir = local_model_dir
        
        print("Loading XTTS model...")
        
        try:
            # First try to load from local directory
            self.xtts, self.config = load_xtts_model(
                config_path=config_path,
                checkpoint_path=checkpoint_path, 
                local_model_dir=local_model_dir
            )
            print("Loaded XTTS from local directory")
            
        except Exception as e:
            print(f"Local model not found: {e}")
            print("Falling back to TTS API...")
            
            # Fallback to TTS API (works with firewalls)
            try:
                from TTS.api import TTS
                tts_api = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
                self.xtts = tts_api.synthesizer.tts_model
                self.config = self.xtts.config
                print("Loaded XTTS via TTS API")
            except Exception as api_error:
                raise RuntimeError(f"Failed to load XTTS via both local and API methods: {api_error}")
        
        verify_xtts_components(self.xtts)
        
        latent_dim = self.config.model_args.gpt_n_model_channels
        self.va_adapter = ValenceArousalAdapter(
            emotion_dim=256,
            latent_dim=latent_dim
        )
        
        freeze_model_parameters(self.xtts, freeze=True)
        
        print("ValenceArousalXTTS initialization complete!")
    
    def to(self, device):
        super().to(device)
        
        if hasattr(self, 'xtts'):
            self.xtts = move_model_to_device(self.xtts, device)
        
        if hasattr(self, 'va_adapter'):
            self.va_adapter = self.va_adapter.to(device)
            
        return self
    
    def cuda(self, device=None):
        super().cuda(device)
        
        target_device = device if device is not None else torch.device('cuda')
        
        if hasattr(self, 'xtts'):
            self.xtts = move_model_to_device(self.xtts, target_device)
        
        if hasattr(self, 'va_adapter'):
            self.va_adapter = self.va_adapter.cuda(device)
            
        return self
    
    def _prepare_audio_tensor(self, audio_tensor):
        """Prepare audio tensor for XTTS conditioning."""
        if audio_tensor.dim() == 3:
            audio_tensor = audio_tensor.squeeze(0)
        if audio_tensor.dim() == 2:
            audio_tensor = audio_tensor.squeeze(0)
        
        if audio_tensor.dim() != 1:
            raise ValueError(f"Expected 1D audio tensor, got {audio_tensor.dim()}D")
        
        return audio_tensor.cpu()
            
    def get_conditioning_latents_with_valence_arousal(self, audio_input, valence, arousal, training=False):
        # Ensure valence and arousal are on correct device
        device = next(self.parameters()).device
        if isinstance(valence, torch.Tensor):
            valence = valence.to(device)
        else:
            valence = torch.tensor(valence, dtype=torch.float32, device=device)
            
        if isinstance(arousal, torch.Tensor):
            arousal = arousal.to(device)
        else:
            arousal = torch.tensor(arousal, dtype=torch.float32, device=device)
        
        # Handle different audio input types
        if training and isinstance(audio_input, torch.Tensor):
            audio_input = self._prepare_audio_tensor(audio_input)
            
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
                        
        elif isinstance(audio_input, (str, list)):
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
        
        # Move to device
        gpt_cond_latent = gpt_cond_latent.to(device)
        speaker_embedding = speaker_embedding.to(device)
        
        # Apply valence-arousal transformation
        emotion_gpt_latent, emotion_speaker_embedding = self.va_adapter(
            gpt_cond_latent, speaker_embedding, valence, arousal
        )
        
        return emotion_gpt_latent, emotion_speaker_embedding
    


    def inference_with_valence_arousal(self, text, language, audio_path, valence, arousal, **kwargs):
        if language not in DEFAULT_XTTS_CONFIG["supported_languages"]:
            language = "en"
        
        print(f"Generating speech with valence: {valence}, arousal: {arousal}")
        
        gpt_cond_latent, speaker_embedding = self.get_conditioning_latents_with_valence_arousal(
            audio_path, valence, arousal, training=False
        )
        
        # Ensure correct dimensions
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
    
    def unfreeze_valence_arousal_adapter(self):
        freeze_model_parameters(self.va_adapter, freeze=False)
        print("Valence-arousal adapter unfrozen for training")
            
    def unfreeze_last_n_gpt_layers(self, n=2):
        gpt_layers = self.xtts.gpt.gpt.layers
        
        for layer in gpt_layers[-n:]:
            freeze_model_parameters(layer, freeze=False)
            
        print(f"Last {n} GPT layers unfrozen for fine-tuning")
    
    def get_model_info(self):
        base_info = get_model_info(self.xtts, self.config, self.local_model_dir)
        
        va_info = {
            "va_adapter_params": sum(p.numel() for p in self.va_adapter.parameters()),
            "va_adapter_trainable": sum(p.numel() for p in self.va_adapter.parameters() if p.requires_grad),
            "supported_languages": DEFAULT_XTTS_CONFIG["supported_languages"]
        }
        
        device_info = get_device_info()
        
        return {**base_info, **va_info, "device_info": device_info}
    
    def save_valence_arousal_adapter(self, save_path):
        torch.save({
            'va_adapter_state_dict': self.va_adapter.state_dict()
        }, save_path)
        print(f"Valence-arousal adapter saved to {save_path}")
    
    def load_valence_arousal_adapter(self, load_path):
        checkpoint = torch.load(load_path, map_location=next(self.parameters()).device)
        self.va_adapter.load_state_dict(checkpoint['va_adapter_state_dict'])
        print(f"Valence-arousal adapter loaded from {load_path}")


if __name__ == "__main__":
    model = ValenceArousalXTTS(local_model_dir="./models/xtts_v2")
    
    if torch.cuda.is_available():
        model = model.cuda()
        print("Model moved to GPU")
    
    info = model.get_model_info()
    print("Model Info:")
    for key, value in info.items():
        if key != "model_files":
            print(f"  {key}: {value}")
    
    # Example inference
    # audio_output = model.inference_with_valence_arousal(
    #     text="Hello, how are you today?",
    #     language="en", 
    #     audio_path="path/to/reference.wav",
    #     valence=0.8,  # Positive emotion
    #     arousal=0.6   # Moderate activation
    # )