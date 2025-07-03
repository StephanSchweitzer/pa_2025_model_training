import torch
import torch.nn as nn
import numpy as np

from utils import (
    load_xtts_model, verify_xtts_components, get_xtts_model_info,
    DEFAULT_XTTS_CONFIG,
    
    get_base_conditioning_latents, prepare_audio_tensor,
    
    move_model_to_device, freeze_model_parameters, get_device_info,
    
    load_config,
    
    validate_and_clamp_av, prepare_av_tensor
)


class ArousaValenceConditioningAdapter(nn.Module):   
    def __init__(self, config):
        super().__init__()
        
        emotion_dim = config['model']['emotion_embedding_dim']
        
        self.av_processor = nn.Sequential(
            nn.Linear(2, emotion_dim // 2),
            nn.ReLU(),
            nn.Linear(emotion_dim // 2, emotion_dim),
            nn.Tanh()
        )
        
        self.gpt_latent_transform = None
        self.speaker_embed_transform = None
        
        self.emotion_gate = nn.Parameter(torch.tensor(0.3))
        self.emotion_dim = emotion_dim
        
    def initialize_transforms(self, latent_dim):
        """Initialize transforms after we know the XTTS latent dimensions"""
        self.gpt_latent_transform = nn.Sequential(
            nn.Linear(latent_dim + self.emotion_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.Tanh()
        )
        
        self.speaker_embed_transform = nn.Sequential(
            nn.Linear(512 + self.emotion_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.Tanh()
        )
        
    def forward(self, gpt_cond_latent, speaker_embedding, av_values, av_ranges):       
        if self.gpt_latent_transform is None:
            raise RuntimeError("Transforms not initialized. Call initialize_transforms() first.")
            
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
        
        av_tensor = prepare_av_tensor(av_values, gpt_cond_latent.device, av_ranges)
        batch_size = av_tensor.shape[0]
                
        if gpt_cond_latent.shape[0] != batch_size:
            if gpt_cond_latent.shape[0] == 1:
                gpt_cond_latent = gpt_cond_latent.expand(batch_size, -1, -1)
            else:
                raise ValueError(f"Batch size mismatch: GPT {gpt_cond_latent.shape[0]} vs AV {batch_size}")
                
        if speaker_embedding.shape[0] != batch_size:
            if speaker_embedding.shape[0] == 1:
                speaker_embedding = speaker_embedding.expand(batch_size, -1)
            else:
                raise ValueError(f"Batch size mismatch: Speaker {speaker_embedding.shape[0]} vs AV {batch_size}")
        
        emotion_emb = self.av_processor(av_tensor)
        
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
    def __init__(self, config, config_path=None, checkpoint_path=None):
        super().__init__()
        
        self.config = config
        
        # Get paths from config
        self.local_model_dir = config.get('paths', {}).get('model_dir', "./models/xtts_v2")
        
        print("Loading XTTS model...")
        
        try:
            self.xtts, self.xtts_config = load_xtts_model(
                config_path=config_path,
                checkpoint_path=checkpoint_path, 
                local_model_dir=self.local_model_dir
            )
            
            print("XTTS model loaded successfully!")
            verify_xtts_components(self.xtts)
            
        except Exception as e:
            print(f"❌ Failed to load XTTS model: {e}")
            raise RuntimeError(f"XTTS model loading failed: {e}")
        
        # Initialize emotion adapter with config
        self.emotion_adapter = ArousaValenceConditioningAdapter(config)
        
        # Initialize transforms now that we have XTTS loaded
        latent_dim = self.xtts_config.model_args.gpt_n_model_channels
        self.emotion_adapter.initialize_transforms(latent_dim)
        
        # Freeze parameters based on config
        unfreeze_layers = config['model'].get('unfreeze_last_n_layers', 0)
        freeze_model_parameters(self.xtts, freeze=True)
        
        if unfreeze_layers > 0:
            self.unfreeze_last_n_gpt_layers(unfreeze_layers)
        
        print("EmotionXTTS initialization complete!")
        print(f"Arousal-Valence ranges: {config['data']['av_ranges']}")
        print(f"Sample rate: {config['data']['sample_rate']}")
    
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
            
    def get_conditioning_latents_with_emotion(self, audio_input, av_values, training=False):        
        device = next(self.parameters()).device
        
        if isinstance(audio_input, torch.Tensor):
            audio_input = prepare_audio_tensor(audio_input)
            print(f"Prepared audio tensor shape: {audio_input.shape}")
        
        gpt_cond_latent, speaker_embedding = get_base_conditioning_latents(
            self.xtts, 
            audio_input, 
            training=training,
            sample_rate=self.config['data']['sample_rate']
        )
        
        # Move to device and apply emotion conditioning
        gpt_cond_latent = gpt_cond_latent.to(device)
        speaker_embedding = speaker_embedding.to(device)
        
        # Apply emotion conditioning while preserving XTTS-expected shapes
        emotion_gpt_latent, emotion_speaker_embedding = self.emotion_adapter(
            gpt_cond_latent, speaker_embedding, av_values, self.config['data']['av_ranges']
        )
        
        return emotion_gpt_latent, emotion_speaker_embedding
    
    def forward_training(self, text, audio_input, av_values, language="en"):
        try:
            if language not in DEFAULT_XTTS_CONFIG["supported_languages"]:
                print(f"⚠️ Warning: Language '{language}' not in supported languages. Defaulting to 'en'")
                language = "en"
            
            print(f"Forward training - text: '{text[:50]}...', audio shape: {audio_input.shape if hasattr(audio_input, 'shape') else 'N/A'}, AV: {av_values}")
            
            gpt_cond_latent, speaker_embedding = self.get_conditioning_latents_with_emotion(
                audio_input, av_values, training=True
            )
            
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

    def inference_with_emotion(self, text, language, audio_path, av_values, **kwargs):
        """
        Generate speech with arousal-valence conditioning.
        
        Args:
            text: Text to synthesize
            language: Language code  
            audio_path: Reference audio path
            av_values: Arousal-valence values as [arousal, valence] or [arousal, dominance, valence]
                      Values should be in configured range. Automatically clamped if outside range.
        """
        
        if language not in DEFAULT_XTTS_CONFIG["supported_languages"]:
            print(f"⚠️ Warning: Language '{language}' not in supported languages. Defaulting to 'en'")
            language = "en"
        
        # Prepare and validate AV values (automatically clamps to configured range)
        av_tensor = prepare_av_tensor(av_values, next(self.parameters()).device, self.config['data']['av_ranges'])
        
        print(f"🎭 Generating speech with A-V values: arousal={av_tensor[0][0]:.3f}, valence={av_tensor[0][1]:.3f}")
        
        gpt_cond_latent, speaker_embedding = self.get_conditioning_latents_with_emotion(
            audio_path, av_values, training=False
        )
        
        # ✅ XTTS v2 Compatible: Only adjust shapes if needed for inference
        if gpt_cond_latent.dim() > 2 and gpt_cond_latent.shape[0] == 1:
            gpt_cond_latent = gpt_cond_latent.squeeze(0)
        if speaker_embedding.dim() > 1 and speaker_embedding.shape[0] == 1:
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
        print("Arousal-Valence adapter unfrozen for training")
        
    def unfreeze_last_n_gpt_layers(self, n=None):
        if n is None:
            n = self.config['model'].get('unfreeze_last_n_layers', 2)
            
        gpt_layers = self.xtts.gpt.gpt.layers
        
        for layer in gpt_layers[-n:]:
            freeze_model_parameters(layer, freeze=False)
            
        print(f"Last {n} GPT layers unfrozen for fine-tuning")
    
    def get_model_info(self):
        base_info = get_xtts_model_info(self.xtts, self.xtts_config, self.local_model_dir)
        
        av_info = {
            "av_ranges": self.config['data']['av_ranges'],
            "emotion_adapter_params": sum(p.numel() for p in self.emotion_adapter.parameters()),
            "emotion_adapter_trainable": sum(p.numel() for p in self.emotion_adapter.parameters() if p.requires_grad),
            "supported_languages": DEFAULT_XTTS_CONFIG["supported_languages"],
            "sample_rate": self.config['data']['sample_rate'],
            "emotion_embedding_dim": self.config['model']['emotion_embedding_dim']
        }
        
        device_info = get_device_info()
        
        return {**base_info, **av_info, "device_info": device_info}
    
    def save_emotion_adapter(self, save_path):
        torch.save({
            'emotion_adapter_state_dict': self.emotion_adapter.state_dict(),
            'config': self.config,
            'model_type': 'arousal_valence'
        }, save_path)
        print(f"Arousal-Valence adapter saved to {save_path}")
    
    def load_emotion_adapter(self, load_path):
        checkpoint = torch.load(load_path, map_location=next(self.parameters()).device)
        self.emotion_adapter.load_state_dict(checkpoint['emotion_adapter_state_dict'])
        print(f"Arousal-Valence adapter loaded from {load_path}")
        
        # Verify model type compatibility
        saved_type = checkpoint.get('model_type', 'unknown')
        if saved_type != 'arousal_valence':
            print(f"Warning: Model type mismatch! Saved: {saved_type}, Expected: arousal_valence")


if __name__ == "__main__":
    # Load config from YAML
    config = load_config('config.yaml')
    
    model = EmotionXTTS(config)
    
    # Move to device specified in config
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cuda' and torch.cuda.is_available():
        model = model.cuda()
        print("Model moved to GPU")
    else:
        model = model.cpu()
        print("Model on CPU")
    
    info = model.get_model_info()
    print("📊 Model Info:")
    for key, value in info.items():
        if key != "model_files":
            print(f"  {key}: {value}")
    
    # Example inference with arousal-valence values from config ranges
    av_ranges = config['data']['av_ranges']
    print(f"\n🎭 Configured AV ranges:")
    print(f"  Arousal: {av_ranges['arousal']}")
    print(f"  Valence: {av_ranges['valence']}")