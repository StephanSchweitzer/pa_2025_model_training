from typing import Dict, Optional, Tuple
from pathlib import Path
import torch
import torchaudio
import numpy as np
import os

# Import our downloader module functions
try:
    from .vad_downloader import ensure_model, get_model_info, check_dependencies
    DOWNLOADER_AVAILABLE = True
except ImportError:
    DOWNLOADER_AVAILABLE = False
    print("vad_downloader module not found. Place vad_downloader.py in the same directory.")

class VADAnalyzer:   
    def __init__(
        self, 
        model_dir: str = "models/vad_model",
        cache_dir: str = "models/vad_cache", 
        auto_download: bool = True,
        verbose: bool = True
    ):
        self.target_sample_rate = 16000
        self.model_dir = model_dir
        self.cache_dir = cache_dir
        self.verbose = verbose
        
        self.model = None
        self.model_available = False
        
        self._initialize_model(auto_download)
        
        
        
    def _initialize_model(self, auto_download: bool = True):
        if not DOWNLOADER_AVAILABLE:
            self._fallback_load()
            return
        
        if not check_dependencies():
            if self.verbose:
                print("Required dependencies not available")
            return
        
        try:
            self.model = ensure_model(
                model_dir=self.model_dir,
                cache_dir=self.cache_dir,
                auto_download=auto_download,
                verbose=self.verbose
            )
            
            if self.model is not None:
                self.model_available = True
                if self.verbose:
                    print("VAD Analyzer ready for emotion analysis!")
            else:
                if self.verbose:
                    print("Failed to load or download VAD model")
                    
        except Exception as e:
            if self.verbose:
                print(f"Error initializing model: {e}")
            self.model_available = False
    
    
    
    def _fallback_load(self):
        try:
            import audonnx
            
            model_path = Path(self.model_dir)
            if model_path.exists() and any(model_path.iterdir()):
                self.model = audonnx.load(str(model_path))
                self.model_available = True
                if self.verbose:
                    print(f"✅ Loaded model from {model_path}")
            else:
                if self.verbose:
                    print(f"❌ Model not found at {model_path}")
                    print("   Run the vad_downloader.py script first to download the model")
                    
        except Exception as e:
            if self.verbose:
                print(f"❌ Fallback model loading failed: {e}")
        
    def extract(self, audio_path: str) -> Tuple[Optional[Dict], str]:
        if not self.model_available:
            return None, "vad_model_unavailable"
        
        try:
            audio_data, sample_rate = torchaudio.load(audio_path)
            
            if audio_data.shape[0] > 1:
                audio_data = torch.mean(audio_data, dim=0, keepdim=True)
            
            if sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=self.target_sample_rate
                )
                audio_data = resampler(audio_data)
            
            audio_array = audio_data.squeeze().numpy().astype(np.float32)
            output = self.model(audio_array, self.target_sample_rate)
            
            return self._parse_model_output(output), "success"
            
        except Exception as e:
            return None, f"vad_error: {e}"
    
    def _parse_model_output(self, output: Dict) -> Dict:
        hidden_states = output['hidden_states']  # Shape: [1, 1024] - embeddings
        logits = output['logits']  # Shape: [1, 3] - [arousal, dominance, valence]
        
        arousal = float(logits[0][0])
        dominance = float(logits[0][1]) 
        valence = float(logits[0][2])
        
        confidence = float(1.0 / (1.0 + np.std(logits[0])))
        
        return {
            "valence": valence,
            "arousal": arousal, 
            "dominance": dominance,
            "vad_scores": [valence, arousal, dominance],
            "vad_embedding": hidden_states[0].tolist(),
            "confidence": confidence,
            "raw_logits": logits[0].tolist(),  # Include raw scores for debugging
            "raw_embeddings_shape": list(hidden_states.shape)
        }

    def get_analyzer_info(self) -> Dict:
        """Get comprehensive information about the analyzer and model."""
        base_info = {
            "analyzer_status": "available" if self.model_available else "unavailable",
            "model_directory": self.model_dir,
            "cache_directory": self.cache_dir,
            "target_sample_rate": self.target_sample_rate,
            "downloader_module": "available" if DOWNLOADER_AVAILABLE else "unavailable"
        }
        
        if not self.model_available:
            return base_info
        
        try:
            if DOWNLOADER_AVAILABLE:
                model_info = get_model_info(self.model_dir)
                base_info.update(model_info)
            
            test_signal = np.random.normal(size=self.target_sample_rate).astype(np.float32)
            test_output = self.model(test_signal, self.target_sample_rate)
            
            base_info.update({
                "model_output_keys": list(test_output.keys()),
                "hidden_states_shape": list(test_output['hidden_states'].shape),
                "logits_shape": list(test_output['logits'].shape),
                "logits_labels": ["arousal", "dominance", "valence"],
                "sample_logits": test_output['logits'][0].tolist()
            })
            
        except Exception as e:
            base_info["model_test_error"] = str(e)
        
        return base_info
    
    

    def reload_model(self, auto_download: bool = True):
        """Reload the model (useful if model files were updated)."""
        if self.verbose:
            print("Reloading VAD model...")
        self._initialize_model(auto_download)
        
        

if __name__ == "__main__":
    print("VAD Analyzer with Integrated Downloader")
    print("=" * 50)
    
    analyzer = VADAnalyzer(
        model_dir="models/vad_model",
        auto_download=True,
        verbose=True
    )
    
    if analyzer.model_available:
        print("\n" + "="*50)
        print("📊 Analyzer Information:")
        info = analyzer.get_analyzer_info()
        for key, value in info.items():
            if isinstance(value, (list, dict)):
                print(f"  {key}: {type(value).__name__} with {len(value)} items")
            else:
                print(f"  {key}: {value}")
        
        print("""
            result, status = analyzer.extract('path/to/audio.wav')
            if status == 'success':
                print(f"Valence: {result['valence']:.3f}")
                print(f"Arousal: {result['arousal']:.3f}")
                print(f"Dominance: {result['dominance']:.3f}")
        """)
        
    else:
        print("\n Analyzer not ready. Check the error messages above.")
        print("\n Troubleshooting:")
        print("1. Ensure you have: pip install audeer audonnx torchaudio")
        print("2. Run: python vad_downloader.py")
        print("3. Check your internet connection")