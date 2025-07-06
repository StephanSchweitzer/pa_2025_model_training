from typing import Dict, Optional, Tuple
from pathlib import Path
import torch
import torchaudio
import numpy as np
import os


class VADAnalyzer:   
    def __init__(
        self, 
        model_dir: str = "models/vad_model",
        cache_dir: str = "models/vad_cache", 
        verbose: bool = True
    ):
        self.target_sample_rate = 16000
        self.model_dir = model_dir
        self.cache_dir = cache_dir
        self.verbose = verbose
        
        self.model = None
        self.model_available = False
        
        self._initialize_model()
    
    
    
    def _initialize_model(self):
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
        base_info = {
            "analyzer_status": "available" if self.model_available else "unavailable",
            "model_directory": self.model_dir,
            "cache_directory": self.cache_dir,
            "target_sample_rate": self.target_sample_rate,
        }
        
        if not self.model_available:
            return base_info
        
        try:            
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
        if self.verbose:
            print("Reloading VAD model...")
        self._initialize_model()
        
        

if __name__ == "__main__":
    print("VAD Analyzer with Integrated Downloader")
    print("=" * 50)
    
    analyzer = VADAnalyzer(
        model_dir="models/vad_model",
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