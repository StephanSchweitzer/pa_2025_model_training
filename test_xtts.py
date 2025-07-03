#!/usr/bin/env python3
"""
Test script to verify XTTS model loading.
Run this before training to ensure everything is set up correctly.
"""

import torch
from emotion_model import EmotionXTTS

def test_xtts_loading():
    """Test if XTTS model loads correctly."""
    print("Testing XTTS model loading...")
    
    try:
        # Test loading EmotionXTTS
        model = EmotionXTTS(num_emotions=7)
        
        # Move entire model to same device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        print(f"✅ EmotionXTTS loaded successfully on {device}!")
        
        # Test basic model components
        assert model.xtts is not None, "XTTS model is None"
        assert model.xtts.gpt is not None, "GPT component is None"
        assert model.emotion_adapter is not None, "Emotion adapter is None"
        
        # Check for vocoder more flexibly
        vocoder_found = False
        vocoder_name = None
        for attr_name in ['hifigan', 'vocoder', 'decoder', 'hifigan_decoder']:
            if hasattr(model.xtts, attr_name) and getattr(model.xtts, attr_name) is not None:
                vocoder_found = True
                vocoder_name = attr_name
                break
        
        if not vocoder_found:
            print("⚠️  Warning: No vocoder component found, but continuing...")
            print("Available XTTS attributes:", [attr for attr in dir(model.xtts) if not attr.startswith('_')])
        else:
            print(f"✅ Vocoder component found: {vocoder_name}")
        
        print("✅ All essential model components are present!")
        
        # Test emotion adapter with proper device handling
        print("Testing emotion adapter...")
        
        dummy_gpt_latent = torch.randn(1, 10, 1024, device=device)  # [B, T, D]
        dummy_speaker_emb = torch.randn(1, 512, device=device)      # [B, D]
        dummy_emotion = torch.tensor([0], device=device)            # [B]
        
        # Verify all tensors are on the same device
        print(f"GPT latent device: {dummy_gpt_latent.device}")
        print(f"Speaker embedding device: {dummy_speaker_emb.device}")
        print(f"Emotion tensor device: {dummy_emotion.device}")
        print(f"Model device: {next(model.parameters()).device}")
        
        with torch.no_grad():
            emotion_gpt, emotion_speaker = model.emotion_adapter(
                dummy_gpt_latent, dummy_speaker_emb, dummy_emotion
            )
            assert emotion_gpt.shape == dummy_gpt_latent.shape, "GPT latent shape mismatch"
            assert emotion_speaker.shape == dummy_speaker_emb.shape, "Speaker embedding shape mismatch"
        
        print("✅ Emotion adapter test passed!")
        
        print("✅ XTTS model is ready for training!")
        return True
        
    except Exception as e:
        print(f"❌ XTTS loading failed: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting tips:")
        print("1. Make sure TTS is installed: pip install TTS")
        print("2. Try running: tts --model_name tts_models/multilingual/multi-dataset/xtts_v2 --list_language_idx")
        print("3. Check your internet connection (model needs to download on first run)")
        print("4. Try clearing TTS cache: rm -rf ~/.local/share/tts/")
        return False

def test_basic_tts():
    """Test basic TTS functionality."""
    print("\nTesting basic TTS functionality...")
    
    try:
        from TTS.api import TTS
        print("Loading TTS model...")
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=torch.cuda.is_available())
        print("✅ TTS model loaded successfully!")
        
        # Test that we can access the internal model
        assert hasattr(tts, 'synthesizer'), "TTS synthesizer not found"
        assert hasattr(tts.synthesizer, 'tts_model'), "TTS model not found in synthesizer"
        
        model = tts.synthesizer.tts_model
        print(f"✅ Internal XTTS model type: {type(model)}")
        print(f"✅ GPT component: {'✓' if hasattr(model, 'gpt') and model.gpt is not None else '✗'}")
        
        # Check for vocoder components more thoroughly
        vocoder_found = False
        vocoder_name = None
        for attr_name in ['hifigan', 'vocoder', 'decoder', 'hifigan_decoder']:
            if hasattr(model, attr_name) and getattr(model, attr_name) is not None:
                vocoder_found = True
                vocoder_name = attr_name
                break
        
        print(f"✅ Vocoder component: {'✓ (' + vocoder_name + ')' if vocoder_found else '✗'}")
        
        if not vocoder_found:
            print("Available model attributes:")
            attrs = [attr for attr in dir(model) if not attr.startswith('_') and not callable(getattr(model, attr))]
            print(f"  {attrs}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic TTS test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔧 XTTS Installation Test")
    print("=" * 50)
    
    # Test basic TTS first
    tts_ok = test_basic_tts()
    
    if tts_ok:
        # Test our custom EmotionXTTS
        emotion_xtts_ok = test_xtts_loading()
        
        if emotion_xtts_ok:
            print("\n🎉 All tests passed! You're ready to start training.")
        else:
            print("\n❌ EmotionXTTS test failed. Check the error messages above.")
    else:
        print("\n❌ Basic TTS test failed. Please install TTS properly.")
        print("Try: pip install TTS")
    
    print("\n" + "=" * 50)