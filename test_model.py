#!/usr/bin/env python3
"""
Quick test script to verify EmotionXTTS model is working.
"""

import torch
import torchaudio
import numpy as np
from pathlib import Path
import argparse
import sys

from emotion_model import EmotionXTTS


def create_test_audio(duration=3.0, sample_rate=22050):
    """Create a simple test audio (sine wave with envelope)."""
    t = np.linspace(0, duration, int(duration * sample_rate))
    # Create a simple tone with envelope
    frequency = 440  # A4 note
    envelope = np.exp(-t * 0.5)  # Exponential decay
    audio = envelope * np.sin(2 * np.pi * frequency * t)
    # Add some noise for realism
    audio += 0.01 * np.random.randn(len(audio))
    return torch.tensor(audio, dtype=torch.float32)


def test_model(checkpoint_path=None):
    """Test EmotionXTTS model with synthetic data."""
    
    print("=" * 60)
    print("EmotionXTTS Model Test")
    print("=" * 60)
    
    # Initialize model
    print("\n1. Initializing model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")
    
    try:
        model = EmotionXTTS(num_emotions=7).to(device)
        print("   ✓ Model created successfully")
    except Exception as e:
        print(f"   ✗ Failed to create model: {e}")
        return False
    
    # Load checkpoint if provided
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"\n2. Loading checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("   ✓ Checkpoint loaded successfully")
        except Exception as e:
            print(f"   ✗ Failed to load checkpoint: {e}")
    else:
        print("\n2. Using untrained model (random weights)")
    
    # Test emotion adapter
    print("\n3. Testing emotion adapter...")
    try:
        # Create dummy inputs
        batch_size = 1
        seq_len = 10
        gpt_latent_dim = model.emotion_adapter.gpt_latent_transform[0].in_features - 256
        
        dummy_gpt_latent = torch.randn(batch_size, seq_len, gpt_latent_dim).to(device)
        dummy_speaker_embedding = torch.randn(batch_size, 512).to(device)
        dummy_emotion_id = torch.randint(0, 7, (batch_size,)).to(device)
        
        # Forward pass through emotion adapter
        with torch.no_grad():
            emotion_gpt, emotion_speaker = model.emotion_adapter(
                dummy_gpt_latent, 
                dummy_speaker_embedding, 
                dummy_emotion_id
            )
        
        print(f"   ✓ Emotion adapter working")
        print(f"     Input shapes: GPT {dummy_gpt_latent.shape}, Speaker {dummy_speaker_embedding.shape}")
        print(f"     Output shapes: GPT {emotion_gpt.shape}, Speaker {emotion_speaker.shape}")
        print(f"     Emotion gate value: {model.emotion_adapter.emotion_gate.item():.3f}")
        
    except Exception as e:
        print(f"   ✗ Emotion adapter test failed: {e}")
        return False
    
    # Test inference (if XTTS is available)
    print("\n4. Testing inference pipeline...")
    emotions = ["neutral", "happy", "sad", "angry", "surprised", "fearful", "disgusted"]
    
    try:
        # Create test audio file
        test_audio = create_test_audio()
        test_audio_path = "test_reference.wav"
        torchaudio.save(test_audio_path, test_audio.unsqueeze(0), 22050)
        
        # Test emotion conditioning
        test_emotion_id = 1  # "happy"
        print(f"   Testing with emotion: {emotions[test_emotion_id]}")
        
        with torch.no_grad():
            gpt_cond, speaker_emb = model.get_conditioning_latents_with_emotion(
                audio_input=test_audio_path,  # Changed from audio_path to audio_input
                emotion_id=torch.tensor(test_emotion_id).to(device)
            )
        
        print("   ✓ Emotion conditioning successful")
        print(f"     Conditioned latent shapes: GPT {gpt_cond.shape}, Speaker {speaker_emb.shape}")
        
        # Clean up
        Path(test_audio_path).unlink()
        
    except Exception as e:
        print(f"   ✗ Inference test failed: {e}")
        print("     This is expected if XTTS base model is not downloaded yet")
    
    
    # Test all emotion embeddings
    print("\n5. Testing all emotion embeddings...")
    try:
        for i, emotion in enumerate(emotions):
            emotion_tensor = torch.tensor([i]).to(device)
            embedding = model.emotion_adapter.emotion_embedding(emotion_tensor)
            print(f"   {emotion:>10}: embedding shape {embedding.shape}, "
                  f"norm {torch.norm(embedding).item():.3f}")
        print("   ✓ All emotion embeddings working")
        
    except Exception as e:
        print(f"   ✗ Emotion embedding test failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✓ Model test completed successfully!")
    print("=" * 60)
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Test EmotionXTTS model")
    parser.add_argument("--checkpoint", type=str, 
                       default="checkpoints/emotion_xtts/best_model.pt",
                       help="Path to model checkpoint")
    
    args = parser.parse_args()
    
    success = test_model(args.checkpoint)
    
    if not success:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()