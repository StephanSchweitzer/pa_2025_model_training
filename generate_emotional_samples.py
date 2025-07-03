#!/usr/bin/env python3
"""
Generate emotional audio samples using the trained EmotionXTTS model.
"""

import torch
import torchaudio
import numpy as np
from pathlib import Path
import argparse
import sys
import os

from emotion_model import EmotionXTTS


def generate_emotional_samples(
    checkpoint_path="checkpoints/emotion_xtts/best_model.pt",
    reference_audio_path=None,
    output_dir="testing_emotional_outputs"
):
    """Generate audio samples with different emotions."""
    
    # Text to synthesize
    text = "I told you a million times, the cheese goes on the bread not the ground!!!"
    
    # Emotions to test (all 7 emotions)
    emotions = ["neutral", "happy", "sad", "angry", "surprised", "fearful", "disgusted"]
    emotion_ids = [0, 1, 2, 3, 4, 5, 6]
    
    print("=" * 60)
    print("EmotionXTTS Audio Generation")
    print("=" * 60)
    print(f"Text: '{text}'")
    print(f"Emotions: {emotions}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    print(f"Output directory: {output_path.absolute()}")
    
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
    
    # Load checkpoint
    if Path(checkpoint_path).exists():
        print(f"\n2. Loading checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("   ✓ Checkpoint loaded successfully")
        except Exception as e:
            print(f"   ✗ Failed to load checkpoint: {e}")
            return False
    else:
        print(f"\n2. ✗ Checkpoint not found: {checkpoint_path}")
        return False
    
    # Handle reference audio
    if reference_audio_path is None:
        print("\n3. Creating synthetic reference audio...")
        # Create a more realistic voice-like reference audio
        duration = 4.0  # Longer duration for better conditioning
        sample_rate = 22050
        t = np.linspace(0, duration, int(duration * sample_rate))
        
        # Create multiple harmonics for more voice-like quality
        fundamental = 150  # Lower fundamental frequency (more human-like)
        harmonics = [fundamental, fundamental * 2, fundamental * 3, fundamental * 4]
        amplitudes = [1.0, 0.5, 0.25, 0.1]  # Decreasing amplitude for harmonics
        
        # Create envelope with more realistic voice patterns
        envelope = np.ones_like(t)
        # Add some volume variation to simulate speech patterns
        for i in range(int(duration * 2)):  # Create "syllables"
            start = i * len(t) // (int(duration * 2))
            end = (i + 1) * len(t) // (int(duration * 2))
            if i % 2 == 0:  # Alternate loud/soft
                envelope[start:end] *= 0.8 + 0.4 * np.sin(np.linspace(0, np.pi, end - start))
        
        # Combine harmonics
        audio = np.zeros_like(t)
        for harmonic, amplitude in zip(harmonics, amplitudes):
            audio += amplitude * envelope * np.sin(2 * np.pi * harmonic * t)
        
        # Add some noise and make it sound more natural
        audio += 0.02 * np.random.randn(len(audio))
        
        # Apply simple formant-like filtering (emphasize certain frequencies)
        try:
            from scipy import signal
            # Create a simple bandpass filter to simulate formants
            b, a = signal.butter(4, [300, 3000], btype='band', fs=sample_rate)
            audio = signal.filtfilt(b, a, audio)
            print("   ✓ Applied formant filtering with scipy")
        except ImportError:
            # If scipy not available, apply simple frequency emphasis
            print("   ✓ Scipy not available, using simple filtering")
            # Simple high-pass to remove very low frequencies
            audio = np.diff(audio, prepend=audio[0])
            audio = audio / np.max(np.abs(audio))  # Renormalize
        
        # Normalize
        audio = audio / np.max(np.abs(audio)) * 0.8
        
        reference_audio_path = output_path / "reference_audio.wav"
        torchaudio.save(
            str(reference_audio_path), 
            torch.tensor(audio, dtype=torch.float32).unsqueeze(0), 
            sample_rate
        )
        print(f"   ✓ Synthetic reference created: {reference_audio_path}")
        print(f"   ✓ Reference duration: {duration}s, sample rate: {sample_rate}Hz")
    else:
        print(f"\n3. Using provided reference audio: {reference_audio_path}")
        if not Path(reference_audio_path).exists():
            print(f"   ✗ Reference audio file not found: {reference_audio_path}")
            return False
    
    # Set model to evaluation mode
    model.eval()
    
    # Generate samples for each emotion
    print("\n4. Generating emotional audio samples...")
    
    for emotion, emotion_id in zip(emotions, emotion_ids):
        print(f"\n   Generating '{emotion}' (ID: {emotion_id})...")
        
        try:
            with torch.no_grad():
                # Get emotion-conditioned latents first
                gpt_cond_latent, speaker_embedding = model.get_conditioning_latents_with_emotion(
                    audio_input=str(reference_audio_path),
                    emotion_id=emotion_id,  # Pass as int, the method will convert to tensor
                    training=False
                )
                
                # XTTS expects specific dimensions - preserve original shapes!
                # GPT latent should be [batch, time, features] for XTTS
                if gpt_cond_latent.dim() == 2:
                    gpt_cond_latent = gpt_cond_latent.unsqueeze(0)  # Add batch dim if missing
                
                # Speaker embedding: XTTS HiFiGAN decoder expects [batch, features, 1]!
                if speaker_embedding.dim() == 1:
                    speaker_embedding = speaker_embedding.unsqueeze(0).unsqueeze(-1)  # [features] -> [1, features, 1]
                elif speaker_embedding.dim() == 2:
                    speaker_embedding = speaker_embedding.unsqueeze(-1)  # [batch, features] -> [batch, features, 1]
                # If it's already 3D with shape [batch, features, 1], keep it as is
                
                print(f"   XTTS input shapes - GPT: {gpt_cond_latent.shape}, Speaker: {speaker_embedding.shape}")
                
                # Call XTTS inference directly with proper tensor shapes
                audio = model.xtts.inference(
                    text=text,
                    language="en", 
                    gpt_cond_latent=gpt_cond_latent,
                    speaker_embedding=speaker_embedding,
                    temperature=1.0,
                    #length_penalty=1.0,
                    #repetition_penalty=2.0,
                    #top_k=100,
                    #top_p=0.8,
                    #speed=1.0,
                    #enable_text_splitting=True

                )
                
                # Handle the output from XTTS inference
                # XTTS typically returns a dict with 'wav' key or direct audio array
                if isinstance(audio, dict):
                    if 'wav' in audio:
                        audio = audio['wav']
                    else:
                        # Take the first value if it's a dict without 'wav' key
                        audio = list(audio.values())[0]
                
                # Convert to tensor if it's a numpy array
                if isinstance(audio, np.ndarray):
                    audio = torch.from_numpy(audio)
                elif isinstance(audio, list):
                    audio = torch.tensor(audio, dtype=torch.float32)
                
                # Ensure audio is on CPU for saving
                audio = audio.cpu()
                
                # Handle dimensions - ensure it's [1, samples] for torchaudio.save
                if audio.dim() == 1:
                    audio = audio.unsqueeze(0)  # [samples] -> [1, samples]
                elif audio.dim() == 3:
                    audio = audio.squeeze(0)   # [1, 1, samples] -> [1, samples]
                elif audio.dim() > 2:
                    # Flatten extra dimensions but keep [channels, samples] format
                    audio = audio.view(audio.shape[-2], audio.shape[-1])
                
                # Ensure we have the right number of channels (mono or stereo)
                if audio.shape[0] > 2:
                    audio = audio[:1]  # Take only first channel if more than stereo
                
                # Save the generated audio
                output_file = output_path / f"{emotion}_sample.wav"
                
                # Determine sample rate (XTTS typically uses 22050)
                sample_rate = 22050
                
                # Save audio file
                torchaudio.save(
                    str(output_file),
                    audio,
                    sample_rate
                )
                
                print(f"   ✓ Saved: {output_file}")
                print(f"     Audio shape: {audio.shape}")
                print(f"     Duration: {audio.shape[-1] / sample_rate:.2f}s")
                
        except Exception as e:
            print(f"   ✗ Failed to generate {emotion}: {e}")
            import traceback
            print("     Full traceback:")
            traceback.print_exc()
            continue
    
    print("\n" + "=" * 60)
    print("✓ Audio generation completed!")
    print(f"✓ Files saved in: {output_path.absolute()}")
    print("=" * 60)
    
    # List generated files
    generated_files = list(output_path.glob("*_sample.wav"))
    if generated_files:
        print(f"\nGenerated {len(generated_files)} audio files:")
        for file in sorted(generated_files):
            print(f"  • {file.name}")
    
    return True


def main():
    # Check if running from Spyder or without command line arguments
    import sys
    running_from_spyder = len(sys.argv) == 1 or 'spyder' in sys.modules or 'runfile' in dir(__builtins__)
    
    if running_from_spyder:
        print("🔧 Detected Spyder environment - using default arguments")
        # Set defaults for Spyder
        checkpoint_path = "checkpoints/emotion_xtts/checkpoint_step_78000.pt"
        reference_audio_path = "ambar/ambar_test_1.wav"
        output_dir = "testing_emotional_outputs"
        
        print(f"📁 Checkpoint: {checkpoint_path}")
        print(f"🎵 Reference audio: {reference_audio_path}")
        print(f"📂 Output directory: {output_dir}")
        
    else:
        # Parse command line arguments normally
        parser = argparse.ArgumentParser(description="Generate emotional audio samples")
        parser.add_argument("--checkpoint", type=str, 
                           default="checkpoints/emotion_xtts/best_model.pt",
                           help="Path to model checkpoint")
        parser.add_argument("--reference", type=str, default="stef/test1.wav",
                           help="Path to reference audio file (default: stef/test1.wav)")
        parser.add_argument("--output-dir", type=str, default="testing_emotional_outputs",
                           help="Output directory for generated samples")
        
        args = parser.parse_args()
        
        checkpoint_path = args.checkpoint
        reference_audio_path = args.reference
        output_dir = args.output_dir
    
    success = generate_emotional_samples(
        checkpoint_path=checkpoint_path,
        reference_audio_path=reference_audio_path,
        output_dir=output_dir
    )
    
    if not success:
        print("\n⚠️  Audio generation failed. Please check the errors above.")
        if not running_from_spyder:
            sys.exit(1)
    else:
        print("\n🎵 You can now listen to the generated files to evaluate emotional quality!")


if __name__ == "__main__":
    main()