import torch
import torchaudio
from emotion_model import ValenceArousalXTTS

# === EASY TO CHANGE VARIABLES ===
REFERENCE_AUDIO = "voices/stef/test1.wav"
TEXT_TO_GENERATE = "I wonder what the day has in store for me, and I hope everything goes well!"
ADAPTER_PATH = "checkpoints/valence_arousal_xtts/test_2_adaptive_with_VAD.pth"

# Emotion settings (valence, arousal):
# Valence: 0.0 = very negative, 1.0 = very positive
# Arousal: 0.0 = very calm, 1.0 = very energetic
EMOTIONS = {
    "happy": (0.9, 0.9),    # High valence (positive), moderate-high arousal (energetic)
    "sad": (0.1, 0.1),      # Low valence (negative), low arousal (low energy)
    "angry": (0.9, 0.9),    # Low valence (negative), high arousal (very energetic)
}

# === LOAD MODEL ===
print("Loading emotional XTTS model...")
model = ValenceArousalXTTS(local_model_dir="./models/xtts_v2")

if torch.cuda.is_available():
    model = model.cuda()
    print("Model loaded on GPU")
else:
    print("Model loaded on CPU")

# Load trained adapter
print(f"Loading trained adapter from {ADAPTER_PATH}")
model.load_valence_arousal_adapter(ADAPTER_PATH)

# === GENERATE EMOTIONAL SPEECH ===
print(f"\nGenerating: '{TEXT_TO_GENERATE}'")
print(f"Reference voice: {REFERENCE_AUDIO}")

for emotion_name, (valence, arousal) in EMOTIONS.items():
    print(f"\n🎭 Generating {emotion_name} version (V={valence}, A={arousal})...")
    
    try:
        # Generate emotional audio
        audio_output = model.inference_with_valence_arousal(
            text=TEXT_TO_GENERATE,
            language="en",
            audio_path=REFERENCE_AUDIO,
            valence=valence,
            arousal=arousal,
            temperature=0.75,
            length_penalty=1.5,
            repetition_penalty=2.0
        )
        
        # Extract audio tensor
        if isinstance(audio_output, dict) and 'wav' in audio_output:
            audio = audio_output['wav']
        else:
            audio = audio_output
        
        # Convert to tensor if needed and ensure proper format
        if not isinstance(audio, torch.Tensor):
            audio = torch.from_numpy(audio)
        
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # Add channel dimension
        
        # Save audio file
        output_file = f"output_{emotion_name}.wav"
        torchaudio.save(output_file, audio.cpu(), 24000)
        print(f"✅ Saved: {output_file}")
        
    except Exception as e:
        print(f"❌ Error generating {emotion_name}: {e}")

print("\n🎉 Emotional inference test complete!")
print("\nGenerated files:")
for emotion_name in EMOTIONS.keys():
    print(f"  - output_{emotion_name}.wav")