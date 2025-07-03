import json
import torch
import torchaudio
import numpy as np
import random
from torch.utils.data import Dataset
from typing import Dict, List, Optional
import os
from pathlib import Path

class EmotionDataset(Dataset):
    def __init__(self, config: Dict):
        self.config = config
        self.valid_emotions = config['data']['emotions']  # Keep as list to preserve order
        self.emotion_to_id = {emotion: idx for idx, emotion in enumerate(self.valid_emotions)}
        self.sample_rate = config['data']['sample_rate']
        self.samples_per_emotion = config['data'].get('samples_per_emotion', None)
        
        # Get data directory from config, with fallback
        self.data_dir = Path(config['data'].get('data_dir', '../data'))
        
        metadata_path = config['data']['metadata_path']
        
        with open(metadata_path, 'r') as f:
            all_metadata = json.load(f)
        
        self.metadata = []
        skipped = 0
        emotion_counts = {}
        limited_counts = {}
        iemocap_skipped = 0
        
        for item in all_metadata:
            # Skip IEMOCAP for now since speaker info isn't available
            if item.get('dataset') == 'iemocap':
                iemocap_skipped += 1
                continue
                
            emotion_label = item.get('emotion_label', 'unknown')
            
            if emotion_label not in self.valid_emotions:
                skipped += 1
                continue
            
            # Check if audio file exists
            audio_path = self.data_dir / item['processed_audio_path']
            if not audio_path.exists():
                print(f"Warning: Audio file not found: {audio_path}")
                skipped += 1
                continue
            
            if self.samples_per_emotion is not None:
                current_count = emotion_counts.get(emotion_label, 0)
                if current_count >= self.samples_per_emotion:
                    limited_counts[emotion_label] = limited_counts.get(emotion_label, 0) + 1
                    continue
            
            self.metadata.append(item)
            emotion_counts[emotion_label] = emotion_counts.get(emotion_label, 0) + 1
        
        print(f"Loaded {len(self.metadata)} samples, skipped {skipped} invalid emotions")
        print(f"Skipped {iemocap_skipped} IEMOCAP samples (no speaker info)")
        
        if self.samples_per_emotion is not None:
            print(f"Limited to {self.samples_per_emotion} samples per emotion")
            if limited_counts:
                total_limited = sum(limited_counts.values())
                print(f"Skipped {total_limited} additional samples due to per-emotion limits:")
                for emotion, count in limited_counts.items():
                    print(f"  {emotion}: {count} skipped")
        
        print("Final emotion distribution:")
        for emotion, count in emotion_counts.items():
            print(f"  {emotion}: {count}")
            
        # Parse speaker IDs and create speaker groupings
        self._parse_speaker_ids()
        self._create_speaker_mappings()
    
    def _parse_speaker_id(self, filename: str, dataset: str) -> str:
        """Parse speaker ID from filename based on dataset."""
        basename = os.path.basename(filename)
        
        if dataset == 'ravdess':
            # ravdess_ravdess_audio_speech_actors_01-24_Actor_05_03-01-05-01-01-01-05_3612.wav
            # Speaker ID is after "Actor_"
            parts = basename.split('_')
            for i, part in enumerate(parts):
                if part == 'Actor' and i + 1 < len(parts):
                    return f"ravdess_{parts[i + 1]}"
            
        elif dataset == 'emovdb':
            # emovdb_full_anger_449-476_0469_7254.wav
            # Speaker ID is the range field (449-476)
            parts = basename.split('_')
            if len(parts) >= 4:
                return f"emovdb_{parts[3]}"
                
        elif dataset == 'cremad':
            # cremad_cremad_1038_IEO_DIS_HI_5551.wav
            # Speaker ID is the first number after cremad_cremad_
            parts = basename.split('_')
            if len(parts) >= 3:
                return f"cremad_{parts[2]}"
        
        return f"{dataset}_unknown"
    
    def _parse_speaker_ids(self):
        """Add speaker IDs to metadata."""
        for item in self.metadata:
            filename = item['processed_audio_path']
            dataset = item['dataset']
            speaker_id = self._parse_speaker_id(filename, dataset)
            item['speaker_id'] = speaker_id
    
    def _create_speaker_mappings(self):
        """Create mappings from speaker to their audio files."""
        self.speaker_to_indices = {}
        
        for idx, item in enumerate(self.metadata):
            speaker_id = item['speaker_id']
            if speaker_id not in self.speaker_to_indices:
                self.speaker_to_indices[speaker_id] = []
            self.speaker_to_indices[speaker_id].append(idx)
        
        print(f"\nSpeaker statistics:")
        single_sample_speakers = 0
        multi_sample_speakers = 0
        for speaker_id, indices in self.speaker_to_indices.items():
            if len(indices) == 1:
                single_sample_speakers += 1
            else:
                multi_sample_speakers += 1
        
        print(f"  Speakers with multiple samples: {multi_sample_speakers}")
        print(f"  Speakers with single sample: {single_sample_speakers}")
        print(f"  Total unique speakers: {len(self.speaker_to_indices)}")
    
    def _get_speaker_reference(self, current_idx: int) -> str:
        """Get a different audio file from the same speaker as reference."""
        current_item = self.metadata[current_idx]
        speaker_id = current_item['speaker_id']
        speaker_indices = self.speaker_to_indices[speaker_id]
        
        # If speaker has multiple samples, use a different one
        if len(speaker_indices) > 1:
            available_indices = [idx for idx in speaker_indices if idx != current_idx]
            ref_idx = random.choice(available_indices)
            return self.metadata[ref_idx]['processed_audio_path']
        else:
            # If speaker has only one sample, use the same file (reconstruction)
            return current_item['processed_audio_path']
    
    def _load_audio_safe(self, audio_path: str, description: str = "audio") -> torch.Tensor:
        """Safely load audio with error handling."""
        try:
            waveform, sr = torchaudio.load(audio_path)
            
            # Resample if necessary
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Normalize audio to prevent clipping
            max_val = torch.max(torch.abs(waveform))
            if max_val > 0:
                waveform = waveform / max_val
            
            return waveform.squeeze(0)
            
        except Exception as e:
            print(f"Error loading {description} from {audio_path}: {e}")
            # Return silence as fallback
            return torch.zeros(int(self.sample_rate * 1.0))  # 1 second of silence
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        sample = self.metadata[idx]
        
        # Load target audio
        audio_path = self.data_dir / sample['processed_audio_path']
        waveform = self._load_audio_safe(str(audio_path), "target audio")
        
        # Load speaker reference audio
        speaker_ref_path = self.data_dir / self._get_speaker_reference(idx)
        speaker_ref_waveform = self._load_audio_safe(str(speaker_ref_path), "speaker reference")
        
        emotion_label = sample['emotion_label']
        emotion_id = self.emotion_to_id[emotion_label]  # This ensures 0-6 range
        
        return {
            'text': sample['text'],
            'audio': waveform,
            'speaker_ref': speaker_ref_waveform,
            'emotion_id': torch.tensor(emotion_id, dtype=torch.long),  # Use mapped ID
            'emotion_label': emotion_label,
            'speaker_id': sample['speaker_id'],
            'audio_path': str(audio_path)
        }

def collate_fn(batch):
    """Custom collate function for variable-length audio."""
    texts = [item['text'] for item in batch]
    
    # Handle variable-length target audio
    audio_lengths = [item['audio'].shape[0] for item in batch]
    max_audio_length = max(audio_lengths)
    
    padded_audios = []
    for item in batch:
        audio = item['audio']
        if audio.shape[0] < max_audio_length:
            padding = max_audio_length - audio.shape[0]
            audio = torch.nn.functional.pad(audio, (0, padding))
        padded_audios.append(audio)
    
    # Handle variable-length speaker reference audio
    speaker_ref_lengths = [item['speaker_ref'].shape[0] for item in batch]
    max_speaker_ref_length = max(speaker_ref_lengths)
    
    padded_speaker_refs = []
    for item in batch:
        speaker_ref = item['speaker_ref']
        if speaker_ref.shape[0] < max_speaker_ref_length:
            padding = max_speaker_ref_length - speaker_ref.shape[0]
            speaker_ref = torch.nn.functional.pad(speaker_ref, (0, padding))
        padded_speaker_refs.append(speaker_ref)
    
    audios = torch.stack(padded_audios)
    speaker_refs = torch.stack(padded_speaker_refs)
    emotion_ids = torch.stack([item['emotion_id'] for item in batch])
    emotion_labels = [item['emotion_label'] for item in batch]
    speaker_ids = [item['speaker_id'] for item in batch]
    
    return {
        'texts': texts,
        'audios': audios,
        'speaker_refs': speaker_refs,
        'emotion_ids': emotion_ids,
        'emotion_labels': emotion_labels,
        'speaker_ids': speaker_ids,
        'audio_lengths': torch.tensor(audio_lengths),
        'speaker_ref_lengths': torch.tensor(speaker_ref_lengths)
    }