import json
import torch
import torchaudio
import numpy as np
import random
from torch.utils.data import Dataset
from typing import Dict, List, Optional
import os
from pathlib import Path


class ValenceArousalDataset(Dataset):
    def __init__(self, config: Dict):
        self.config = config
        self.sample_rate = config['data']['sample_rate']
        self.data_dir = Path(config['data'].get('data_dir', '../data'))
        
        metadata_path = config['data']['metadata_path']
        
        with open(metadata_path, 'r') as f:
            all_metadata = json.load(f)
        
        self.metadata = []
        skipped = 0
        
        for item in all_metadata:
            # Skip IEMOCAP for now since speaker info isn't available
            if item.get('dataset') == 'iemocap':
                continue
            
            # Check for valence and arousal values
            if 'valence' not in item or 'arousal' not in item:
                skipped += 1
                continue
                
            # Validate valence and arousal range [-1, 1]
            try:
                valence = float(item['valence'])
                arousal = float(item['arousal'])
                if not (-1 <= valence <= 1) or not (-1 <= arousal <= 1):
                    skipped += 1
                    continue
            except (ValueError, TypeError):
                skipped += 1
                continue
            
            # Check if audio file exists
            audio_path = self.data_dir / item['processed_audio_path']
            if not audio_path.exists():
                skipped += 1
                continue
            
            self.metadata.append(item)
        
        print(f"Loaded {len(self.metadata)} samples, skipped {skipped} invalid samples")
            
        # Parse speaker IDs and create speaker groupings
        self._parse_speaker_ids()
        self._create_speaker_mappings()
    
    def _parse_speaker_id(self, filename: str, dataset: str) -> str:
        """Parse speaker ID from filename based on dataset."""
        basename = os.path.basename(filename)
        
        if dataset == 'ravdess':
            # ravdess_ravdess_audio_speech_actors_01-24_Actor_05_03-01-05-01-01-01-05_3612.wav
            parts = basename.split('_')
            for i, part in enumerate(parts):
                if part == 'Actor' and i + 1 < len(parts):
                    return f"ravdess_{parts[i + 1]}"
            
        elif dataset == 'emovdb':
            # emovdb_full_anger_449-476_0469_7254.wav
            parts = basename.split('_')
            if len(parts) >= 4:
                return f"emovdb_{parts[3]}"
                
        elif dataset == 'cremad':
            # cremad_cremad_1038_IEO_DIS_HI_5551.wav
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
            # If speaker has only one sample, use the same file
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
            
            # Normalize audio
            max_val = torch.max(torch.abs(waveform))
            if max_val > 0:
                waveform = waveform / max_val
            
            return waveform.squeeze(0)
            
        except Exception as e:
            print(f"Error loading {description} from {audio_path}: {e}")
            return torch.zeros(int(self.sample_rate * 1.0))  # 1 second of silence
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        sample = self.metadata[idx]
        
        # Load target audio
        audio_path = self.data_dir / sample['processed_audio_path']
        waveform = self._load_audio_safe(str(audio_path), "target audio")
        
        # Get speaker reference file path (DON'T load the audio - just return the path!)
        speaker_ref_relative_path = self._get_speaker_reference(idx)
        speaker_ref_path = str((self.data_dir / speaker_ref_relative_path).resolve())  # Convert to absolute path
        
        # Get valence and arousal values
        valence = float(sample['valence'])
        arousal = float(sample['arousal'])
        
        return {
            'text': sample['text'],
            'audio': waveform,
            'speaker_ref': speaker_ref_path,  # ← File path string instead of tensor!
            'valence': torch.tensor(valence, dtype=torch.float32),
            'arousal': torch.tensor(arousal, dtype=torch.float32),
            'speaker_id': sample['speaker_id'],
            'audio_path': str(audio_path)
        }


def valence_arousal_collate_fn(batch):
    """Custom collate function for variable-length audio with valence-arousal."""
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
    
    # Speaker refs are now just file paths - no tensor loading or padding needed!
    speaker_refs = [item['speaker_ref'] for item in batch]
    
    audios = torch.stack(padded_audios)
    valences = torch.stack([item['valence'] for item in batch])
    arousals = torch.stack([item['arousal'] for item in batch])
    speaker_ids = [item['speaker_id'] for item in batch]
    audio_paths = [item['audio_path'] for item in batch]
    
    return {
        'texts': texts,
        'audios': audios,
        'speaker_refs': speaker_refs,  # ← List of file path strings (no tensors!)
        'valence': valences,
        'arousal': arousals,
        'speaker_ids': speaker_ids,
        'audio_paths': audio_paths,
        'audio_lengths': torch.tensor(audio_lengths),
        'languages': ['en'] * len(batch)  # Add default language for compatibility
    }