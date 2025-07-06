import json
import torch
import torchaudio
import numpy as np
import random
from torch.utils.data import Dataset
from typing import Dict, List, Optional
import os
from pathlib import Path
from collections import defaultdict


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
                
            # Validate valence and arousal range [0, 1] with clamping
            try:
                valence = float(item['valence'])
                arousal = float(item['arousal'])
                
                # Clamp values to [0, 1] range instead of rejecting
                valence = max(0.0, min(1.0, valence))
                arousal = max(0.0, min(1.0, arousal))
                
                # Update the clamped values back to the item
                item['valence'] = valence
                item['arousal'] = arousal
                
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
            
        # Parse speaker IDs and emotions, create cross-emotional mappings
        self._parse_speaker_ids()
        self._parse_emotions()
        self._create_cross_emotional_mappings()
    
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
    
    def _parse_emotion_from_filename(self, filename: str, dataset: str) -> str:
        """Extract emotion label from filename."""
        basename = os.path.basename(filename)
        
        if dataset == 'cremad':
            # cremad_cremad_1084_TIE_SAD_XX_2607
            parts = basename.split('_')
            if len(parts) >= 5:
                return parts[4]  # "SAD", "ANG", "HAP", "NEU", "DIS", "FEA"
        
        elif dataset == 'ravdess':
            # ravdess_ravdess_Actor_24_03-01-03-02-02-02-24_2761
            parts = basename.split('_')
            # Find the part with dashes (emotion encoding)
            for part in parts:
                if '-' in part and len(part.split('-')) >= 3:
                    emotion_code = part.split('-')[2]  # 3rd position is emotion
                    emotion_map = {
                        '01': 'NEU', '02': 'CAL', '03': 'HAP', '04': 'SAD', 
                        '05': 'ANG', '06': 'FEA', '07': 'DIS', '08': 'SUR'
                    }
                    return emotion_map.get(emotion_code, 'UNK')
        
        elif dataset == 'emovdb':
            # emovdb_full_anger_449-476_0469_7254.wav
            parts = basename.split('_')
            if len(parts) >= 3:
                emotion_raw = parts[2]
                # Map to standard abbreviations
                emotion_map = {
                    'anger': 'ANG', 'angry': 'ANG',
                    'disgust': 'DIS', 'disgusted': 'DIS',
                    'fear': 'FEA', 'fearful': 'FEA',
                    'happy': 'HAP', 'happiness': 'HAP',
                    'neutral': 'NEU',
                    'sad': 'SAD', 'sadness': 'SAD',
                    'surprise': 'SUR', 'surprised': 'SUR'
                }
                return emotion_map.get(emotion_raw.lower(), 'UNK')
        
        return 'UNK'
    
    def _parse_speaker_ids(self):
        """Add speaker IDs to metadata."""
        for item in self.metadata:
            filename = item['processed_audio_path']
            dataset = item['dataset']
            speaker_id = self._parse_speaker_id(filename, dataset)
            item['speaker_id'] = speaker_id
    
    def _parse_emotions(self):
        """Add emotion labels to metadata."""
        for item in self.metadata:
            filename = item['processed_audio_path']
            dataset = item['dataset']
            emotion = self._parse_emotion_from_filename(filename, dataset)
            item['emotion_label'] = emotion
    
    def _create_cross_emotional_mappings(self):
        """Create mappings for cross-emotional training pairs."""
        # Fix pickle issue: use regular dict instead of lambda defaultdict
        self.speaker_emotion_map = {}
        self.speaker_to_emotions = {}
        
        for idx, item in enumerate(self.metadata):
            speaker_id = item['speaker_id']
            emotion = item['emotion_label']
            
            # Skip unknown emotions
            if emotion == 'UNK':
                continue
            
            # Initialize nested structure manually
            if speaker_id not in self.speaker_emotion_map:
                self.speaker_emotion_map[speaker_id] = {}
                self.speaker_to_emotions[speaker_id] = set()
            
            if emotion not in self.speaker_emotion_map[speaker_id]:
                self.speaker_emotion_map[speaker_id][emotion] = []
                
            self.speaker_emotion_map[speaker_id][emotion].append(idx)
            self.speaker_to_emotions[speaker_id].add(emotion)
        
        # Find speakers with multiple emotions for cross-emotional training
        self.multi_emotion_speakers = {
            speaker: emotions for speaker, emotions in self.speaker_to_emotions.items()
            if len(emotions) >= 2
        }
        
        # Create SMART cross-emotional pairs (limited per speaker to avoid explosion)
        self.cross_emotional_pairs = []
        max_pairs_per_speaker_emotion = 3  # Limit to 3 pairs per speaker-emotion combo
        
        for speaker, emotions in self.multi_emotion_speakers.items():
            emotions_list = list(emotions)
            
            # Limit cross-emotional combinations to prevent explosion
            for ref_emotion in emotions_list:
                for target_emotion in emotions_list:
                    if ref_emotion != target_emotion:
                        ref_indices = self.speaker_emotion_map[speaker][ref_emotion]
                        target_indices = self.speaker_emotion_map[speaker][target_emotion]
                        
                        # Smart sampling: only take a few pairs per emotion combination
                        ref_sample = random.sample(ref_indices, min(max_pairs_per_speaker_emotion, len(ref_indices)))
                        target_sample = random.sample(target_indices, min(max_pairs_per_speaker_emotion, len(target_indices)))
                        
                        # Create limited pairs
                        for ref_idx in ref_sample:
                            for target_idx in target_sample:
                                self.cross_emotional_pairs.append((ref_idx, target_idx))
        
        print(f"Created {len(self.cross_emotional_pairs)} cross-emotional training pairs")
        print(f"Speakers with multiple emotions: {len(self.multi_emotion_speakers)}")
        print(f"Limited to {max_pairs_per_speaker_emotion} pairs per speaker-emotion combination")
        
        # Quick test mode support
        if self.config.get('quick_test', {}).get('enabled', False):
            max_pairs = self.config['quick_test'].get('max_samples', 1000)
            if len(self.cross_emotional_pairs) > max_pairs:
                self.cross_emotional_pairs = random.sample(self.cross_emotional_pairs, max_pairs)
                print(f"Quick test mode: Limited to {len(self.cross_emotional_pairs)} pairs")
    
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
        return len(self.cross_emotional_pairs)
    
    def __getitem__(self, idx):
        """Return cross-emotional pair for training."""
        ref_idx, target_idx = self.cross_emotional_pairs[idx]
        
        ref_sample = self.metadata[ref_idx]
        target_sample = self.metadata[target_idx]
        
        # Load reference audio (for speaker conditioning)
        ref_audio_path = self.data_dir / ref_sample['processed_audio_path']
        ref_audio = self._load_audio_safe(str(ref_audio_path), "reference audio")
        
        # Reference speaker file path (for XTTS conditioning)
        ref_speaker_path = str(ref_audio_path.resolve())
        
        # Target emotion values (what we want to generate) - already clamped to [0,1]
        target_valence = float(target_sample['valence'])
        target_arousal = float(target_sample['arousal'])
        
        # Target audio (ground truth for VAD comparison)
        target_audio_path = self.data_dir / target_sample['processed_audio_path']
        target_audio = self._load_audio_safe(str(target_audio_path), "target audio")
        
        return {
            # Generation inputs
            'text': target_sample['text'],  # Text to generate
            'speaker_ref': ref_speaker_path,  # Speaker reference for conditioning
            'target_valence': torch.tensor(target_valence, dtype=torch.float32),
            'target_arousal': torch.tensor(target_arousal, dtype=torch.float32),
            
            # Ground truth for comparison
            'target_audio': target_audio,
            'target_audio_path': str(target_audio_path.resolve()),
            
            # Metadata
            'ref_emotion': ref_sample['emotion_label'],
            'target_emotion': target_sample['emotion_label'],
            'speaker_id': ref_sample['speaker_id'],
            'ref_audio_path': str(ref_audio_path.resolve())
        }


def cross_emotional_collate_fn(batch):
    """Custom collate function for cross-emotional training."""
    
    # Generation inputs
    texts = [item['text'] for item in batch]
    speaker_refs = [item['speaker_ref'] for item in batch]
    target_valences = torch.stack([item['target_valence'] for item in batch])
    target_arousals = torch.stack([item['target_arousal'] for item in batch])
    
    # Handle variable-length target audio for VAD comparison
    target_audios = []
    audio_lengths = []
    
    for item in batch:
        target_audio = item['target_audio']
        target_audios.append(target_audio)
        audio_lengths.append(target_audio.shape[0])
    
    # Pad target audios to same length
    max_length = max(audio_lengths)
    padded_target_audios = []
    
    for audio in target_audios:
        if audio.shape[0] < max_length:
            padding = max_length - audio.shape[0]
            audio = torch.nn.functional.pad(audio, (0, padding))
        padded_target_audios.append(audio)
    
    target_audios_tensor = torch.stack(padded_target_audios)
    
    # Metadata
    target_audio_paths = [item['target_audio_path'] for item in batch]
    ref_emotions = [item['ref_emotion'] for item in batch]
    target_emotions = [item['target_emotion'] for item in batch]
    speaker_ids = [item['speaker_id'] for item in batch]
    ref_audio_paths = [item['ref_audio_path'] for item in batch]
    
    return {
        # Generation inputs
        'texts': texts,
        'speaker_refs': speaker_refs,
        'target_valences': target_valences,
        'target_arousals': target_arousals,
        
        # Ground truth for VAD comparison
        'target_audios': target_audios_tensor,
        'target_audio_paths': target_audio_paths,
        'audio_lengths': torch.tensor(audio_lengths),
        
        # Metadata
        'ref_emotions': ref_emotions,
        'target_emotions': target_emotions,
        'speaker_ids': speaker_ids,
        'ref_audio_paths': ref_audio_paths,
        'languages': ['en'] * len(batch)  # Add default language
    }