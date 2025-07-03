"""
Audio Processing Utilities

Handles audio tensor operations, file I/O, and conditioning latent extraction.
Clean separation of audio-specific functionality.
"""

import os
import tempfile
from contextlib import contextmanager
import torch
import torchaudio


@contextmanager
def tensor_to_temp_file(audio_tensor, sample_rate):
    """
    Context manager to save audio tensor to temporary file.
    
    Args:
        audio_tensor: Audio tensor (1D or 2D)
        sample_rate: Sample rate for audio file
        
    Yields:
        str: Path to temporary audio file
    """
    # Ensure proper shape for saving (add channel dimension if needed)
    audio_to_save = audio_tensor.unsqueeze(0) if audio_tensor.dim() == 1 else audio_tensor
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        temp_path = tmp_file.name
        
    try:
        torchaudio.save(temp_path, audio_to_save, sample_rate)
        yield temp_path
    finally:
        try:
            os.unlink(temp_path)
        except:
            pass


def prepare_audio_tensor(audio_tensor):
    """
    Standardize audio tensor format for processing.
    
    Args:
        audio_tensor: Input audio tensor of various shapes
        
    Returns:
        torch.Tensor: 1D audio tensor on CPU
    """
    if audio_tensor.dim() == 3:
        audio_tensor = audio_tensor.squeeze(0)
    if audio_tensor.dim() == 2:
        audio_tensor = audio_tensor.squeeze(0)
    
    if audio_tensor.dim() != 1:
        raise ValueError(f"Expected 1D audio tensor after processing, got {audio_tensor.dim()}D")
    
    return audio_tensor.cpu()


def get_base_conditioning_latents(xtts_model, audio_input, training=False, sample_rate=22050):
    """
    Extract conditioning latents from audio input using XTTS model.
    
    Args:
        xtts_model: Loaded XTTS model
        audio_input: Audio tensor, file path, or list of paths
        training: Whether in training mode (affects gradient computation)
        sample_rate: Sample rate for audio processing
        
    Returns:
        tuple: (gpt_cond_latent, speaker_embedding)
    """
    
    # Handle different input types
    if isinstance(audio_input, torch.Tensor):
        with tensor_to_temp_file(audio_input, sample_rate) as temp_path:
            audio_path = [temp_path]
    elif isinstance(audio_input, (str, list)):
        audio_path = audio_input if isinstance(audio_input, list) else [audio_input]
    else:
        raise ValueError(f"Unsupported audio_input type: {type(audio_input)}")
    
    # Extract conditioning latents with appropriate gradient handling
    if training:
        return xtts_model.get_conditioning_latents(audio_path=audio_path)
    else:
        with torch.no_grad():
            return xtts_model.get_conditioning_latents(audio_path=audio_path)


def validate_audio_input(audio_input, expected_sample_rate=22050):
    """
    Validate audio input format and properties.
    
    Args:
        audio_input: Audio tensor or file path
        expected_sample_rate: Expected sample rate
        
    Returns:
        dict: Validation results
    """
    validation_info = {
        "valid": True,
        "messages": [],
        "sample_rate": None,
        "duration": None,
        "channels": None
    }
    
    try:
        if isinstance(audio_input, torch.Tensor):
            validation_info["sample_rate"] = expected_sample_rate  # Assumed
            validation_info["duration"] = audio_input.shape[-1] / expected_sample_rate
            validation_info["channels"] = 1 if audio_input.dim() == 1 else audio_input.shape[0]
            
            if audio_input.dim() > 2:
                validation_info["valid"] = False
                validation_info["messages"].append("Audio tensor has too many dimensions")
                
        elif isinstance(audio_input, str):
            if not os.path.exists(audio_input):
                validation_info["valid"] = False
                validation_info["messages"].append(f"Audio file does not exist: {audio_input}")
            else:
                # Try to load audio info
                info = torchaudio.info(audio_input)
                validation_info["sample_rate"] = info.sample_rate
                validation_info["duration"] = info.num_frames / info.sample_rate
                validation_info["channels"] = info.num_channels
                
                if info.sample_rate != expected_sample_rate:
                    validation_info["messages"].append(
                        f"Sample rate mismatch: got {info.sample_rate}, expected {expected_sample_rate}"
                    )
                    
        else:
            validation_info["valid"] = False
            validation_info["messages"].append(f"Unsupported audio input type: {type(audio_input)}")
            
    except Exception as e:
        validation_info["valid"] = False
        validation_info["messages"].append(f"Validation error: {e}")
    
    return validation_info


def convert_audio_format(audio_tensor, target_sample_rate=None, target_channels=1):
    """
    Convert audio tensor to target format.
    
    Args:
        audio_tensor: Input audio tensor
        target_sample_rate: Target sample rate (if None, no resampling)
        target_channels: Target number of channels
        
    Returns:
        torch.Tensor: Converted audio tensor
    """
    processed_audio = audio_tensor
    
    # Handle channel conversion
    if processed_audio.dim() == 2 and processed_audio.shape[0] > target_channels:
        if target_channels == 1:
            # Convert to mono by averaging channels
            processed_audio = processed_audio.mean(dim=0, keepdim=False)
        else:
            # Take first N channels
            processed_audio = processed_audio[:target_channels]
    
    # Handle resampling (if needed and torchaudio supports it)
    if target_sample_rate is not None:
        # Note: This would require knowing the original sample rate
        # For now, just return as-is and let caller handle resampling
        pass
    
    return processed_audio


def batch_audio_processing(audio_list, sample_rate=22050, max_length=None):
    """
    Process a batch of audio inputs for consistent formatting.
    
    Args:
        audio_list: List of audio tensors or file paths
        sample_rate: Target sample rate
        max_length: Maximum length in samples (for padding/truncation)
        
    Returns:
        list: Processed audio tensors
    """
    processed_audios = []
    
    for audio in audio_list:
        if isinstance(audio, str):
            # Load audio file
            waveform, sr = torchaudio.load(audio)
            if sr != sample_rate:
                # Resample if needed
                resampler = torchaudio.transforms.Resample(sr, sample_rate)
                waveform = resampler(waveform)
            audio_tensor = waveform.squeeze(0)  # Remove channel dimension
        else:
            audio_tensor = prepare_audio_tensor(audio)
        
        # Apply length constraint if specified
        if max_length is not None:
            if audio_tensor.shape[-1] > max_length:
                audio_tensor = audio_tensor[:max_length]
            elif audio_tensor.shape[-1] < max_length:
                # Pad with zeros
                pad_length = max_length - audio_tensor.shape[-1]
                audio_tensor = torch.nn.functional.pad(audio_tensor, (0, pad_length))
        
        processed_audios.append(audio_tensor)
    
    return processed_audios