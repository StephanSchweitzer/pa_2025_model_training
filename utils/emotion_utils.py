import torch
import numpy as np
from typing import Union, Tuple, List, Dict


# Standard emotion mapping for categorical emotions
EMOTION_MAPPING = {
    0: "neutral",
    1: "happy", 
    2: "sad",
    3: "angry",
    4: "surprised",
    5: "fearful",
    6: "disgusted"
}

# Reverse mapping for emotion names to IDs
EMOTION_NAME_TO_ID = {v: k for k, v in EMOTION_MAPPING.items()}

# Default arousal-valence values for common emotions
DEFAULT_EMOTION_AV = {
    "neutral": [0.5, 0.5],
    "happy": [0.7, 0.8],
    "sad": [0.3, 0.2],
    "angry": [0.8, 0.2],
    "surprised": [0.8, 0.6],
    "fearful": [0.7, 0.2],
    "disgusted": [0.6, 0.3]
}


def validate_and_clamp_av(av_values: List[float], av_ranges: Dict[str, List[float]]) -> Tuple[float, float]:
    """
    Validate and clamp arousal-valence values to configured range.
    
    Args:
        av_values: List containing [arousal, valence] values
        av_ranges: Dictionary with 'arousal' and 'valence' keys mapping to [min, max] ranges
        
    Returns:
        tuple: (clamped_arousal, clamped_valence)
    """
    if len(av_values) < 2:
        raise ValueError("AV values must contain at least arousal and valence")
    
    arousal, valence = av_values[0], av_values[1]
    
    # Get ranges from config
    arousal_min, arousal_max = av_ranges['arousal']
    valence_min, valence_max = av_ranges['valence']
    
    # Clamp values to configured range
    arousal = max(arousal_min, min(arousal_max, float(arousal)))
    valence = max(valence_min, min(valence_max, float(valence)))
    
    return arousal, valence


def prepare_av_tensor(av_input: Union[List, Tuple, torch.Tensor, np.ndarray], 
                     device: torch.device, 
                     av_ranges: Dict[str, List[float]]) -> torch.Tensor:
    """
    Convert various AV input formats to tensor, ignoring dominance if present.
    
    Args:
        av_input: Arousal-valence input in various formats
        device: Target device for tensor
        av_ranges: Valid ranges for arousal and valence
        
    Returns:
        torch.Tensor: Properly formatted AV tensor [batch_size, 2]
    """
    
    # Handle different input types and extract arousal, valence
    if isinstance(av_input, (list, tuple)):
        if len(av_input) >= 3:
            # If 3 values provided (arousal, dominance, valence), extract arousal and valence
            av_values = [av_input[0], av_input[2]]  # arousal, valence (skip dominance)
        elif len(av_input) == 2:
            av_values = list(av_input)  # arousal, valence
        else:
            raise ValueError("AV input must have at least 2 values [arousal, valence]")
    elif isinstance(av_input, torch.Tensor):
        if av_input.shape[-1] >= 3:
            # Extract arousal and valence, skip dominance
            av_values = av_input[..., [0, 2]]  # arousal, valence
        elif av_input.shape[-1] == 2:
            av_values = av_input
        else:
            raise ValueError("AV tensor must have at least 2 dimensions")
    elif isinstance(av_input, np.ndarray):
        if av_input.shape[-1] >= 3:
            av_values = av_input[[0, 2]]  # arousal, valence
        elif av_input.shape[-1] == 2:
            av_values = av_input
        else:
            raise ValueError("AV array must have at least 2 dimensions")
    else:
        raise ValueError(f"Unsupported AV input type: {type(av_input)}")
    
    # Convert to tensor if not already
    if not isinstance(av_values, torch.Tensor):
        av_values = torch.tensor(av_values, dtype=torch.float32)
    
    # Ensure proper shape [batch_size, 2]
    if av_values.dim() == 1:
        av_values = av_values.unsqueeze(0)
    
    # Validate and clamp values using config ranges
    arousal, valence = validate_and_clamp_av(av_values[0].tolist(), av_ranges)
    av_values[0] = torch.tensor([arousal, valence], dtype=torch.float32)
    
    return av_values.to(device)


def prepare_emotion_tensor(emotion_id: Union[int, str, torch.Tensor], device: torch.device) -> torch.Tensor:
    """
    Prepare emotion tensor for processing.
    
    Args:
        emotion_id: Emotion identifier (int, string name, or tensor)
        device: Target device for tensor
        
    Returns:
        torch.Tensor: Emotion tensor on specified device
    """
    if isinstance(emotion_id, str):
        if emotion_id in EMOTION_NAME_TO_ID:
            emotion_id = EMOTION_NAME_TO_ID[emotion_id]
        else:
            raise ValueError(f"Unknown emotion name: {emotion_id}")
    
    if isinstance(emotion_id, torch.Tensor):
        return emotion_id.to(device)
    else:
        return torch.tensor(emotion_id, dtype=torch.long, device=device)


def emotion_name_to_av(emotion_name: str, custom_mapping: Dict[str, List[float]] = None) -> List[float]:
    """
    Convert emotion name to arousal-valence values.
    
    Args:
        emotion_name: Name of emotion
        custom_mapping: Optional custom emotion-to-AV mapping
        
    Returns:
        list: [arousal, valence] values
    """
    mapping = custom_mapping if custom_mapping is not None else DEFAULT_EMOTION_AV
    
    if emotion_name not in mapping:
        if emotion_name in EMOTION_NAME_TO_ID:
            # Use neutral as fallback
            return mapping.get("neutral", [0.5, 0.5])
        else:
            raise ValueError(f"Unknown emotion: {emotion_name}")
    
    return mapping[emotion_name]


def av_to_emotion_name(arousal: float, valence: float, threshold: float = 0.2) -> str:
    """
    Convert arousal-valence values to closest emotion name.
    
    Args:
        arousal: Arousal value [0, 1]
        valence: Valence value [0, 1]
        threshold: Distance threshold for emotion classification
        
    Returns:
        str: Closest emotion name
    """
    min_distance = float('inf')
    closest_emotion = "neutral"
    
    for emotion_name, (emo_arousal, emo_valence) in DEFAULT_EMOTION_AV.items():
        distance = np.sqrt((arousal - emo_arousal)**2 + (valence - emo_valence)**2)
        if distance < min_distance:
            min_distance = distance
            closest_emotion = emotion_name
    
    # If distance is too large, return neutral
    if min_distance > threshold:
        return "neutral"
    
    return closest_emotion


def interpolate_av_values(av1: List[float], av2: List[float], alpha: float) -> List[float]:
    """
    Interpolate between two arousal-valence value pairs.
    
    Args:
        av1: First AV pair [arousal, valence]
        av2: Second AV pair [arousal, valence]
        alpha: Interpolation factor [0, 1]
        
    Returns:
        list: Interpolated [arousal, valence] values
    """
    if not (0 <= alpha <= 1):
        raise ValueError("Alpha must be between 0 and 1")
    
    arousal = av1[0] * (1 - alpha) + av2[0] * alpha
    valence = av1[1] * (1 - alpha) + av2[1] * alpha
    
    return [arousal, valence]


def generate_av_grid(arousal_steps: int = 5, valence_steps: int = 5, 
                    av_ranges: Dict[str, List[float]] = None) -> List[List[float]]:
    """
    Generate a grid of arousal-valence values for systematic evaluation.
    
    Args:
        arousal_steps: Number of steps along arousal axis
        valence_steps: Number of steps along valence axis
        av_ranges: Optional custom ranges for arousal and valence
        
    Returns:
        list: List of [arousal, valence] pairs covering the grid
    """
    if av_ranges is None:
        av_ranges = {"arousal": [0.0, 1.0], "valence": [0.0, 1.0]}
    
    arousal_min, arousal_max = av_ranges["arousal"]
    valence_min, valence_max = av_ranges["valence"]
    
    arousal_values = np.linspace(arousal_min, arousal_max, arousal_steps)
    valence_values = np.linspace(valence_min, valence_max, valence_steps)
    
    av_grid = []
    for arousal in arousal_values:
        for valence in valence_values:
            av_grid.append([float(arousal), float(valence)])
    
    return av_grid


def validate_emotion_config(emotions: List[str], av_ranges: Dict[str, List[float]]) -> List[str]:
    """
    Validate emotion configuration for consistency.
    
    Args:
        emotions: List of emotion names
        av_ranges: Arousal-valence ranges
        
    Returns:
        list: List of validation warnings
    """
    warnings = []
    
    # Check if all emotions are known
    for emotion in emotions:
        if emotion not in DEFAULT_EMOTION_AV and emotion not in EMOTION_NAME_TO_ID:
            warnings.append(f"Unknown emotion: {emotion}")
    
    # Validate AV ranges
    for av_type, range_vals in av_ranges.items():
        if av_type not in ["arousal", "valence"]:
            warnings.append(f"Unknown AV type: {av_type}")
        elif len(range_vals) != 2:
            warnings.append(f"Invalid range for {av_type}: expected [min, max]")
        elif range_vals[0] >= range_vals[1]:
            warnings.append(f"Invalid range for {av_type}: min >= max")
    
    return warnings


def create_emotion_batch(emotions: List[str], av_ranges: Dict[str, List[float]], 
                        device: torch.device) -> torch.Tensor:
    """
    Create a batch of AV tensors from emotion names.
    
    Args:
        emotions: List of emotion names
        av_ranges: Arousal-valence ranges for validation
        device: Target device
        
    Returns:
        torch.Tensor: Batch of AV tensors [batch_size, 2]
    """
    av_batch = []
    
    for emotion in emotions:
        av_values = emotion_name_to_av(emotion)
        av_tensor = prepare_av_tensor(av_values, device, av_ranges)
        av_batch.append(av_tensor)
    
    return torch.cat(av_batch, dim=0)


def get_emotion_statistics(av_values_list: List[List[float]]) -> Dict[str, float]:
    """
    Calculate statistics for a collection of AV values.
    
    Args:
        av_values_list: List of [arousal, valence] pairs
        
    Returns:
        dict: Statistics including mean, std, min, max for arousal and valence
    """
    if not av_values_list:
        return {}
    
    av_array = np.array(av_values_list)
    arousal_values = av_array[:, 0]
    valence_values = av_array[:, 1]
    
    stats = {
        "arousal_mean": float(np.mean(arousal_values)),
        "arousal_std": float(np.std(arousal_values)),
        "arousal_min": float(np.min(arousal_values)),
        "arousal_max": float(np.max(arousal_values)),
        "valence_mean": float(np.mean(valence_values)),
        "valence_std": float(np.std(valence_values)),
        "valence_min": float(np.min(valence_values)),
        "valence_max": float(np.max(valence_values)),
        "total_samples": len(av_values_list)
    }
    
    return stats