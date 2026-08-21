"""
Utilities for sound synthesis.

This module provides layer mappings, padding utilities, and helper functions
specific to synthesis operations.
"""
import os
import torch
import numpy as np
import librosa
from tqdm.auto import tqdm


# Layer name to module path mapping for CochResNet50
layers = {
    'relu1': '1.layer1.2.relu',
    'relu2': '1.layer2.3.relu',
    'relu3': '1.layer3.5.relu',
    'relu4': '1.layer4.2.relu',
    'avgpool': '1.avgpool',
}

# Sampling rates for each layer's activations (in Hz)
layer_sampling_rates = {
    'relu1': 50,
    'relu2': 25,
    'relu3': 12,
    'relu4': 6,
    'avgpool': 1/3
}

# Padding required for each layer at different durations
layer_pads = {
    3: {
        'relu1': 150,
        'relu2': 75,
        'relu3': 36,
        'relu4': 18,
        'avgpool': 1
    }
}


def pad_to_dimension(tensor, target_length):
    """
    Pad tensor on the left to reach target length.
    
    Args:
        tensor: Input tensor to pad
        target_length: Target length for last dimension
    
    Returns:
        Padded tensor with last dimension of length target_length
    """
    current_length = tensor.size(-1)
    padding_length = target_length - current_length
    padded_tensor = torch.nn.functional.pad(
        tensor, (padding_length, 0, 0, 0), mode='constant', value=0
    )
    return padded_tensor


def get_keys(d):
    """
    Recursively extract all keys from nested dictionary.
    
    Args:
        d: Dictionary (possibly nested)
    
    Returns:
        List of all keys including from nested dictionaries
    """
    klist = []
    for k, v in d.items():
        klist.append(k)
        if isinstance(v, dict):
            klist.extend(get_keys(v))
    return klist


def generate_unique_filename(base_path, filename):
    """
    Generate unique filename by appending counter if file exists.
    
    If a file with the given name exists, appends _2, _3, etc. until
    a unique filename is found.
    
    Args:
        base_path: Directory where file will be saved
        filename: Original filename (with extension)
    
    Returns:
        Unique filename (basename only, not full path)
    """
    extension = os.path.splitext(filename)[1]
    base_filename = os.path.splitext(filename)[0]
    final_filename = filename
    counter = 2
    
    full_path = os.path.join(base_path, final_filename)
    
    while os.path.exists(full_path):
        final_filename = f"{base_filename}_{counter}{extension}"
        full_path = os.path.join(base_path, final_filename)
        counter += 1
    
    return final_filename


def get_huggingface_activations(model, processor, stimuli):
    """
    Extract activations from HuggingFace model for audio stimuli.
    
    Args:
        model: HuggingFace model with output_hidden_states support
        processor: HuggingFace processor for audio preprocessing
        stimuli: List of paths to audio files
    
    Returns:
        Dictionary mapping stimulus paths to stacked hidden states
    """
    activations = {}
    for stim in tqdm(stimuli):
        waveform, sr = librosa.load(stim, sr=16000)
        inputs = processor(waveform, return_tensors="pt", sampling_rate=16000).to('cuda')
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                out = model(**inputs, output_hidden_states=True)
        activations[stim] = torch.stack(out['hidden_states'])
    return activations

