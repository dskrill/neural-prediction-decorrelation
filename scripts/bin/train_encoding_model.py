#!/usr/bin/env python
"""
Train encoding models on original subjects' fMRI data.

This script trains encoding models using cochresnet50 features to predict
fMRI responses to natural sounds. It supports training on different stimulus
sets and repetitions.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import numpy as np
import torch
import einops
import librosa
from lib.encoding import EncodingModel, layers, save_h5
from lib.data import load_dict_h5


def process_rms_normalize_waveform(waveform, rms_level=0.1):
    """RMS normalize waveform to specified level."""
    out = waveform.clone()
    out -= out.mean()
    waveform_rms = torch.sqrt(torch.mean(torch.pow(out, 2)))
    out = out * rms_level / waveform_rms
    return out


def load_stimuli(stim_names, stimuli_dir):
    """Load audio stimuli from disk and normalize."""
    X = []
    for stim in stim_names:
        fpath = os.path.join(stimuli_dir, f"{stim}.wav")
        waveform, sr = librosa.load(fpath, sr=20000)
        waveform = torch.tensor(waveform).reshape(1, -1)
        waveform = process_rms_normalize_waveform(waveform, rms_level=0.1)
        X.append(waveform)
    
    if len(X) == 0:
        return None
    return torch.stack(X)


def main():
    parser = argparse.ArgumentParser(
        description='Train encoding models on fMRI data using cochresnet50 features.'
    )
    parser.add_argument(
        '--data', 
        type=str, 
        default="../data/original-subjects/natsounds.h5",
        help='Path to data file'
    )
    parser.add_argument(
        '--n_pcs', 
        type=int, 
        default=80,
        help='Number of principal components to use for encoding model'
    )
    parser.add_argument(
        '--stimuli', 
        type=str, 
        choices=['first', 'second', 'all'],
        default='first',
        help='Which stimulus set to train on'
    )
    parser.add_argument(
        '--rep', 
        type=str, 
        choices=['first', 'second', 'avg'],
        default='avg',
        help='Which repetition to use (first, second, or avg)'
    )
    parser.add_argument(
        '--robust',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Use robust model'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='encoding_model.h5',
        help='Path to output file'
    )
    parser.add_argument(
        '--device', 
        type=str, 
        default='cuda',
        help='Device to use (cuda or cpu)'
    )
    parser.add_argument(
        '--stimuli_dir',
        type=str,
        default='../stimuli/natural_sounds',
        help='Directory containing stimulus audio files'
    )

    parser.add_argument(
        '--hemi',
        type=str,
        choices=['L', 'R', 'both'],
        default='L',
        help='Hemisphere to train on (L, R, or both -- concatenates D_L and D_R along the voxel axis)'
    )

    args = parser.parse_args()

    if not args.robust:
        print("WARNING: Using non-robust model")

    # Load data
    print(f"Loading data from {args.data}")
    data = load_dict_h5(args.data)
    hemi = args.hemi
    if hemi == 'both':
        D = np.concatenate([data['D_L'], data['D_R']], axis=1)  # (165, 2316+2464, 3, 30)
    else:
        D = data[f'D_{hemi}']  # (165, 2316, 3, 30) # (stim, vox, rep, subject)
    
    # Select repetition
    if args.rep == 'first':
        D = D[:, :, 0]
    elif args.rep == 'second':
        D = D[:, :, 1]
    elif args.rep == 'avg': 
        D = np.nanmean(D, axis=2)
    else:
        raise ValueError(f"Rep {args.rep} not recognized")
    
    # Normalize D using training data statistics
    # Calculate SD by taking SD over sounds and averaging over voxels
    # D is stim x vox x subject
    print("Normalizing fMRI data using training set statistics")
    std_over_sounds = np.nanstd(D[..., 10:], 0, keepdims=True)  # 1 x vox x 20
    std_over_sounds_averaged_voxels = std_over_sounds.mean()  # mean over voxels AND subjects
    D = (D - np.nanmean(D[..., 10:])) / std_over_sounds_averaged_voxels
    
    # Split data according to stimulus set
    if args.stimuli == 'first':
        D_train = D[:55]
        D_val = D[55:110]
        D_test = D[110:]
        stim_names_train = data['stim_names'][:55]
        stim_names_val = data['stim_names'][55:110]
        stim_names_test = data['stim_names'][110:]
    elif args.stimuli == 'second':
        D_train = D[55:110]
        D_val = D[:55]
        D_test = D[110:]
        stim_names_train = data['stim_names'][55:110]
        stim_names_val = data['stim_names'][:55]
        stim_names_test = data['stim_names'][110:]
    elif args.stimuli == 'all':
        stim_names_train = data['stim_names']
        stim_names_test = []
        D_train = D
        D_val = None
        D_test = None
    
    # Reshape data: stim x (subject*voxel)
    D_train = einops.rearrange(D_train, 'stim voxel subject -> stim (subject voxel)')
    if D_test is not None:
        D_test = einops.rearrange(D_test, 'stim voxel subject -> stim (subject voxel)')
    
    # Load audio stimuli
    print("Loading audio stimuli...")
    X_train = load_stimuli(stim_names_train, args.stimuli_dir)
    X_val = load_stimuli(stim_names_val, args.stimuli_dir) if len(stim_names_val) > 0 else None
    X_test = load_stimuli(stim_names_test, args.stimuli_dir) if len(stim_names_test) > 0 else None
    
    print(f"Training set: {X_train.shape[0]} stimuli")
    if X_test is not None:
        print(f"Test set: {X_test.shape[0]} stimuli")
    
    # Create and train model
    print(f"Creating encoding model (robust={args.robust}, n_pcs={args.n_pcs})")
    model = EncodingModel(
        robust=args.robust,
        n_pcs=args.n_pcs,
        ridge_kwargs={'alphas': np.logspace(-8, 8, 500), 'fit_intercept': True},
        layers=layers,
        device=args.device
    )
    
    print("Training model...")
    model.fit(D_train, X_train, D_test, X_test)
    
    # Validation checks
    if np.isnan(model.best_predictions).sum() > 0:
        print("WARNING: NaNs in predictions")
    if model.best_predictions.var(0).sum() == 0:
        print("WARNING: Zero variance in predictions")
    
    # Save model
    print(f"Saving model to {args.output}")
    save_h5(model, args.output)
    print("Done!")


if __name__ == '__main__':
    main()

