#!/usr/bin/env python
"""
Fit encoding models to new subjects' fMRI data with stratified cross-validation.

This script trains robust or standard encoding models on new subjects' data,
using stratified splitting to ensure balanced stimulus categories in train/test sets.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import numpy as np
import torch
import einops
import librosa
from sklearn.model_selection import StratifiedKFold

from lib.encoding import EncodingModel, layers, save_h5
from lib.utils import rms_normalize
from lib.data import load_dict_h5


def load_first_stage_output(first_stage_path):
    """Load first-stage fMRI analysis output."""
    return load_dict_h5(first_stage_path)


def organize_observed_data(first_stage_output, subject_ids,reps="all"):
    """
    Organize observed fMRI data from first-stage output.
    
    Args:
        first_stage_output: Dictionary from first-stage analysis
        subject_ids: List of subject IDs to include
    
    Returns:
        Array of shape (stimuli, voxels) with concatenated hemispheres
    """
    assert reps in ["all","even","odd"], "reps must be 'all', 'even', or 'odd'"
    npd_observed = []

    for sid in subject_ids:
        # Determine number of sessions
        if sid in ['sub-MRI004', 'sub-MRI009']:
            n_sess = 6
        else:
            n_sess = 2
        print(f"{sid} has {n_sess} sessions")
        if reps == "all":
            reps_range = np.arange(1,n_sess+1)
        elif reps == "odd":
            reps_range = np.arange(1, n_sess+1, 2)
        else:  # even
            reps_range = np.arange(2, n_sess+1, 2)
        print(f"  Using sessions: {reps_range}")
        
        # Collect data for both hemispheres
        tmp = []
        for hemi in ['L', 'R']:
            tmp.append(
                np.stack([
                    first_stage_output[f'coefs_{hemi}'][sid][f'sess{sess}']
                    for sess in reps_range
                ])
            )
        
        # Stack: hemi x sessions x stim x voxels
        tmp = np.stack(tmp)
        print(f"  Data shape: {tmp.shape}")
        
        # Average across sessions: hemi x stim x voxels
        tmp = tmp.mean(axis=1)
        npd_observed.append(tmp)

    # Concatenate subjects along voxel dimension
    npd_observed = np.concatenate(npd_observed, -1)
    print(f"Combined shape: {npd_observed.shape}")

    # Rearrange to (stim, voxels) with both hemispheres concatenated
    npd_observed = einops.rearrange(
        npd_observed,
        'hemi stim voxels -> stim (hemi voxels)'
    )

    return npd_observed


def stratified_split_stimuli(stim_names, stim_categories_lookup, flip_splits=False):
    """
    Create stratified train/test split based on stimulus categories.
    
    Args:
        stim_names: Array of stimulus names
        stim_categories_lookup: Dictionary mapping stimulus names to categories
    
    Returns:
        Tuple of (train_indices, test_indices, reordered_names)
    """
    # Get categories for natural sounds (skip NPD)
    stim_categories = [
        stim_categories_lookup[sn]
        for sn in stim_names
        if "NPD" not in sn
    ]
    
    # Stratified 2-fold split
    kf = StratifiedKFold(n_splits=2, shuffle=False)
    if not flip_splits:
        test_indices, train_indices = next(kf.split(np.zeros(120), stim_categories))
    else:
        print("WARNING:Flipping train/test splits")
        train_indices, test_indices = next(kf.split(np.zeros(120), stim_categories))
    
    return train_indices, test_indices


def load_stimuli(
    stim_names,
    train_indices,
    test_indices,
    npd_stim_path,
    natural_stim_dir,
    rms_level=0.1
):
    """
    Load and normalize stimulus waveforms.

    Args:
        stim_names: Array of stimulus names
        train_indices: Indices for training stimuli
        test_indices: Indices for test stimuli
        npd_stim_path: Path to NPD stimuli array
        natural_stim_dir: Directory containing natural sound WAV files
        rms_level: RMS normalization level

    Returns:
        Tuple of (X_train, X_test, X_npd) as torch tensors
    """
    # Load NPD stimuli
    npd_stim_long = np.load(npd_stim_path)
    npd_stim = einops.rearrange(
        npd_stim_long,
        '(stim sample) -> stim sample',
        stim=60
    )

    X_train = []
    X_test = []
    X_npd = []
    
    # Load training stimuli
    for stim in stim_names[train_indices + 60]:
        fpath = os.path.join(natural_stim_dir, f"{stim}.wav")
        waveform, sr = librosa.load(fpath, sr=20000)
        waveform = torch.tensor(waveform).reshape(1, -1)
        waveform = rms_normalize(waveform, rms_level=rms_level)
        X_train.append(waveform)
    
    # Load test stimuli
    for stim in stim_names[test_indices + 60]:
        fpath = os.path.join(natural_stim_dir, f"{stim}.wav")
        waveform, sr = librosa.load(fpath, sr=20000)
        waveform = torch.tensor(waveform).reshape(1, -1)
        waveform = rms_normalize(waveform, rms_level=rms_level)
        X_test.append(waveform)
    
    # Load NPD stimuli
    for stim in stim_names[:60]:
        idx = int(stim.split('_')[-1])
        waveform = torch.tensor(npd_stim[idx]).reshape(1, -1)
        waveform = rms_normalize(waveform, rms_level=rms_level)
        X_npd.append(waveform)

    return (
        torch.stack(X_train),
        torch.stack(X_test),
        torch.stack(X_npd)
    )


def main():
    parser = argparse.ArgumentParser(
        description='Fit encoding models to new subjects fMRI data with stratified CV'
    )
    
    # Model configuration
    parser.add_argument(
        '--model_type',
        type=str,
        default='robust',
        choices=['robust', 'standard'],
        help='Type of encoding model to fit'
    )
    parser.add_argument(
        '--analysis_type',
        type=str,
        default='group',
        choices=['group', 'single_subject'],
        help='Type of analysis to perform'
    )
    parser.add_argument(
        '--subject_id',
        type=str,
        default=None,
        help='Subject ID for single subject analysis'
    )
    
    # Data paths
    parser.add_argument(
        '--first-stage-output',
        type=str,
        default='data/new-subjects-clean/first_stage_output_smooth3mm.h5',
        help='Path to first-stage analysis output'
    )
    parser.add_argument(
        '--npd-stimuli',
        type=str,
        default='output/npd_stimuli.npy',
        help='Path to NPD stimuli array'
    )
    parser.add_argument(
        '--natural-stimuli-dir',
        type=str,
        default='stimuli/natural_sounds',
        help='Directory containing natural sound WAV files'
    )
    parser.add_argument(
        '--category-lookup',
        type=str,
        default='components/stim_category_lookup.h5',
        help='Path to stimulus category lookup dictionary'
    )
    
    # Model parameters
    parser.add_argument(
        '--n-pcs',
        type=int,
        default=55,
        help='Number of principal components'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to use for model training'
    )

    parser.add_argument(
        '--reps',
        type=str,
        default='all',
        choices=['all', 'even', 'odd'],
        help='Which repetitions to use for training (all, even, or odd sessions)'
    )
    
    # Output
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models',
        help='Directory to save trained models'
    )

    parser.add_argument(
        '--flip-splits',
        action='store_true',
        default=False,
        help='Flip the train/test splits'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.analysis_type == "group" and args.subject_id is not None:
        raise ValueError("Subject ID cannot be specified for group analysis")
    if args.analysis_type == "single_subject" and args.subject_id is None:
        raise ValueError("Subject ID must be specified for single subject analysis")
    
    print(f"{'='*70}")
    print(f"Training {args.model_type} model ({args.analysis_type})")
    if args.subject_id:
        print(f"Subject: {args.subject_id}")
    print(f"{'='*70}")
    print()
    
    # Load first-stage output
    print("Loading first-stage output...")
    first_stage_output = load_first_stage_output(args.first_stage_output)
    
    # Determine subject IDs
    if args.analysis_type == "group":
        subject_ids = first_stage_output['ids']
    else:
        subject_ids = [args.subject_id]
    
    print(f"Subjects: {subject_ids}")
    print()
    
    # Organize observed data
    print("Organizing observed data...")
    observed_data = organize_observed_data(first_stage_output, subject_ids,reps=args.reps)
    print(f"Observed data shape: {observed_data.shape}")
    print()
    
    # Get stimulus names
    stim_names = np.array(first_stage_output['stim_names'])

    # Load category lookup and create stratified split
    print("Creating stratified train/test split...")
    stim_categories_lookup = load_dict_h5(args.category_lookup)
    train_indices, test_indices = stratified_split_stimuli(stim_names, stim_categories_lookup, flip_splits=args.flip_splits)
    
    print(f"Train indices: {train_indices} ({len(train_indices)} stimuli)")
    print(f"Test indices: {test_indices} ({len(test_indices)} stimuli)")
    print()
    
    # Save indices for reproducibility
    os.makedirs(args.output_dir, exist_ok=True)
    np.save(
        os.path.join(args.output_dir, "train_indices_new_subjects_stratified.npy"),
        train_indices
    )
    np.save(
        os.path.join(args.output_dir, "test_indices_new_subjects_stratified.npy"),
        test_indices
    )
    
    # Reorder observed data: NPD (60) + train + test
    observed_data = np.concatenate([
        observed_data[:60],
        observed_data[train_indices + 60],
        observed_data[test_indices + 60]
    ], 0)

    # Load stimuli
    print("Loading stimulus waveforms...")
    X_train, X_test, X_npd = load_stimuli(
        stim_names,
        train_indices,
        test_indices,
        args.npd_stimuli,
        args.natural_stimuli_dir,
        rms_level=0.1
    )

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"X_npd shape: {X_npd.shape}")
    print()
    
    # Create and train model
    print(f"Creating {args.model_type} encoding model...")
    model = EncodingModel(
        robust=(args.model_type == 'robust'),
        n_pcs=args.n_pcs,
        ridge_kwargs={'alphas': np.logspace(-8, 8, 500), 'fit_intercept': True},
        layers=layers,
        device=args.device
    )
    
    print("Training model...")
    print(f"  Training data: {observed_data[60:120].shape}")
    print(f"  Test data: {observed_data[120:180].shape}")
    model.fit(
        observed_data[60:120],  # Training responses
        X_train,                 # Training stimuli
        observed_data[120:180],  # Test responses
        X_test                   # Test stimuli
    )
    
    # Save model
    if args.subject_id is not None:
        output_filename = f"{args.model_type}_model_new_subjects_{args.subject_id}_smoothed_stratified.h5"
    else:
        output_filename = f"{args.model_type}_model_new_subjects_group_smoothed_stratified.h5"

    output_path = os.path.join(args.output_dir, output_filename)
    save_h5(model, output_path)
    
    print()
    print(f"{'='*70}")
    print(f"✓ Model saved to: {output_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

