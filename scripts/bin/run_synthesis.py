#!/usr/bin/env python
"""
Run sound synthesis to generate NPD stimuli.

This script uses encoding models to synthesize audio that has matched variances
but decorrelated predictions between robust and standard models.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
from datetime import datetime
import numpy as np
import torch
import einops

from lib.synthesis import BestLayerSynthesizer
from lib.encoding import EncodingModel, load_h5


def main():
    parser = argparse.ArgumentParser(
        description='Synthesize audio with matched variances and decorrelated predictions'
    )
    parser.add_argument(
        '--variance_multiplier', 
        type=int, 
        default=1,
        help='Multiplier for variance scaling'
    )
    parser.add_argument(
        '--training_set_10', 
        action='store_true',
        help='Use the first 10 subjects for training'
    )
    parser.add_argument(
        '--models_dir',
        type=str,
        default='../models',
        help='Directory containing encoding model files'
    )
    parser.add_argument(
        '--masks_dir',
        type=str,
        default='../masks',
        help='Directory containing subject masks'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='.',
        help='Directory for output files'
    )
    parser.add_argument(
        '--mask_filename',
        type=str,
        default='original_subjects_reliability_40_mask.npy',
        help=(
            'Reliability mask filename within --masks_dir. Use '
            'original_subjects_reliability_40_mask.npy for left-hemisphere-only '
            'models, or original_subjects_reliability_40_mask_both_hemis.npy for '
            'models trained with --hemi both.'
        )
    )
    args = parser.parse_args()
    
    variance_multiplier = args.variance_multiplier
    training_set = '10' if args.training_set_10 else '20'
    today = datetime.today().strftime('%m%d%Y')
    
    # Load encoding models
    print("Loading encoding models...")
    robust_model_path = os.path.join(
        args.models_dir,
        'natsounds_fmri_encoding_model_stimuli_first_rep_avg_robust.h5'
    )
    standard_model_path = os.path.join(
        args.models_dir,
        'natsounds_fmri_encoding_model_stimuli_first_rep_avg_standard.h5'
    )

    robust_model = load_h5(robust_model_path)
    standard_model = load_h5(standard_model_path)

    # Create synthesizer
    print("Initializing synthesizer...")
    checkpoint_dir = os.path.join(
        args.output_dir,
        f'../checkpoints/adversarially_robust_standard_inflate_variance_{variance_multiplier}_training_{training_set}_waveform_{today}'
    )
    log_dir = os.path.join(
        args.output_dir,
        f'../logs/adversarially_robust_standard_inflate_variance_{variance_multiplier}_training_{training_set}_waveform_{today}'
    )

    synth = BestLayerSynthesizer(
        encoding_models={
            'robust': robust_model,
            'standard': standard_model
        },
        batch_size=60,
        window=2000,
        max_iters=10000,
        monitor=20,
        fixup_every=100,
        match_variances=True,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=200,
        checkpoint_name='checkpoint.npy',
        log_dir=log_dir,
        device={'robust': 'cuda:0', 'standard': 'cuda:1'}
    )
    
    # Load subject masks
    print("Loading subject masks...")
    # Per-voxel reliability (r > 0.4) mask, tiled across all 30 subjects.
    mask_path = os.path.join(args.masks_dir, args.mask_filename)
    train_mask = np.load(mask_path).astype(bool)
    test_mask = train_mask.copy()

    # 30 subjects total: first 10 are NH2015, last 20 are B2021. Voxels-per-subject
    # is derived from the mask itself, so this works for both single-hemisphere
    # masks (2316 or 2464 voxels/subject) and combined-hemisphere masks (4780).
    n_voxels_per_subject = len(train_mask) // 30
    if args.training_set_10:
        train_mask[n_voxels_per_subject*10:] = False
        test_mask[:n_voxels_per_subject*10] = False
    else:
        train_mask[:n_voxels_per_subject*10] = False
        test_mask[n_voxels_per_subject*10:] = False
    
    # Run synthesis
    print("Running synthesis...")
    synth.fit(
        train_indices=train_mask,
        test_indices=test_mask,
        lr=.001,
        c=1,
        variance_scale=variance_multiplier,
        maximize_variance=False,
        X=None
    )
    
    # Save results
    print("Saving results...")
    X_ = synth.X.detach().cpu().numpy()
    X_long = einops.rearrange(X_, 'b s -> (b s)')
    output_filename = f'X_long_inflate_variance_{variance_multiplier}_training_{training_set}_waveform_{today}.npy'
    output_path = os.path.join(args.output_dir, output_filename)
    np.save(output_path, X_long)
    print(f"Saved synthesized audio to {output_path}")
    print("Done!")


if __name__ == '__main__':
    main()
