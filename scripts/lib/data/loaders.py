"""
Data loading utilities for fMRI analysis.

Functions for loading collated NPZ data files and reconstructing data structures.
"""
import sys
import numpy as np
from typing import Dict, List, Optional, Any
from .io_h5 import load_dict_h5

# sys.path.extend(['../../scripts/', '../../data/', '../../masks/', '../../components/'])


def load_data(
    filename: str = 'observed_and_predicted_data_new_subjects_smoothed_stratified.npz',
    components: Optional[str] = "components_reordered_stratified.h5",
    include_prediction_reps: bool = False
) -> Dict[str, Any]:
    """
    Load collated fMRI data from NPZ format.
    
    This function loads observed fMRI data, model predictions, and stimulus waveforms
    from a standardized NPZ file format. Optionally loads component analysis results.
    
    Args:
        filename: Path to NPZ file containing collated data
        components: Path to components HDF5 file, or None to skip loading
    
    Returns:
        Dictionary containing:
            - 'subjects': List of subject IDs
            - 'stimuli_names': List of stimulus names
            - 'observed': Dict of observed data by subject and hemisphere
            - 'predictions': Dict of model predictions by subject, model, and hemisphere
            - 'stimuli_waveforms': Dict of stimulus waveforms by name
            - 'components': Components data if provided, otherwise None
    
    Example:
        >>> data = load_data('output/observed_and_predicted_data.npz')
        >>> print(data['subjects'])
        ['sub-MRI004', 'sub-MRI005', ...]
        >>> print(data['observed']['sub-MRI004']['L'].shape)
        (6, 180, 163842)  # sessions x stimuli x voxels
    """
    # Load components if provided
    components_data = None
    if components is not None:
        components_data = load_dict_h5(components)
    
    npz_file = np.load(filename)
    
    # Reconstruct the original structure
    subjects = npz_file['subjects'].tolist()
    stimuli_names = npz_file['stimuli_names'].tolist()
    n_sessions = npz_file['n_sessions']
    
    # Reconstruct observed data
    observed_data = {}
    for i, sid in enumerate(subjects):
        observed_data[sid] = {
            'n_sessions': int(n_sessions[i]),
            'L': npz_file.get(f'observed_{sid}_L', np.array([])),
            'R': npz_file.get(f'observed_{sid}_R', np.array([]))
        }
    
    # Reconstruct predictions
    prediction_data = {}
    for sid in subjects:
        prediction_data[sid] = {
            'robust': {
                'L': npz_file[f'robust_{sid}_L'],
                'R': npz_file[f'robust_{sid}_R']
            },
            'standard': {
                'L': npz_file[f'standard_{sid}_L'],
                'R': npz_file[f'standard_{sid}_R']
            }
        }
        if include_prediction_reps:
            #e.g. standard_odd_sub-MRI007_L
            prediction_data[sid]['robust_even'] = {
                'L': npz_file[f'robust_even_{sid}_L'],
                'R': npz_file[f'robust_even_{sid}_R']
            }
            prediction_data[sid]['robust_odd'] = {
                'L': npz_file[f'robust_odd_{sid}_L'],
                'R': npz_file[f'robust_odd_{sid}_R']
            }
            prediction_data[sid]['standard_even'] = {
                'L': npz_file[f'standard_even_{sid}_L'],
                'R': npz_file[f'standard_even_{sid}_R']
            }
            prediction_data[sid]['standard_odd'] = {
                'L': npz_file[f'standard_odd_{sid}_L'],
                'R': npz_file[f'standard_odd_{sid}_R']
            }
    # Reconstruct stimuli waveforms
    stimuli_waveforms = {}
    for i, name in enumerate(stimuli_names):
        key = f'stimulus_{i:03d}_{name}'
        if key in npz_file:
            stimuli_waveforms[name] = npz_file[key]
    
    return {
        'subjects': subjects,
        'stimuli_names': stimuli_names,
        'observed': observed_data,
        'predictions': prediction_data,
        'stimuli_waveforms': stimuli_waveforms,
        'components': components_data
    }


def load_original_subjects_data(
    filename: str = 'observed_and_predicted_data_original_subjects.npz'
) -> Dict[str, Any]:
    """
    Load data from NPZ format for original subjects.
    
    Original subjects have 4 encoding models (2 repetitions x 2 training schemes)
    compared to 2 models for new subjects.
    
    Args:
        filename: Path to NPZ file
    
    Returns:
        Dictionary with same structure as load_data() but with 4 model types:
        'robust1', 'standard1', 'robust2', 'standard2'
    
    Example:
        >>> data = load_original_subjects_data()
        >>> print(data['subjects'])
        ['S1', 'S2', ..., 'S30']
    """
    npz_file = np.load(filename)
    
    # Reconstruct metadata
    subjects = npz_file['subjects'].tolist()
    stimuli_names = npz_file['stimuli_names'].tolist()
    n_sessions = npz_file['n_sessions']
    
    # Reconstruct observed data
    observed_data = {}
    for i, sid in enumerate(subjects):
        observed_data[sid] = {
            'n_sessions': int(n_sessions[i]),
            'data_L': npz_file[f'observed_{sid}_L'],
            'data_R': npz_file[f'observed_{sid}_R']
        }
    
    # Reconstruct predictions (4 models)
    prediction_data = {}
    for sid in subjects:
        prediction_data[sid] = {
            'robust1_L': npz_file[f'robust1_{sid}_L'],
            'robust1_R': npz_file[f'robust1_{sid}_R'],
            'standard1_L': npz_file[f'standard1_{sid}_L'],
            'standard1_R': npz_file[f'standard1_{sid}_R'],
            'robust2_L': npz_file[f'robust2_{sid}_L'],
            'robust2_R': npz_file[f'robust2_{sid}_R'],
            'standard2_L': npz_file[f'standard2_{sid}_L'],
            'standard2_R': npz_file[f'standard2_{sid}_R']
        }
    
    # Reconstruct stimuli waveforms
    stimuli_waveforms = {}
    for i, name in enumerate(stimuli_names):
        key = f'stimulus_{i:03d}_{name}'
        if key in npz_file:
            stimuli_waveforms[name] = npz_file[key]
    
    return {
        'subjects': subjects,
        'stimuli_names': stimuli_names,
        'observed': observed_data,
        'predictions': prediction_data,
        'stimuli_waveforms': stimuli_waveforms
    }


def get_subject_data(
    data: Dict[str, Any], 
    subject_id: str, 
    data_type: str = 'observed',
    model_type: str = 'robust',
    hemisphere: Optional[str] = None
) -> np.ndarray:
    """
    Extract data for a specific subject.
    
    Args:
        data: Data dict from load_data() or load_original_subjects_data()
        subject_id: Subject ID string (e.g., 'sub-MRI004' or 'S1')
        data_type: 'observed' or 'predictions'
        model_type: Model name (e.g., 'robust', 'standard', 'robust1', etc.)
        hemisphere: 'L', 'R', or None for both/all
    
    Returns:
        numpy array with requested data
    
    Example:
        >>> data = load_data()
        >>> obs = get_subject_data(data, 'sub-MRI004', 'observed', hemisphere='L')
        >>> pred = get_subject_data(data, 'sub-MRI004', 'predictions', 'robust', 'L')
    """
    if data_type == 'observed':
        subject_data = data['observed'][subject_id]
        if hemisphere is None:
            # Return both hemispheres if structure supports it
            if 'data' in subject_data:
                return subject_data['data']
            else:
                return {
                    'L': subject_data.get('L', np.array([])),
                    'R': subject_data.get('R', np.array([]))
                }
        else:
            return subject_data.get(hemisphere, subject_data.get('data', np.array([])))
    else:  # predictions
        pred_data = data['predictions'][subject_id][model_type]
        if hemisphere is None:
            return pred_data
        else:
            return pred_data.get(hemisphere, pred_data)

