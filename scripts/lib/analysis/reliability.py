"""
Split-half reliability analysis for fMRI data.

Functions for calculating split-half reliability across voxels and subjects,
used to assess the consistency of neural responses.
"""
import numpy as np
from tqdm.autonotebook import tqdm
from ..utils import mycorr, identity


# Default preprocessing function (can be changed to demean, z_score, or identity)
preprocess = identity


def calculate_split_half_reliability(
    data,
    threshold: float = 0.1,
    mask_L_path: str = "fsaverage_L_mask.npy",
    mask_R_path: str = "fsaverage_R_mask.npy"
):
    """
    Calculate split-half reliability for each subject and hemisphere.
    
    Computes correlation between odd and even repetitions of stimuli presentations
    to assess voxel-wise reliability. Creates masks based on correlation threshold
    and anatomical constraints.
    
    Args:
        data: Dictionary with 'observed' key containing subject data
              Format: data['observed'][subject][hemisphere] with shape (reps, stimuli, voxels)
        threshold: Correlation threshold for including voxels (default: 0.1)
        mask_L_path: Path to left hemisphere anatomical mask
        mask_R_path: Path to right hemisphere anatomical mask
    
    Returns:
        Dictionary of masks for each subject and hemisphere, plus group-level masks
        Format: masks[subject][hemisphere] = boolean array
    
    Example:
        >>> from lib.analysis import calculate_split_half_reliability
        >>> from lib.data import load_data
        >>> 
        >>> data = load_data('output/observed_and_predicted_data.npz')
        >>> masks = calculate_split_half_reliability(data, threshold=0.1)
        >>> print(masks['group']['L'].sum(), "reliable voxels in left hemisphere")
    
    Notes:
        - Splits data into odd/even repetitions (split1: 0,2,4..., split2: 1,3,5...)
        - Computes correlation between averaged splits
        - Applies threshold and anatomical mask
        - Creates group-level mask by averaging across subjects
    """
    anatomical_mask_L = np.load(mask_L_path).astype(bool)
    anatomical_mask_R = np.load(mask_R_path).astype(bool)
    
    masks = {}
    corrs = {}
    
    for subject in tqdm(data['observed']):
        masks[subject] = {}
        corrs[subject] = {}
        
        for hemisphere in ['L', 'R']:
            n_reps = data['observed'][subject][hemisphere].shape[0]
            split1 = np.arange(0, n_reps, 2)
            split2 = np.arange(1, n_reps, 2)
            
            corr = mycorr(
                data['observed'][subject][hemisphere][split1].mean(0),
                data['observed'][subject][hemisphere][split2].mean(0)
            )
            
            corrs[subject][hemisphere] = corr.copy()
            mask = corr > threshold
            
            if hemisphere == 'L':
                mask = mask & ~np.isnan(anatomical_mask_L)
            else:
                mask = mask & ~np.isnan(anatomical_mask_R)
            
            masks[subject][hemisphere] = mask
            
            print(f"Subject {subject}, hemisphere {hemisphere}: "
                  f"{mask.sum()}/{len(mask)} "
                  f"({mask.sum()/len(mask)*100:.2f}%) voxels retained")
    
    # Add group level mask
    subjects = list(corrs.keys())
    avg_corr_L = []
    avg_corr_R = []
    
    for subject in subjects:
        avg_corr_L.append(corrs[subject]['L'])
        avg_corr_R.append(corrs[subject]['R'])
    
    avg_corr_L = np.stack(avg_corr_L, axis=0).mean(0)
    avg_corr_R = np.stack(avg_corr_R, axis=0).mean(0)
    corrs['group'] = {'L': avg_corr_L, 'R': avg_corr_R}
    
    mask_L = avg_corr_L > threshold
    mask_R = avg_corr_R > threshold
    mask_L = mask_L & ~np.isnan(anatomical_mask_L)
    mask_R = mask_R & ~np.isnan(anatomical_mask_R)
    masks['group'] = {'L': mask_L, 'R': mask_R}
    
    print(f"Group level mask for hemisphere L: "
          f"{mask_L.sum()}/{len(mask_L)} "
          f"({mask_L.sum()/len(mask_L)*100:.2f}%) voxels retained")
    print(f"Group level mask for hemisphere R: "
          f"{mask_R.sum()}/{len(mask_R)} "
          f"({mask_R.sum()/len(mask_R)*100:.2f}%) voxels retained")
    
    return masks


def calculate_subject_split_half_reliability(
    data,
    threshold: float,
    mask_L_path: str = "fsaverage_L_mask.npy",
    mask_R_path: str = "fsaverage_R_mask.npy"
):
    """
    Calculate split-half reliability across subjects.
    
    Splits subjects into two groups (odd/even indexed) and computes correlation
    between averaged responses. This assesses consistency at the group level.
    
    Args:
        data: Dictionary with 'observed' key containing subject data
        threshold: Correlation threshold for masking voxels
        mask_L_path: Path to left hemisphere anatomical mask
        mask_R_path: Path to right hemisphere anatomical mask
    
    Returns:
        Tuple of (mask_L, mask_R, corrs_L, corrs_R)
        - mask_L/R: Boolean arrays indicating reliable voxels
        - corrs_L/R: Correlation values for each voxel
    
    Example:
        >>> from lib.analysis import calculate_subject_split_half_reliability
        >>> from lib.data import load_data
        >>> 
        >>> data = load_data('output/observed_and_predicted_data.npz')
        >>> mask_L, mask_R, corrs_L, corrs_R = calculate_subject_split_half_reliability(
        ...     data, threshold=0.2
        ... )
        >>> print(f"Left hemisphere: {mask_L.sum()} reliable voxels")
    
    Notes:
        - Uses global `preprocess` function (default: identity)
        - Group 1: subjects at indices 0, 2, 4, ...
        - Group 2: subjects at indices 1, 3, 5, ...
        - Concatenates and averages data within each group
        - Computes correlation between groups
    """
    anatomical_mask_L = np.load(mask_L_path).astype(bool)
    anatomical_mask_R = np.load(mask_R_path).astype(bool)
    
    subjects = np.array(list(data['observed'].keys()))
    n_subjects = len(subjects)
    
    group1 = subjects[np.arange(0, n_subjects, 2)]
    group2 = subjects[np.arange(1, n_subjects, 2)]
    
    D1_left = np.concatenate(
        [preprocess(data['observed'][subject]['L']) for subject in tqdm(group1)], 0
    ).mean(0)
    D1_right = np.concatenate(
        [preprocess(data['observed'][subject]['R']) for subject in tqdm(group1)], 0
    ).mean(0)
    D2_left = np.concatenate(
        [preprocess(data['observed'][subject]['L']) for subject in tqdm(group2)], 0
    ).mean(0)
    D2_right = np.concatenate(
        [preprocess(data['observed'][subject]['R']) for subject in tqdm(group2)], 0
    ).mean(0)
    
    corrs_L = mycorr(D1_left, D2_left, 0)
    corrs_R = mycorr(D1_right, D2_right, 0)
    
    mask_L = corrs_L > threshold
    mask_R = corrs_R > threshold
    mask_L = mask_L & ~np.isnan(anatomical_mask_L)
    mask_R = mask_R & ~np.isnan(anatomical_mask_R)
    
    return mask_L, mask_R, corrs_L, corrs_R

def proj(v1,v2):
    return v2@(v2.T/np.linalg.norm(v2)**2)@v1
def reliability_b2021_single_voxel(v1,v2):
    v1 = v1.reshape(-1,1)
    v2 = v2.reshape(-1,1)
    r = 1 - (np.linalg.norm(v1-proj(v1,v2))**2)/(np.linalg.norm(v1)**2)
    return r

def reliability_b2021(D1,D2):
    out = np.zeros(D1.shape[1])
    for i in range(D1.shape[1]):
        out[i] = reliability_b2021_single_voxel(D1[:,i],D2[:,i])
    return out