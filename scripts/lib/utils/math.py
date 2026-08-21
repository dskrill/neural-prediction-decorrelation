"""
Mathematical utility functions.

This module provides correlation, covariance, and other mathematical operations
used across the project.
"""
import numpy as np
import torch
import math
import scipy
from scipy import stats


def mycorr(A, B, axis=-1, device=None):
    """
    Compute correlation between A and B along specified axis.
    
    Handles both numpy arrays and torch tensors. Returns the same type as inputs.
    
    Args:
        A: First array/tensor (observations x features)
        B: Second array/tensor (observations x features)
        axis: Axis along which to compute correlation (default: -2)
        demean: Whether to demean data (default: False)
        demean_axis: Axis along which to demean (default: -1)
        device: Torch device to use for computation (optional)
    
    Returns:
        Correlation values (same type as input)
    """
    
    # Track input types
    A_was_numpy = isinstance(A, np.ndarray)
    B_was_numpy = isinstance(B, np.ndarray)
    
    # Convert to torch if needed
    if A_was_numpy:
        A = torch.from_numpy(A)
    if B_was_numpy:
        B = torch.from_numpy(B)
    
    # Move to device if specified
    if device is not None:
        A = A.to(device)
        B = B.to(device)
    

    # Compute correlation
    A = A - torch.nanmean(A, axis=axis, keepdims=True)
    B = B - torch.nanmean(B, axis=axis, keepdims=True)
    A = A / torch.sqrt(torch.nansum(torch.square(A), axis=axis, keepdims=True))
    B = B / torch.sqrt(torch.nansum(torch.square(B), axis=axis, keepdims=True))
    C = torch.nansum(A * B, axis=axis)
    
    # Convert back to numpy if inputs were numpy
    if A_was_numpy or B_was_numpy:
        C = C.cpu().numpy()
    
    return C


def degenerate_entries(*arrays, axis=0):
    """Positions where any array has zero variance or NaNs along `axis`.

    mycorr() computes correlation via nansum/nanmean, so a zero-variance
    (or all-NaN) input produces a 0/0 that silently resolves to an exact
    0.0 instead of the mathematically undefined NaN -- these entries are
    not "uncorrelated," they're degenerate and should be excluded/masked
    before use. `axis` should match the axis passed to the corresponding
    mycorr() call.

    Args:
        *arrays: one or more arrays that were (or will be) passed to mycorr()
        axis: axis along which mycorr() computes the correlation

    Returns:
        Boolean array, True where any input array is degenerate along axis
    """
    bad = None
    for a in arrays:
        v = np.var(a, axis=axis)
        b = ~(np.isfinite(v) & (v > 0))
        bad = b if bad is None else (bad | b)
    return bad


def mycov(A, B, axis=-2, device=None):
    """
    Compute covariance between A and B along specified axis.
    
    Handles both numpy arrays and torch tensors. Returns the same type as inputs.
    
    Args:
        A: First array/tensor (observations x features)
        B: Second array/tensor (observations x features)
        axis: Axis along which to compute covariance (default: -2)
        device: Torch device to use for computation (optional)
    
    Returns:
        Covariance values (same type as input)
    """
    # Track input types
    A_was_numpy = isinstance(A, np.ndarray)
    B_was_numpy = isinstance(B, np.ndarray)
    
    # Convert to torch if needed
    if A_was_numpy:
        A = torch.from_numpy(A)
    if B_was_numpy:
        B = torch.from_numpy(B)
    
    # Move to device if specified
    if device is not None:
        A = A.to(device)
        B = B.to(device)

    # Compute covariance
    A = A - torch.nanmean(A, axis=axis, keepdims=True)
    B = B - torch.nanmean(B, axis=axis, keepdims=True)
    C = torch.nanmean(A * B, axis=axis)
    
    # Convert back to numpy if inputs were numpy
    if A_was_numpy or B_was_numpy:
        C = C.cpu().numpy()
    
    return C


def demean(D):
    """
    Demean data along axis 1.
    
    Args:
        D: Data array (numpy)
    
    Returns:
        Demeaned data
    """
    return D - D.mean(1, keepdims=True)


def z_score(D):
    """
    Z-score normalize data along axis 1.
    
    Args:
        D: Data array (numpy)
    
    Returns:
        Z-scored data
    """
    return (D - D.mean(1, keepdims=True)) / D.std(1, keepdims=True)


def identity(D):
    """
    Identity function (returns input unchanged).
    
    Args:
        D: Data array
    
    Returns:
        D unchanged
    """
    return D


def lcm_float(a, b, precision):
    """
    Compute least common multiple of two floating point numbers.
    
    Args:
        a: First number
        b: Second number
        precision: Decimal precision to use
    
    Returns:
        Least common multiple as float
    """
    a = round(a, precision)
    b = round(b, precision)
    return math.lcm(int(a * 10**precision), int(b * 10**precision)) / 10**precision


def get_converted_sampling_rates(rate1, rate2, precision=3):
    """
    Convert two sampling rates to integer ratio.
    
    Finds the simplest integer ratio that represents the relationship
    between two sampling rates.
    
    Args:
        rate1: First sampling rate
        rate2: Second sampling rate
        precision: Decimal precision (default: 3)
    
    Returns:
        Tuple of (converted_rate1, converted_rate2) as integers
    """
    if isinstance(rate1, int) and isinstance(rate2, int):
        if (rate1 % rate2 == 0) or (rate2 % rate1 == 0):
            return int(rate1), int(rate2)
    
    lcm = lcm_float(rate1, rate2, precision)
    scale_factor = int(lcm // min(rate1, rate2))
    out1, out2 = int(rate1 * scale_factor), int(rate2 * scale_factor)
    
    if out1 % out2 == 0:
        return out1 // out2, 1
    elif out2 % out1 == 0:
        return 1, out2 // out1
    else:
        return out1, out2

def concordance_correlation_coefficient(x, y, alpha=0.05):
    """
    Calculate Lin's concordance correlation coefficient with confidence interval.
    
    Parameters
    ----------
    x, y : array-like
        Two measures to compare (must be same length)
    alpha : float
        Significance level for CI (default 0.05 for 95% CI)
    
    Returns
    -------
    dict with keys:
        'ccc': concordance correlation coefficient
        'ci_lower': lower bound of CI
        'ci_upper': upper bound of CI
        'pearson_r': Pearson correlation (for reference)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Remove any NaN pairs
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]
    
    n = len(x)
    
    # Means and variances
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    var_x = np.var(x, ddof=1)
    var_y = np.var(y, ddof=1)
    
    # Pearson correlation
    r = np.corrcoef(x, y)[0, 1]
    
    # CCC formula
    ccc = 2 * r * np.sqrt(var_x) * np.sqrt(var_y) / (var_x + var_y + (mean_x - mean_y)**2)
    
    # Confidence interval using Fisher's Z transformation
    # (asymptotic method from Lin 1989, 2000)
    z_ccc = np.arctanh(ccc)
    
    # Standard error (approximate)
    se_z = np.sqrt(1 / (n - 2))
    
    # CI in Z space
    z_critical = stats.norm.ppf(1 - alpha/2)
    z_lower = z_ccc - z_critical * se_z
    z_upper = z_ccc + z_critical * se_z
    
    # Transform back
    ci_lower = np.tanh(z_lower)
    ci_upper = np.tanh(z_upper)
    
    return {
        'ccc': ccc,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'pearson_r': r,
        'n': n
    }

def cohen_d(x,y):
    return (x.mean() - y.mean()) / np.sqrt((np.std(x, ddof=1) ** 2 + np.std(y, ddof=1) ** 2) / 2.0)