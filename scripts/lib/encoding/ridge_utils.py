"""
Ridge regression utilities.

This module provides helper functions for ridge regression,
particularly for checking selected regularization parameters.
"""
import numpy as np


def check_ridge_alphas(ridge):
    """
    Check if selected ridge alphas are at the boundaries of the search range.
    
    This function helps diagnose whether the alpha range needs to be expanded
    by checking if many voxels selected the minimum or maximum alpha value.
    
    Args:
        ridge: Fitted RidgeCV model with best_alphas_ attribute
    
    Prints:
        Information about how many alphas are at the min/max of the range
    """
    min_alpha = ridge.alphas.min()
    max_alpha = ridge.alphas.max()
    where_min = np.where(np.isclose(ridge.best_alphas_, min_alpha))[0]
    where_max = np.where(np.isclose(ridge.best_alphas_, max_alpha))[0]
    
    if len(where_min) > 0:
        print(f'{len(where_min)} selected alpha at minimum of range ({where_min})')
    else:
        print(f"No alpha at minimum of range (minimum selected: {ridge.best_alphas_.min()})")
    
    if len(where_max) > 0:
        print(f'{len(where_max)} selected alpha at maximum of range ({where_max})')
    else:
        print(f"No alpha at maximum of range (maximum selected: {ridge.best_alphas_.max()})")

