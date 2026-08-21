"""
Preprocessing utilities for encoding models.

This module provides classes for feature preprocessing including
lagged features and standardization.
"""
import numpy as np
import torch


class Lag:
    """
    Create lagged versions of features.
    
    This class creates lagged copies of input features, useful for
    modeling temporal dependencies in time-series data.
    """
    
    def __init__(self, p):
        """
        Initialize Lag transformer.
        
        Args:
            p: Number of lags to create (0 means no lagging)
        """
        self.p = p

    def __call__(self, X, axis=-1, interleave=True):
        """
        Apply lagging to input features.
        
        Args:
            X: Input features (time x features) or (batch x time x features)
            axis: Axis along which to concatenate lags (default: -1)
            interleave: Whether to interleave lags (default: True)
        
        Returns:
            Lagged features. If interleave=True, shape is (..., features*(p+1)).
            If interleave=False, shape depends on axis.
        """
        if axis != -1 and interleave:
            raise ValueError("Can't interleave if axis is not -1")
        
        # Convert to torch if needed
        if isinstance(X, np.ndarray):
            X_ = torch.from_numpy(X).clone()
        else:
            X_ = X.clone()
        
        # Add batch dimension if needed
        if X_.ndim == 2:
            X_ = X_.unsqueeze(0)

        # Preallocate tensors
        out = [torch.zeros_like(X_) for lag in range(self.p + 1)]
        for lag in range(self.p + 1):
            new = torch.roll(X_, lag, dims=1)
            new[:, :lag, :] = 0
            out[lag][:, :, :] = new

        if interleave:
            out = torch.stack(out, dim=-1).reshape(X_.shape[0], X_.shape[1], -1)
        else:
            out = torch.cat(out, dim=axis)
        
        # Remove batch dimension if input was 2D
        if X.ndim == 2:
            out = out.squeeze(0)
        
        return out


class StandardScaler:
    """
    Standardize features by removing the mean and scaling to unit variance.
    
    This class is compatible with PyTorch tensors and provides
    fit/transform interface similar to sklearn.
    """

    def __init__(self, mean=None, std=None, epsilon=1e-7, dim=None, 
                 use_norm=False, mean_dim=None, std_dim=None):
        """
        Initialize StandardScaler.
        
        Args:
            mean: Pre-computed mean (optional, will be computed during fit)
            std: Pre-computed std (optional, will be computed during fit)
            epsilon: Small value to avoid division by zero (default: 1e-7)
            dim: Dimension along which to compute statistics
            use_norm: If True, use norm instead of std (default: False)
            mean_dim: Dimension for computing mean (overrides dim if specified)
            std_dim: Dimension for computing std (overrides dim if specified)
        """
        self.mean = mean
        self.std = std
        self.epsilon = epsilon
        self.dim = dim
        self.use_norm = use_norm
        self.mean_dim = mean_dim
        self.std_dim = std_dim

    def fit(self, values):
        """
        Compute mean and std from training data.
        
        Args:
            values: Training data tensor
        """
        self.mean = torch.mean(
            values, 
            dim=self.mean_dim if self.mean_dim is not None else self.dim, 
            keepdim=True
        )
        
        if self.use_norm:
            self.std = torch.norm(
                values, 
                dim=self.std_dim if self.std_dim is not None else self.dim, 
                keepdim=True
            )
        else:
            self.std = torch.std(
                values, 
                dim=self.std_dim if self.std_dim is not None else self.dim, 
                keepdim=True
            )

    def transform(self, values):
        """
        Standardize features using fitted statistics.
        
        Args:
            values: Data to transform
        
        Returns:
            Standardized data
        """
        return (values - self.mean) / (self.std + self.epsilon)

    def fit_transform(self, values):
        """
        Fit and transform in one step.
        
        Args:
            values: Data to fit and transform
        
        Returns:
            Standardized data
        """
        self.fit(values)
        return self.transform(values)

    def to(self, device):
        """
        Move scaler parameters to device.
        
        Args:
            device: Target device
        
        Returns:
            self
        """
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

