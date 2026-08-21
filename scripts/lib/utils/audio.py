"""
Audio processing utility functions.

This module provides audio normalization, fading, and resampling utilities.
"""
import torch
import torchaudio
from torchaudio.functional.functional import _apply_sinc_resample_kernel


def rms_normalize(waveform, rms_level=0.1, dim=1):
    """
    RMS normalize waveform to specified level.
    
    Unified function that handles both batch and single waveforms.
    
    Args:
        waveform: Audio waveform tensor (..., samples) or (batch, samples)
        rms_level: Target RMS level (default: 0.1)
        dim: Dimension along which to compute RMS (default: 1 for batch dim)
    
    Returns:
        RMS-normalized waveform
    """
    out = waveform - waveform.mean(dim=dim, keepdim=True)
    waveform_rms = torch.sqrt(torch.mean(torch.pow(out, 2), dim=dim, keepdim=True))
    out = out * rms_level / waveform_rms
    return out


def rms(X, dim=-1):
    """
    Compute root mean square of signal.
    
    Args:
        X: Input tensor
        dim: Dimension along which to compute RMS (default: -1)
    
    Returns:
        RMS value(s)
    """
    return torch.sqrt(torch.mean(torch.pow(X, 2), dim=dim, keepdims=True))


def linear_ramp(total_length, ramp_length, epsilon=1e-6):
    """
    Create linear fade-in/fade-out ramp.
    
    Creates a window that linearly ramps from epsilon to 1 at the start,
    stays at 1 in the middle, and linearly ramps from 1 to epsilon at the end.
    
    Args:
        total_length: Total length of ramp
        ramp_length: Length of fade-in and fade-out sections
        epsilon: Minimum value (default: 1e-6)
    
    Returns:
        Ramp tensor of shape (total_length,)
    """
    output = torch.ones(total_length)
    output[:ramp_length] = torch.linspace(epsilon, 1, ramp_length)
    output[-ramp_length:] = torch.linspace(1, epsilon, ramp_length)
    return output


class CustomFade:
    """
    Gradual fade that adapts over iterations.
    
    Implements a custom fading strategy that gradually applies a fade window
    over multiple iterations, useful for synthesis optimization.
    """
    
    def __init__(self, fade_length, n_iters):
        """
        Initialize CustomFade.
        
        Args:
            fade_length: Length of fade-in/fade-out in samples
            n_iters: Number of iterations to fully apply fade
        """
        self.fade_length = fade_length
        self.n_iters = n_iters
        self.window = None
        self.signal_length = None
        self.initialized_window = False

    def initialize_window(self, X):
        """
        Initialize fade window based on input shape.
        
        Args:
            X: Input tensor to determine signal length
        """
        print(f"Creating fade window for input of shape {X.shape}")
        signal_length = X.shape[-1]
        self.signal_length = signal_length
        window = linear_ramp(signal_length, self.fade_length)
        self.log_window = torch.log(window).to(X.device)
        self.log_h = torch.zeros_like(self.log_window).to(X.device)

    def __call__(self, X):
        """
        Apply fade to input.
        
        Args:
            X: Input tensor to fade
        
        Returns:
            Faded tensor
        """
        if not self.initialized_window:
            self.initialize_window(X)
            self.initialized_window = True
        log_g = (self.log_window - self.log_h) / self.n_iters
        self.log_h += log_g
        return torch.exp(self.log_h) * X


class CustomResample(torchaudio.transforms.Resample):
    """
    Custom resampling transform.
    
    Extends torchaudio.transforms.Resample with modified forward behavior.
    """
    
    def forward(self, waveform):
        """
        Resample waveform.
        
        Args:
            waveform: Input waveform tensor (..., time)
        
        Returns:
            Resampled waveform tensor (..., new_time)
        """
        if self.orig_freq == self.new_freq:
            return waveform
        out = _apply_sinc_resample_kernel(
            waveform, self.orig_freq, self.new_freq, self.gcd, self.kernel, self.width
        )
        return out

