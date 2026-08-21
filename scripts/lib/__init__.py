"""
Neural Prediction Decorrelation Library

A modular library for fMRI encoding models, sound synthesis, and neural data analysis.

Main modules:
- lib.data: Data loading and collation
- lib.encoding: Encoding model functionality
- lib.synthesis: Sound synthesis
- lib.analysis: Analysis functions (reliability, statistics)
- lib.utils: Utilities (math, audio, visualization)
"""

__version__ = "1.0.0"

# Convenience imports for common functionality
from .data import load_data
from .encoding import EncodingModel
from .synthesis import BestLayerSynthesizer
from .analysis import calculate_split_half_reliability
from .utils import mycorr, rms_normalize, plot_brain

__all__ = [
    'load_data',
    'EncodingModel',
    'BestLayerSynthesizer',
    'calculate_split_half_reliability',
    'mycorr',
    'rms_normalize',
    'plot_brain',
]

