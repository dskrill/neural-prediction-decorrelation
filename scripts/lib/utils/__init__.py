"""
Utility functions for audio processing, mathematics, and visualization.
"""
from .math import mycorr, mycov, demean, z_score, identity, lcm_float, get_converted_sampling_rates, concordance_correlation_coefficient, cohen_d, degenerate_entries
from .audio import rms_normalize, rms, linear_ramp, CustomFade, CustomResample
from .visualization import (
    plot_brain,
    set_plot_area,
    get_category_color,
    color_lookup,
    NPD_color,
    natural_color,
    fsaverage,
    inflated_L,
    inflated_R,
    bg_L,
    bg_R,
)

__all__ = [
    # Math utilities
    'mycorr',
    'mycov',
    'demean',
    'z_score',
    'identity',
    'lcm_float',
    'get_converted_sampling_rates',
    'concordance_correlation_coefficient',
    'degenerate_entries',
    # Audio utilities
    'rms_normalize',
    'rms',
    'linear_ramp',
    'CustomFade',
    'CustomResample',
    # Visualization utilities
    'plot_brain',
    'set_plot_area',
    'get_category_color',
    'color_lookup',
    'NPD_color',
    'natural_color',
    'fsaverage',
    'inflated_L',
    'inflated_R',
    'bg_L',
    'bg_R',
]

