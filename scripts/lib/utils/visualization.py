"""
Visualization utilities for fMRI brain data.

Functions for plotting brain surface maps using nilearn and plotly.
"""
import numpy as np
from nilearn import datasets, plotting
import nilearn.surface
from pathlib import Path
from typing import Optional, Dict
from scipy.interpolate import griddata


# Load fsaverage surfaces for brain visualization
fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage')
inflated_L = fsaverage.infl_left
inflated_R = fsaverage.infl_right
curv_L = nilearn.surface.load_surf_data(fsaverage.curv_left)
curv_R = nilearn.surface.load_surf_data(fsaverage.curv_right)
bg_L = np.array([0.9 if c > 0 else 0.5 for c in curv_L])
bg_R = np.array([0.9 if c > 0 else 0.5 for c in curv_R])


# Color schemes for stimulus categories
color_lookup = {
    # Speech categories - blues/greens
    'EngSpeech': '#0e563e',     # Dark green
    'ForSpeech': '#61b48c',     # Light green
    
    # Human vocalizations
    'HumVoc': '#5d2e83',        # Purple
    'HumNonVoc': '#9f3216',     # Red-brown
    'Song': '#1880cd',          # Cyan
    
    # Animal sounds
    'AniVoc': '#8862ac',        # Light purple
    'AniNonVoc': '#e45c71',     # Pink
    
    # Musical/artistic
    'Music': '#22388c',         # Dark blue
    
    # Environmental/nature
    'Nature': '#eae257',        # Yellow
    'EnvSound': '#646464',      # Gray
    
    # Mechanical/artificial
    'Mechanical': '#d58123',    # Orange
    'SfX': '#bcbcbc',          # Light gray
    
    # NPD stimuli
    'NPD': '#d62728'   # Red
}

# Simplified color scheme for NPD vs natural sounds
NPD_color = '#DC3220'  # Red
natural_color = '#005AB5'        # Blue


def plot_brain(
    stat: np.ndarray,
    mask: Optional[np.ndarray] = None,
    fname: Optional[str] = None,\
    cmap: str = "RdBu_r",
    hemi: str = "L",
    **kwargs
):
    """
    Plot brain statistics on inflated surface using plotly.
    
    Args:
        stat: Statistical map to plot (vertex values)
        mask: Optional boolean mask to apply to stat
        fname: If provided, save figure to this path instead of showing
        **kwargs: Additional arguments passed to plot_surf_stat_map
    
    Example:
        >>> import numpy as np
        >>> from lib.utils.visualization import plot_brain
        >>> 
        >>> # Create some example data
        >>> stat = np.random.randn(163842)
        >>> mask = stat > 0
        >>> 
        >>> # Plot and save
        >>> plot_brain(stat, mask, fname='output/brain_plot.png')
    
    Notes:
        - Uses 1st and 99th percentiles for color scaling
        - Plots left hemisphere by default
        - Engine is set to 'plotly' for interactive plots
    """
    # Clip to 1st and 99th percentiles for better visualization
    q1, q99 = np.quantile(stat, (0.001, 0.999))
    stat_ = stat.copy()
    stat_[stat_ < q1] = q1
    stat_[stat_ > q99] = q99
    
    # Apply mask if provided
    stat_map = stat_ * mask if mask is not None else stat_

    # Create the plot
    fig = plotting.plot_surf_stat_map(
        inflated_L if hemi == "L" else inflated_R,
        stat_map=stat_map,
        bg_map=bg_L if hemi == "L" else bg_R,
        engine="plotly",
        cmap=cmap,
        colorbar=False,
        hemi="left" if hemi == "L" else "right",
        **kwargs
    )
    fig.figure.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
    )
    # Save or show
    if fname is not None:
        fig.figure.write_image(fname,scale=4)
    else:
        fig.figure.show()

# 

_DEFAULT_MASKS_DIR = Path(__file__).resolve().parents[3] / "masks"


def stat2grid(
    stat: np.ndarray,
    natsounds: dict,
    # aud_ctx_mask: np.ndarray,
    # reliability_mask: np.ndarray,
    hemi = "L",
    masks_dir: Optional[Path] = None,
) -> np.ndarray:
    """
    Project a per-voxel statistic from auditory cortex grid to fsaverage surface.

    Parameters
    ----------
    stat : np.ndarray
        One-dimensional array of statistics on auditory cortex voxels.
    natsounds : dict
        Stimulus grid metadata loaded from `natsounds.pkl`.
    masks_dir : Path, optional
        Directory containing `old_subjects_grid_{hemi}.npy` and
        `old_subjects_reliability_x_anatomical_fsaverage_{hemi}.npy`.
        Defaults to the repo's top-level `masks/` directory.

    Returns
    -------
    np.ndarray
        Statistic on fsaverage vertices (length 163842), with NaNs outside
        the reliable mask.
    """
    G = natsounds['original']
    q = 1 if hemi == "L" else 0
    values = np.nan * np.zeros(G['grid_x'][q].shape)
    masks_dir = Path(masks_dir) if masks_dir is not None else _DEFAULT_MASKS_DIR
    aud_ctx_mask = np.load(masks_dir / f"old_subjects_grid_{hemi}.npy")#.astype(bool)
    reliability_mask = np.load(masks_dir / f"old_subjects_reliability_x_anatomical_fsaverage_{hemi}.npy")
    values[aud_ctx_mask] = stat
    Z = np.nan * np.zeros(163842)
    points = np.array([G['grid_x'][q].flatten(), G['grid_y'][q].flatten()]).T
    xi = np.array([G['vras'][q][:, 0], G['vras'][q][:, 1]]).T
    Z[G['vi'][q]] = griddata(points, values.flatten(), xi, method='linear')
    Z[~(reliability_mask == 1)] = np.nan
    return Z

def set_plot_area(
    fig,
    plot_w_in: float = 2.5,
    plot_h_in: float = 2.5,
    font_pt: int = 10,
    font_family: str = "Times New Roman",
    left_em: float = 3.5,
    right_em: float = 1.2,
    top_em: float = 1.25,
    bottom_em: float = 2.8
):
    """
    Fix the plot rectangle size while allowing labels to extend into margins.
    
    This function sets precise dimensions for matplotlib/plotly figures to ensure
    consistent sizing for publication. Margins are specified as multiples of font
    size (em units).
    
    Args:
        fig: Plotly figure object to modify
        plot_w_in: Inner plotting area width in inches
        plot_h_in: Inner plotting area height in inches
        font_pt: Font size in points
        font_family: Font family name
        left_em: Left margin as multiple of font size
        right_em: Right margin as multiple of font size
        top_em: Top margin as multiple of font size
        bottom_em: Bottom margin as multiple of font size
    
    Returns:
        Modified figure object
    
    Example:
        >>> import plotly.graph_objects as go
        >>> from lib.utils.visualization import set_plot_area
        >>> 
        >>> fig = go.Figure(data=go.Scatter(x=[1,2,3], y=[4,5,6]))
        >>> fig = set_plot_area(fig, plot_w_in=3.0, plot_h_in=2.5)
        >>> fig.show()
    
    Notes:
        - Designed for publication-quality figures
        - Ensures consistent sizing across different plot types
        - Increase margin *_em values if labels are cut off
    """
    PX_PER_IN = 96.0
    PT_PER_IN = 72.0
    
    # Convert inches to pixels
    plot_w_px = int(round(plot_w_in * PX_PER_IN))
    plot_h_px = int(round(plot_h_in * PX_PER_IN))
    font_px = font_pt * (PX_PER_IN / PT_PER_IN)  # ≈ pt * 1.3333
    
    # Calculate margins in pixels, scaled by font size
    l = int(round(left_em * font_px))
    r = int(round(right_em * font_px))
    t = int(round(top_em * font_px))
    b = int(round(bottom_em * font_px))
    
    # Total canvas size
    width = plot_w_px + l + r
    height = plot_h_px + t + b
    
    # Update layout
    fig.update_layout(
        width=width,
        height=height,
        margin=dict(l=l, r=r, t=t, b=b),
        font=dict(size=font_pt, family=font_family, color="black"),
        xaxis=dict(automargin=False, ticks="outside"),
        yaxis=dict(automargin=False, ticks="outside")
    )
    
    return fig


def get_category_color(category: str, default: str = '#7f7f7f') -> str:
    """
    Get color for a stimulus category.
    
    Args:
        category: Category name (e.g., 'EngSpeech', 'Music', 'NPD')
        default: Default color if category not found

    Returns:
        Hex color string

    Example:
        >>> from lib.utils.visualization import get_category_color
        >>> color = get_category_color('NPD')
        >>> print(color)
        '#d62728'
    """
    return color_lookup.get(category, default)

