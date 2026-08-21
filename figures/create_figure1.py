#!/usr/bin/env python3
"""
Create Figure 1 - Prediction Accuracy Comparisons
"""

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

import argparse
from pathlib import Path
import sys

sys.path.append('../scripts')

import numpy as np
from lib.utils import mycorr, concordance_correlation_coefficient, degenerate_entries
from lib.data import load_original_subjects_data, load_dict_h5
from lib.utils.visualization import plot_brain, stat2grid
import plotly.graph_objects as go
from tqdm.autonotebook import trange
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import nilearn
from nilearn import datasets,plotting


N_SUBJECTS: int = 20
DOT_SIZE: int = 2
DOT_OPACITY: float = 0.04
NPD_INDICES = np.arange(60)
NATURAL_INDICES = np.arange(60, 115)


def masked_corr(A, B, axis):
    """mycorr(A, B, axis) with degenerate (zero-variance/NaN) entries set to NaN.

    mycorr() computes correlation via nansum/nanmean, so a zero-variance (or
    all-NaN) input produces a 0/0 that silently resolves to an exact 0.0
    instead of the mathematically undefined NaN -- these entries are not
    "uncorrelated," they're degenerate and should be excluded.
    """
    C = mycorr(A, B, axis=axis)
    return np.where(degenerate_entries(A, B, axis=axis), np.nan, C)


def masked_group_corr_mean(A, B):
    """Per-subject masked_corr(A, B, axis=1), then nanmean over subjects (axis 0).

    Using nanmean (rather than mean) means one degenerate subject doesn't
    blank out an otherwise-valid voxel's group-average correlation.
    """
    return np.nanmean(masked_corr(A, B, axis=1), axis=0)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the Figure 1 script.

    Returns
    -------
    argparse.Namespace
        Namespace with `data_dir`, `masks_dir`, and `output_dir` attributes.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('../data'))
    parser.add_argument('--masks-dir', type=Path, default=Path('../masks'))
    parser.add_argument('--output-dir', type=Path, default=Path('output_create_figure1'))
    parser.add_argument('--log-file', type=Path, default=None,
                        help='Path to stats log file (default: <output-dir>/stats.log)')
    return parser.parse_args()


def load_original_subjects(data_dir: Path):
    """
    Load original subjects data used for Figure 1.

    Parameters
    ----------
    data_dir : Path
        Base directory containing the `original-subjects` data.

    Returns
    -------
    dict
        Nested dictionary with keys `observed` and `predictions` as expected
        by downstream computations in this script.
    """
    npz_path = data_dir / 'original-subjects' / 'observed_and_predicted_data_original_subjects_hemi.npz'
    return load_original_subjects_data(str(npz_path))


def load_and_clean_roi_masks(masks_dir: Path, reliability_mask: np.ndarray):
    """
    Load ROI masks and clean NaNs for both hemispheres.

    Parameters
    ----------
    masks_dir : Path
        Directory containing `ROI_masks.pkl`.
    reliability_mask : np.ndarray
        Boolean reliability mask on fsaverage vertices.

    Returns
    -------
    dict
        Dictionary with keys 'L' and 'R', each mapping ROI names to boolean
        masks on the fsaverage surface.
    """
    masks_path = masks_dir / "ROI_masks.h5"
    boolean_masks = load_dict_h5(str(masks_path))
    for mask in boolean_masks['L']:
        if np.isnan(boolean_masks['L'][mask]).any():
            boolean_masks['L'][mask] = np.where(np.isnan(boolean_masks['L'][mask]), False, True)
        if np.isnan(boolean_masks['R'][mask]).any():
            boolean_masks['R'][mask] = np.where(np.isnan(boolean_masks['R'][mask]), False, True)
    return boolean_masks

def slice_and_dice(data: dict, kind: str = 'observed', test_individual: str = 'S30'):
    """
    Extract individual and group responses for natural and npd stimuli.

    Parameters
    ----------
    data : dict
        Loaded original-subjects data structure.
    kind : {'observed', 'robust1', 'standard1', 'robust2', 'standard2'}
        Type of responses to extract.
    test_individual : str
        Subject ID to use as the individual example (default 'S30').

    Returns
    -------
    tuple of np.ndarray
        (D_individual_natural, D_individual_npd,
         D_group_natural, D_group_npd)
    """
    test_subjects = [f"S{i}" for i in range(11, 31)]
    if kind == 'observed':
        D_individual_L = data['observed'][test_individual]['data_L'].squeeze()
        D_individual_R = data['observed'][test_individual]['data_R'].squeeze()
        D_group_L = np.array([data['observed'][sid]['data_L'].squeeze() for sid in test_subjects])
        D_group_R = np.array([data['observed'][sid]['data_R'].squeeze() for sid in test_subjects])
    else:
        assert kind in ['robust1', 'standard1', 'robust2', 'standard2']
        D_individual_L = data['predictions'][test_individual][f"{kind}_L"]
        D_individual_R = data['predictions'][test_individual][f"{kind}_R"]
        D_group_L = np.array([data['predictions'][sid][f"{kind}_L"] for sid in test_subjects])
        D_group_R = np.array([data['predictions'][sid][f"{kind}_R"] for sid in test_subjects])

    D_individual_natural_L = D_individual_L[NATURAL_INDICES]
    D_individual_npd_L = D_individual_L[NPD_INDICES]
    D_individual_natural_R = D_individual_R[NATURAL_INDICES]
    D_individual_npd_R = D_individual_R[NPD_INDICES]

    D_group_natural_L = D_group_L[:, NATURAL_INDICES]
    D_group_npd_L = D_group_L[:, NPD_INDICES]
    D_group_natural_R = D_group_R[:, NATURAL_INDICES]
    D_group_npd_R = D_group_R[:, NPD_INDICES]
    return (
        D_individual_natural_L,
        D_individual_natural_R,
        D_individual_npd_L,
        D_individual_npd_R,
        D_group_natural_L,
        D_group_natural_R,
        D_group_npd_L,
        D_group_npd_R,
    )



def main() -> None:
    """
    Orchestrate loading data, computing statistics, and generating Figure 1.
    """
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    class _Tee:
        def __init__(self, *streams): self._s = streams
        def write(self, d): [s.write(d) for s in self._s]
        def flush(self): [s.flush() for s in self._s]

    _log_path = args.log_file if args.log_file is not None else args.output_dir / 'stats.log'
    _log_fh = open(_log_path, 'w')
    sys.stdout = _Tee(sys.__stdout__, _log_fh)

    print(f"Output: {args.output_dir}")

    #-------------------------------------------------------------------------------
    # Data loading and computations
    #-------------------------------------------------------------------------------

    data = load_original_subjects(args.data_dir)

    reliability_mask_L = np.load(str(args.masks_dir / "old_subjects_reliability_x_anatomical_fsaverage_L.npy"))
    reliability_mask_R = np.load(str(args.masks_dir / "old_subjects_reliability_x_anatomical_fsaverage_R.npy"))
    reliability_mask_L = np.where(np.isnan(reliability_mask_L), False, True)
    reliability_mask_R = np.where(np.isnan(reliability_mask_R), False, True)
    # aud_ctx_mask = np.load(str(args.masks_dir / "old_subjects_grid.npy")).astype(bool)
    natsounds = load_dict_h5(str(args.data_dir / 'original-subjects/natsounds.h5'))

    import h5py
    _h5 = h5py.File(str(args.masks_dir / 'NH2015_B2021_test-retest_0p4.h5'), 'r')
    # h5 subjects are sub-01..sub-30; figure 1 uses S11..S30 = indices 10..29
    subject_masks_L = np.array(_h5['fsaverage/lh'][10:30]) == 1   # (20, 163842)
    subject_masks_R = np.array(_h5['fsaverage/rh'][10:30]) == 1   # (20, 163842)
    _h5.close()

    (
        D_individual_natural_observed_L,
        D_individual_natural_observed_R,
        D_individual_npd_observed_L,
        D_individual_npd_observed_R,
        D_group_natural_observed_L,
        D_group_natural_observed_R,
        D_group_npd_observed_L,
        D_group_npd_observed_R,
    ) = slice_and_dice(data, 'observed')
    (
        D_individual_natural_robust1_L,
        D_individual_natural_robust1_R,
        D_individual_npd_robust1_L,
        D_individual_npd_robust1_R,
        D_group_natural_robust1_L,
        D_group_natural_robust1_R,
        D_group_npd_robust1_L,
        D_group_npd_robust1_R,
    ) = slice_and_dice(data, 'robust1')
    (
        D_individual_natural_standard1_L,
        D_individual_natural_standard1_R,
        D_individual_npd_standard1_L,
        D_individual_npd_standard1_R,
        D_group_natural_standard1_L,
        D_group_natural_standard1_R,
        D_group_npd_standard1_L,
        D_group_npd_standard1_R,
    ) = slice_and_dice(data, 'standard1')
    (
        D_individual_natural_robust2_L,
        D_individual_natural_robust2_R,
        D_individual_npd_robust2_L,
        D_individual_npd_robust2_R,
        D_group_natural_robust2_L,
        D_group_natural_robust2_R,
        D_group_npd_robust2_L,
        D_group_npd_robust2_R,
    ) = slice_and_dice(data, 'robust2')
    (
        D_individual_natural_standard2_L,
        D_individual_natural_standard2_R,
        D_individual_npd_standard2_L,
        D_individual_npd_standard2_R,
        D_group_natural_standard2_L,
        D_group_natural_standard2_R,
        D_group_npd_standard2_L,
        D_group_npd_standard2_R,
    ) = slice_and_dice(data, 'standard2')

    AR_prediction_accuracy_L = masked_corr(
        D_group_natural_observed_L,
        D_group_natural_robust1_L,
        axis=1
    )
    AR_prediction_accuracy_R = masked_corr(
        D_group_natural_observed_R,
        D_group_natural_robust1_R,
        axis=1
    )

    standard_prediction_accuracy_L = masked_corr(
        D_group_natural_observed_L,
        D_group_natural_standard1_L,
        axis=1
    )
    standard_prediction_accuracy_R = masked_corr(
        D_group_natural_observed_R,
        D_group_natural_standard1_R,
        axis=1
    )

    # import pdb; pdb.set_trace()
    AR_prediction_accuracy_fsaverage_L = np.array(
        [stat2grid(AR_prediction_accuracy_L[i], natsounds,hemi="L") for i in trange(N_SUBJECTS)]
    )
    AR_prediction_accuracy_fsaverage_R = np.array(
        [stat2grid(AR_prediction_accuracy_R[i], natsounds,hemi="R") for i in trange(N_SUBJECTS)]
    )
    standard_prediction_accuracy_fsaverage_L = np.array(
        [stat2grid(standard_prediction_accuracy_L[i], natsounds,hemi="L") for i in trange(N_SUBJECTS)]
    )
    standard_prediction_accuracy_fsaverage_R = np.array(
        [stat2grid(standard_prediction_accuracy_R[i], natsounds,hemi="R") for i in trange(N_SUBJECTS)]
    )

    print(f"Median AR model prediction accuracy (all voxels L): {np.nanmedian(AR_prediction_accuracy_fsaverage_L)}")
    print(f"Median AR model prediction accuracy (all voxels R): {np.nanmedian(AR_prediction_accuracy_fsaverage_R)}")
    print(f"Median Standard model prediction accuracy (all voxels L): {np.nanmedian(standard_prediction_accuracy_fsaverage_L)}")
    print(f"Median Standard model prediction accuracy (all voxels R): {np.nanmedian(standard_prediction_accuracy_fsaverage_R)}")
    # paired t-test (paired voxels)
    nan_mask_L = np.isnan(AR_prediction_accuracy_fsaverage_L[0]) | np.isnan(standard_prediction_accuracy_fsaverage_L[0])
    nan_mask_R = np.isnan(AR_prediction_accuracy_fsaverage_R[0]) | np.isnan(standard_prediction_accuracy_fsaverage_R[0])
    # t_stat, p_value = scipy.stats.ttest_rel(
    #     AR_prediction_accuracy_fsaverage[...,~nan_mask].flatten(),
    #     standard_prediction_accuracy_fsaverage[...,~nan_mask].flatten(),
    # )
    # print(f"Paired t-test between AR and Standard model prediction accuracies (reliable voxels): t={t_stat}, p={p_value}")


    fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage')
    inflated_L = fsaverage.infl_left
    inflated_R = fsaverage.infl_right
    curv_L = nilearn.surface.load_surf_data(fsaverage.curv_left)
    curv_R = nilearn.surface.load_surf_data(fsaverage.curv_right)
    bg_L = np.array([0.9 if c > 0 else 0.5 for c in curv_L])
    bg_R = np.array([0.9 if c > 0 else 0.5 for c in curv_R])

    # different models AND different training stimuli
    c_corrs_between_predictions_individual_L = masked_corr(D_individual_natural_robust1_L,D_individual_natural_standard2_L,axis=0)
    c_corrs_between_predictions_individual_R = masked_corr(D_individual_natural_robust1_R,D_individual_natural_standard2_R,axis=0)
    c_corrs_between_predictions_group_L = masked_group_corr_mean(D_group_natural_robust1_L,D_group_natural_standard2_L)
    c_corrs_between_predictions_group_R = masked_group_corr_mean(D_group_natural_robust1_R,D_group_natural_standard2_R)

    # SAME model, different training stimuli
    ceiling_corrs_individual_robust_L = masked_corr(D_individual_natural_robust1_L,D_individual_natural_robust2_L,axis=0)
    ceiling_corrs_individual_robust_R = masked_corr(D_individual_natural_robust1_R,D_individual_natural_robust2_R,axis=0)
    ceiling_corrs_group_robust_L = masked_group_corr_mean(D_group_natural_robust1_L,D_group_natural_robust2_L)
    ceiling_corrs_group_robust_R = masked_group_corr_mean(D_group_natural_robust1_R,D_group_natural_robust2_R)
    ceiling_corrs_individual_standard_L = masked_corr(D_individual_natural_standard1_L,D_individual_natural_standard2_L,axis=0)
    ceiling_corrs_individual_standard_R = masked_corr(D_individual_natural_standard1_R,D_individual_natural_standard2_R,axis=0)
    ceiling_corrs_group_standard_L = masked_group_corr_mean(D_group_natural_standard1_L,D_group_natural_standard2_L)
    ceiling_corrs_group_standard_R = masked_group_corr_mean(D_group_natural_standard1_R,D_group_natural_standard2_R)
    
    c_corrs_between_predictions_individual_fsaverage_L = stat2grid(
        c_corrs_between_predictions_individual_L,
        natsounds,
        hemi="L"
    )
    c_corrs_between_predictions_individual_fsaverage_R = stat2grid(
        c_corrs_between_predictions_individual_R,
        natsounds,
        hemi="R"
    )
    c_corrs_between_predictions_group_fsaverage_L = stat2grid(
        c_corrs_between_predictions_group_L,
        natsounds,
        hemi="L"
    )
    c_corrs_between_predictions_group_fsaverage_R = stat2grid(
        c_corrs_between_predictions_group_R,
        natsounds,
        hemi="R"
    )
    ceiling_corrs_individual_robust_fsaverage_L = stat2grid(
        ceiling_corrs_individual_robust_L,
        natsounds,
        hemi="L"
    )
    ceiling_corrs_individual_robust_fsaverage_R = stat2grid(
        ceiling_corrs_individual_robust_R,
        natsounds,
        hemi="R"
    )
    ceiling_corrs_group_robust_fsaverage_L = stat2grid(
        ceiling_corrs_group_robust_L,
        natsounds,
        hemi="L"
    )
    ceiling_corrs_group_robust_fsaverage_R = stat2grid(
        ceiling_corrs_group_robust_R,
        natsounds,
        hemi="R"
    )
    ceiling_corrs_individual_standard_fsaverage_L = stat2grid(
        ceiling_corrs_individual_standard_L,
        natsounds,
        hemi="L"
    )
    ceiling_corrs_individual_standard_fsaverage_R = stat2grid(
        ceiling_corrs_individual_standard_R,
        natsounds,
        hemi="R"
    )
    ceiling_corrs_group_standard_fsaverage_L = stat2grid(
        ceiling_corrs_group_standard_L,
        natsounds,
        hemi="L"
    )
    ceiling_corrs_group_standard_fsaverage_R = stat2grid(
        ceiling_corrs_group_standard_R,
        natsounds,
        hemi="R"
    )

    #-------------------------------------------------------------------------------
    # Plotting
    #-------------------------------------------------------------------------------
    plot_brain(np.nan * np.empty(163842),fname=str(args.output_dir / "blank_brain.png"))
    #-------------------------------------------------------------------------------
    # Figure: Surface prediction accuracy maps
    #-------------------------------------------------------------------------------

    # plot_brain(
    #     c_corrs_between_predictions_individual_fsaverage,
    #     vmin=-1,
    #     vmax=1,
    #     cmap="PiYG",
    #     fname=str(args.output_dir / "between_predictions_individual.png"),
    # )

    #####################
    # Left hemisphere
    #####################
    plot_brain(
        np.nanmean(standard_prediction_accuracy_fsaverage_L, axis=0),
        vmin=-1,
        vmax=1,
        cmap="RdBu_r",
        fname=str(args.output_dir / "standard_prediction_accuracy_group_L.png"),
    )
    plot_brain(
        np.nanmean(AR_prediction_accuracy_fsaverage_L, axis=0),
        vmin=-1,
        vmax=1,
        cmap="RdBu_r",
        fname=str(args.output_dir / "AR_prediction_accuracy_group_L.png"),
    )

    plot_brain(
        standard_prediction_accuracy_fsaverage_L[-1],
        vmin=-1,
        vmax=1,
        cmap="RdBu_r",
        fname=str(args.output_dir / "standard_prediction_accuracy_individual_S30_L.png"),
    )
    plot_brain(
        AR_prediction_accuracy_fsaverage_L[-1],
        vmin=-1,
        vmax=1,
        cmap="RdBu_r",
        fname=str(args.output_dir / "AR_prediction_accuracy_individual_S30_L.png"),
    )
    plot_brain(
        c_corrs_between_predictions_individual_fsaverage_L,
        vmin=-1, vmax=1, cmap="PiYG",
        fname=str(args.output_dir / "between_predictions_individual_S30_L.png"),
    )
    plot_brain(
        ceiling_corrs_individual_robust_fsaverage_L,
        vmin=-1, vmax=1, cmap="PiYG",
        fname=str(args.output_dir / "ceiling_individual_robust_S30_L.png"),
    )
    plot_brain(
        ceiling_corrs_individual_standard_fsaverage_L,
        vmin=-1, vmax=1, cmap="PiYG",
        fname=str(args.output_dir / "ceiling_individual_nonrobust_S30_L.png"),
    )
    plot_brain(
        c_corrs_between_predictions_group_fsaverage_L,
        vmin=-1,
        vmax=1,
        cmap="PiYG",
        fname=str(args.output_dir / "between_predictions_group_L.png"),
    )
    # plot_brain(
    #     ceiling_corrs_individual_fsaverage,
    #     vmin=-1,
    #     vmax=1,
    #     cmap="PiYG",
    #     fname=str(args.output_dir / "ceiling_individual.png"),
    # )
    plot_brain(
        ceiling_corrs_group_robust_fsaverage_L,
        vmin=-1,
        vmax=1,
        cmap="PiYG",
        fname=str(args.output_dir / "ceiling_group_robust_L.png"),
    )
    plot_brain(
        ceiling_corrs_group_standard_fsaverage_L,
        vmin=-1,
        vmax=1,
        cmap="PiYG",
        fname=str(args.output_dir / "ceiling_group_nonrobust_L.png"),
    )

    #####################
    # Right hemisphere
    #####################
    plot_brain(
        np.nanmean(standard_prediction_accuracy_fsaverage_R, axis=0),
        vmin=-1,
        vmax=1,
        cmap="RdBu_r",
        fname=str(args.output_dir / "standard_prediction_accuracy_group_R.png"),
        hemi="R"
    )
    plot_brain(
        np.nanmean(AR_prediction_accuracy_fsaverage_R, axis=0),
        vmin=-1,
        vmax=1,
        cmap="RdBu_r",
        fname=str(args.output_dir / "AR_prediction_accuracy_group_R.png"),
        hemi="R"
    )

    plot_brain(
        standard_prediction_accuracy_fsaverage_R[-1],
        vmin=-1,
        vmax=1,
        cmap="RdBu_r",
        fname=str(args.output_dir / "standard_prediction_accuracy_individual_S30_R.png"),
        hemi="R"
    )
    plot_brain(
        AR_prediction_accuracy_fsaverage_R[-1],
        vmin=-1,
        vmax=1,
        cmap="RdBu_r",
        fname=str(args.output_dir / "AR_prediction_accuracy_individual_S30_R.png"),
        hemi="R"
    )
    plot_brain(
        c_corrs_between_predictions_individual_fsaverage_R,
        vmin=-1, vmax=1, cmap="PiYG",
        fname=str(args.output_dir / "between_predictions_individual_S30_R.png"),
        hemi="R",
    )
    plot_brain(
        ceiling_corrs_individual_robust_fsaverage_R,
        vmin=-1, vmax=1, cmap="PiYG",
        fname=str(args.output_dir / "ceiling_individual_robust_S30_R.png"),
        hemi="R",
    )
    plot_brain(
        ceiling_corrs_individual_standard_fsaverage_R,
        vmin=-1, vmax=1, cmap="PiYG",
        fname=str(args.output_dir / "ceiling_individual_nonrobust_S30_R.png"),
        hemi="R",
    )
    plot_brain(
        c_corrs_between_predictions_group_fsaverage_R,
        vmin=-1,
        vmax=1,
        cmap="PiYG",
        fname=str(args.output_dir / "between_predictions_group_R.png"),
        hemi="R"
    )
    # plot_brain(
    #     ceiling_corrs_individual_fsaverage,
    #     vmin=-1,
    #     vmax=1,
    #     cmap="PiYG",
    #     fname=str(args.output_dir / "ceiling_individual.png"),
    # )
    plot_brain(
        ceiling_corrs_group_robust_fsaverage_R,
        vmin=-1,
        vmax=1,
        cmap="PiYG",
        fname=str(args.output_dir / "ceiling_group_robust_R.png"),
        hemi="R"
    )
    plot_brain(
        ceiling_corrs_group_standard_fsaverage_R,
        vmin=-1,
        vmax=1,
        cmap="PiYG",
        fname=str(args.output_dir / "ceiling_group_nonrobust_R.png"),
        hemi="R"
    )

    #-------------------------------------------------------------------------------
    # End Figure: Surface prediction accuracy maps
    #-------------------------------------------------------------------------------

    #-------------------------------------------------------------------------------
    # Figure: Individual standard vs robust prediction correlations
    #-------------------------------------------------------------------------------

    # fig = go.Figure()
    # fig.add_trace(
    #     go.Scatter(
    #         x=[-1,1],
    #         y=[-1,1],
    #         mode='lines',
    #         line=dict(
    #             color="cyan",
    #             dash="dash"
    #         ),
    #         showlegend=False
    #     )
    # )
    # fig.add_trace(
    #     go.Scatter(
    #         x=c_corrs_between_predictions_individual,
    #         y=ceiling_corrs_individual_robust,
    #         mode='markers',
    #         marker=dict(
    #             size=DOT_SIZE,
    #             opacity=DOT_OPACITY,
    #             color="black",
    #         ),
    #         showlegend=False,
    #     )
    # )
    # # 1-1 line
    
    # fig.update_xaxes(
    #     scaleanchor="y",
    #     scaleratio=1,
    #     zeroline=True,
    #     tickvals=[-1,-.5,0,.5,1],
    #     showticklabels = False,
    #     gridcolor="lightgrey",
    #     zerolinecolor="black"
    # )
    # fig.update_yaxes(
    #     scaleanchor="x",
    #     scaleratio=1,
    #     zeroline=True,
    #     tickvals=[-.5,0,.5,1],
    #     showticklabels = False,
    #     gridcolor="lightgrey",
    #     zerolinecolor="black"
    # )
    # fig.update_layout(
    #     width = .75*100,
    #     height = .75*100,
    #     xaxis_range=[-1,1.1],
    #     yaxis_range=[-1,1.1],
    #     plot_bgcolor="rgba(0,0,0,0)",
    #     paper_bgcolor="rgba(0,0,0,0)",
    #     margin=dict(l=0,r=0,t=0,b=0),
    # )

    # base_name_individual = "individual_nonrobust_vs_robust"
    # fig.write_image(
    #     str(args.output_dir / f"{base_name_individual}.png"),
    #     scale=12,
    #     width=.75 * 100,
    #     height=.75 * 100,
    # )
    # fig.write_html(str(args.output_dir / f"{base_name_individual}.html"))

    #-------------------------------------------------------------------------------
    # End Figure: Individual standard vs robust prediction correlations
    #-------------------------------------------------------------------------------

    #-------------------------------------------------------------------------------
    # Figure: Group standard vs robust prediction correlations
    #-------------------------------------------------------------------------------
    # ccc_prediction_accuracy_group = concordance_correlation_coefficient(
    #     standard_prediction_accuracy.mean(0),
    #     AR_prediction_accuracy.mean(0),
    # )
    # print("CCC prediction accuracy (group):")
    # for k,v in ccc_prediction_accuracy_group.items():
    #     print(f"  {k}: {v:.2f}")


    fig = go.Figure()
        # 1-1 line
    fig.add_trace(
        go.Scatter(
            x=[-1,1],
            y=[-1,1],
            mode='lines',
            line=dict(
                color="cyan",
                dash="dash"
            ),
            showlegend=False
        )
    )
    _group_acc_x = np.concatenate([
        np.nanmean(standard_prediction_accuracy_L, axis=0), np.nanmean(standard_prediction_accuracy_R, axis=0)
    ])
    _group_acc_y = np.concatenate([
        np.nanmean(AR_prediction_accuracy_L, axis=0), np.nanmean(AR_prediction_accuracy_R, axis=0)
    ])
    _group_acc_valid = np.isfinite(_group_acc_x) & np.isfinite(_group_acc_y)
    fig.add_trace(
        go.Scatter(
            x=_group_acc_x[_group_acc_valid],
            y=_group_acc_y[_group_acc_valid],
            mode='markers',
            marker=dict(
                size=DOT_SIZE,
                opacity=DOT_OPACITY,
                color="black",
            ),
            showlegend=False,
        )
    )

    fig.update_xaxes(
        scaleanchor="y",
        scaleratio=1,
        zeroline=True,
        tickvals=[-.5,0,.5,1],
        showticklabels = False,
        gridcolor="lightgrey",
        zerolinecolor="black"
    )
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
        zeroline=True,
        tickvals=[-.5,0,.5,1],
        showticklabels = False,
        gridcolor="lightgrey",
        zerolinecolor="black"
    )
    fig.update_layout(
        width = .75*100,
        height = .75*100,
        xaxis_range=[-.5,1.1],
        yaxis_range=[-.5,1.1],
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0,r=0,t=0,b=0),
    )

    base_name_group = "group_prediction_accuracy"
    fig.write_image(
        str(args.output_dir / f"{base_name_group}.png"),
        scale=12,
        width=.75 * 100,
        height=.75 * 100,
    )
    # fig.write_html(str(args.output_dir / f"{base_name_group}.html", include_plotlyjs=False), include_plotlyjs=False)

    #-------------------------------------------------------------------------------
    # End Figure: Group prediction accuracy
    #-------------------------------------------------------------------------------

    #-------------------------------------------------------------------------------
    # Figure: Individual prediction accuracy
    #-------------------------------------------------------------------------------

    # ccc_prediction_accuracy_individual = concordance_correlation_coefficient(
    #     standard_prediction_accuracy[-1,:],
    #     AR_prediction_accuracy[-1,:],
    # )
    # print("CCC prediction accuracy (individual):")
    # for k,v in ccc_prediction_accuracy_individual.items():
    #     print(f"  {k}: {v:.3f}")

    fig = go.Figure()
    # 1-1 line
    fig.add_trace(
        go.Scatter(
            x=[-1,1],
            y=[-1,1],
            mode='lines',
            line=dict(
                color="cyan",
                dash="dash"
            ),
            showlegend=False
        )
    )
    _s30_acc_x = np.concatenate([standard_prediction_accuracy_L[-1,:], standard_prediction_accuracy_R[-1,:]])
    _s30_acc_y = np.concatenate([AR_prediction_accuracy_L[-1,:], AR_prediction_accuracy_R[-1,:]])
    _s30_acc_valid = np.isfinite(_s30_acc_x) & np.isfinite(_s30_acc_y)
    fig.add_trace(
        go.Scatter(
            x=_s30_acc_x[_s30_acc_valid],
            y=_s30_acc_y[_s30_acc_valid],
            mode='markers',
            marker=dict(
                size=DOT_SIZE,
                opacity=DOT_OPACITY,
                color="black",
            ),
            showlegend=False,
        )
    )

    fig.update_xaxes(
        scaleanchor="y",
        scaleratio=1,
        zeroline=True,
        tickvals=[-.5,0,.5,1],
        showticklabels = False,
        gridcolor="lightgrey",
        zerolinecolor="black"
    )
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
        zeroline=True,
        tickvals=[-.5,0,.5,1],
        showticklabels = False,
        gridcolor="lightgrey",
        zerolinecolor="black"
    )
    fig.update_layout(
        width = .75*100,
        height = .75*100,
        xaxis_range=[-.5,1.1],
        yaxis_range=[-.5,1.1],
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0,r=0,t=0,b=0),
    )

    base_name_individual = "individual_prediction_accuracy_S30"
    fig.write_image(
        str(args.output_dir / f"{base_name_individual}.png"),
        scale=12,
        width=.75 * 100,
        height=.75 * 100,
    )
    # fig.write_html(str(args.output_dir / f"{base_name_individual}.html", include_plotlyjs=False), include_plotlyjs=False)

    #-------------------------------------------------------------------------------
    # End Figure: Individual prediction accuracy
    #-------------------------------------------------------------------------------

    #-------------------------------------------------------------------------------
    # Figure: Group similarity between predictions and ceiling (robust model)
    #-------------------------------------------------------------------------------
    _group_pred_ceiling_valid = (
        ~nan_mask_L
        & np.isfinite(c_corrs_between_predictions_group_fsaverage_L)
        & np.isfinite(ceiling_corrs_group_robust_fsaverage_L)
    )
    pearsonr_corrs_bw_pred_group = scipy.stats.pearsonr(
            c_corrs_between_predictions_group_fsaverage_L[_group_pred_ceiling_valid],ceiling_corrs_group_robust_fsaverage_L[_group_pred_ceiling_valid])[0]
    print(f"Pearson correlation between group similarity between predictions and ceiling (robust model): {pearsonr_corrs_bw_pred_group}")
    _individual_pred_ceiling_valid = (
        ~nan_mask_L
        & np.isfinite(c_corrs_between_predictions_individual_fsaverage_L)
        & np.isfinite(ceiling_corrs_individual_robust_fsaverage_L)
    )
    pearsonr_corrs_bw_pred_individual = scipy.stats.pearsonr(c_corrs_between_predictions_individual_fsaverage_L[_individual_pred_ceiling_valid],ceiling_corrs_individual_robust_fsaverage_L[_individual_pred_ceiling_valid])[0]
    print(f"Pearson correlation between individual similarity between predictions and ceiling (robust model): {pearsonr_corrs_bw_pred_individual}")
    # paired t-test (paired voxels)
    # t_stat, p_value = scipy.stats.ttest_rel(
    #     c_corrs_between_predictions_group,ceiling_corrs_group_robust,
    # )
    # print(f"Paired t-test between group similarity between predictions and ceiling (robust model): t={t_stat}, p={p_value}")

    # ccc_corrs_ceiling_group_robust = concordance_correlation_coefficient(
    #     mycorr(D_group_natural_robust1, D_group_natural_standard2, axis=1).mean(0),
    #     mycorr(D_group_natural_robust1, D_group_natural_robust2, axis=1).mean(0),
    # )
    # print("CCC correlation between predictions and ceiling (robust model, group):")
    # for k,v in ccc_corrs_ceiling_group_robust.items():
    #     print(f"  {k}: {v:.3f}")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[-.5,1.1],
            y=[-.5,1.1],
            mode='lines',
            line=dict(
                color="cyan",
                dash="dash"
            ),
            showlegend=False
        )
    )

    #-------------------------------------------------------------------------------
    # Figure: Group similarity between predictions vs ceiling (robust model)
    #-------------------------------------------------------------------------------

    _group_ceiling_vs_pred_x = np.concatenate([ceiling_corrs_group_robust_L, ceiling_corrs_group_robust_R])
    _group_ceiling_vs_pred_y = np.concatenate([c_corrs_between_predictions_group_L, c_corrs_between_predictions_group_R])
    _group_ceiling_vs_pred_valid = np.isfinite(_group_ceiling_vs_pred_x) & np.isfinite(_group_ceiling_vs_pred_y)
    fig.add_trace(
        go.Scatter(
            x=_group_ceiling_vs_pred_x[_group_ceiling_vs_pred_valid],
            y=_group_ceiling_vs_pred_y[_group_ceiling_vs_pred_valid],
            mode='markers',
            marker=dict(
                size=DOT_SIZE,
                opacity=DOT_OPACITY,
                color="black",
            ),
            showlegend=False,
        )
    )
    # 1-1 line

    fig.update_xaxes(
        scaleanchor="y",
        scaleratio=1,
        zeroline=True,
        tickvals=[-1,-.5,0,.5,1],
        showticklabels = False,
        gridcolor="lightgrey",
        zerolinecolor="black"
    )
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
        zeroline=True,
        tickvals=[-.5,0,.5,1],
        showticklabels = False,
        gridcolor="lightgrey",
        zerolinecolor="black"
    )
    fig.update_layout(
        width = .75*100,
        height = .75*100,
        xaxis_range=[-.5,1.1],
        yaxis_range=[-.5,1.1],
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0,r=0,t=0,b=0),
    )

    base_name_group_robust = "group_similarity_predictions_vs_ceiling_robust"
    fig.write_image(
        str(args.output_dir / f"{base_name_group_robust}.png"),
        scale=12,
        width=.75 * 100,
        height=.75 * 100,
    )
    # fig.write_html(str(args.output_dir / f"{base_name_group_robust}.html", include_plotlyjs=False), include_plotlyjs=False)
    #-------------------------------------------------------------------------------
    # End Figure: Group similarity between predictions vs ceiling (robust model)
    #-------------------------------------------------------------------------------

    #-------------------------------------------------------------------------------
    # Figure: Group similarity between predictions vs ceiling (standard model)
    #-------------------------------------------------------------------------------

    fig = go.Figure()
    # 1-1 line
    fig.add_trace(
        go.Scatter(
            x=[-.5,1.1],
            y=[-.5,1.1],
            mode='lines',
            line=dict(
                color="cyan",
                dash="dash"
            ),
            showlegend=False
        )
    )
    
    _group_standard_x = np.concatenate([ceiling_corrs_group_standard_L, ceiling_corrs_group_standard_R])
    _group_standard_y = np.concatenate([c_corrs_between_predictions_group_L, c_corrs_between_predictions_group_R])
    _group_standard_valid = np.isfinite(_group_standard_x) & np.isfinite(_group_standard_y)
    fig.add_trace(
        go.Scatter(
            x=_group_standard_x[_group_standard_valid],
            y=_group_standard_y[_group_standard_valid],
            mode='markers',
            marker=dict(
                size=DOT_SIZE,
                opacity=DOT_OPACITY,
                color="black",
            ),
            showlegend=False,
        )
    )

    fig.update_xaxes(
        scaleanchor="y",
        scaleratio=1,
        zeroline=True,
        tickvals=[-1,-.5,0,.5,1],
        showticklabels = False,
        gridcolor="lightgrey",
        zerolinecolor="black"
    )
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
        zeroline=True,
        tickvals=[-.5,0,.5,1],
        showticklabels = False,
        gridcolor="lightgrey",
        zerolinecolor="black"
    )
    fig.update_layout(
        width = .75*100,
        height = .75*100,
        xaxis_range=[-.5,1.1],
        yaxis_range=[-.5,1.1],
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0,r=0,t=0,b=0),
    )

    base_name_group_standard = "group_similarity_predictions_vs_ceiling_nonrobust"
    fig.write_image(
        str(args.output_dir / f"{base_name_group_standard}.png"),
        scale=12,
        width=.75 * 100,
        height=.75 * 100,
    )
    # fig.write_html(str(args.output_dir / f"{base_name_group_standard}.html", include_plotlyjs=False), include_plotlyjs=False)

    #-------------------------------------------------------------------------------
    # End Figure: Group similarity between predictions vs ceiling (standard model)
    #-------------------------------------------------------------------------------

    #-------------------------------------------------------------------------------
    # Figure: Individual similarity between predictions vs ceiling (robust model)
    #-------------------------------------------------------------------------------

    # ccc_corrs_ceiling_individual_robust = concordance_correlation_coefficient(
    #     c_corrs_between_predictions_individual,
    #     ceiling_corrs_individual_robust,
    # )
    # print("CCC correlation between predictions and ceiling (robust model, individual):")
    # for k,v in ccc_corrs_ceiling_individual_robust.items():
    #     print(f"  {k}: {v:.3f}")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[-.5,1.1],
            y=[-.5,1.1],
            mode='lines',
            line=dict(
                color="cyan",
                dash="dash"
            ),
            showlegend=False
        )
    )
    _individual_robust_x = np.concatenate([ceiling_corrs_individual_robust_L, ceiling_corrs_individual_robust_R])
    _individual_robust_y = np.concatenate([c_corrs_between_predictions_individual_L, c_corrs_between_predictions_individual_R])
    _individual_robust_valid = np.isfinite(_individual_robust_x) & np.isfinite(_individual_robust_y)
    fig.add_trace(
        go.Scatter(
            x=_individual_robust_x[_individual_robust_valid],
            y=_individual_robust_y[_individual_robust_valid],
            mode='markers',
            marker=dict(
                size=DOT_SIZE,
                opacity=DOT_OPACITY,
                color="black",
            ),
            showlegend=False,
        )
    )
    fig.update_xaxes(
        scaleanchor="y",
        scaleratio=1,
        zeroline=True,
        tickvals=[-1,-.5,0,.5,1],
        showticklabels = False,
        gridcolor="lightgrey",
        zerolinecolor="black"
    )
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
        zeroline=True,
        tickvals=[-1,-.5,0,.5,1],
        showticklabels = False,
        gridcolor="lightgrey",
        zerolinecolor="black"
    )
    fig.update_layout(
        width = .75*100,
        height = .75*100,
        xaxis_range=[-1,1.1],
        yaxis_range=[-1,1.1],
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0,r=0,t=0,b=0),
    )
    base_name_individual_robust = "individual_similarity_predictions_vs_ceiling_robust_S30"
    fig.write_image(
        str(args.output_dir / f"{base_name_individual_robust}.png"),
        scale=12,
        width=.75 * 100,
        height=.75 * 100,
    )
    # fig.write_html(str(args.output_dir / f"{base_name_individual_robust}.html", include_plotlyjs=False), include_plotlyjs=False)
    #-------------------------------------------------------------------------------
    # End Figure: Individual similarity between predictions vs ceiling (robust model)
    #-------------------------------------------------------------------------------

    #-------------------------------------------------------------------------------
    # Figure: Individual similarity between predictions vs ceiling (standard model)
    #-------------------------------------------------------------------------------
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[-1,1.1],
            y=[-1,1.1],
            mode='lines',
            line=dict(
                color="cyan",
                dash="dash"
            ),
            showlegend=False
        )
    )
    _individual_standard_x = np.concatenate([ceiling_corrs_individual_standard_L, ceiling_corrs_individual_standard_R])
    _individual_standard_y = np.concatenate([c_corrs_between_predictions_individual_L, c_corrs_between_predictions_individual_R])
    _individual_standard_valid = np.isfinite(_individual_standard_x) & np.isfinite(_individual_standard_y)
    fig.add_trace(
        go.Scatter(
            x=_individual_standard_x[_individual_standard_valid],
            y=_individual_standard_y[_individual_standard_valid],
            mode='markers',
            marker=dict(
                size=DOT_SIZE,
                opacity=DOT_OPACITY,
                color="black",
            ),
            showlegend=False,
        )
    )
    fig.update_xaxes(
        scaleanchor="y",
        scaleratio=1,
        zeroline=True,
        tickvals=[-1,-.5,0,.5,1],
        showticklabels = False,
        gridcolor="lightgrey",
        zerolinecolor="black"
    )
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
        zeroline=True,
        tickvals=[-1,-.5,0,.5,1],
        showticklabels = False,
        gridcolor="lightgrey",
        zerolinecolor="black"
    )
    fig.update_layout(
        width = .75*100,
        height = .75*100,
        xaxis_range=[-1,1.1],
        yaxis_range=[-1,1.1],
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0,r=0,t=0,b=0),
    )
    base_name_individual_standard = "individual_similarity_predictions_vs_ceiling_nonrobust_S30"
    fig.write_image(
        str(args.output_dir / f"{base_name_individual_standard}.png"),
        scale=12,
        width=.75 * 100,
        height=.75 * 100,
    )
    # fig.write_html(str(args.output_dir / f"{base_name_individual_standard}.html", include_plotlyjs=False), include_plotlyjs=False)


    corrs_between_models_L = masked_corr(D_group_natural_robust1_L,D_group_natural_standard2_L,axis=1)
    ceiling_corrs_L = masked_corr(D_group_natural_robust1_L,D_group_natural_robust2_L,axis=1)
    
    corrs_between_models_fsaverage = np.array(
        [stat2grid(corrs_between_models_L[i], natsounds) for i in range(N_SUBJECTS)]
    )
    ceiling_corrs_fsaverage = np.array(
        [stat2grid(ceiling_corrs_L[i], natsounds) for i in range(N_SUBJECTS)]
    )
    
    
    
    print(f"Median correlation between model predictions (all voxels): {np.nanmedian(corrs_between_models_fsaverage)}")
    print(f"Median ceiling correlation (all voxels): {np.nanmedian(ceiling_corrs_fsaverage)}")

    opacity = 1
    boolean_masks = load_and_clean_roi_masks(args.masks_dir, reliability_mask_L)

    selected_ROIs = list(boolean_masks['L'].keys())
    excluded_ROIs = []
    
    ROI_colors = {
        "Medial Heschl's gyrus": "rgb(49, 40, 123)",
        "Lateral Heschl's gyrus": "rgb(92, 132, 193)",
        "Planum polare": "rgb(187, 238, 102)",
        "Planum temporale": "rgb(233, 123, 33)",
        "Superior temporal gyrus": "rgb(134, 30, 26)",
    }
    
    def roi_summary_figure(
        quantity1_L: np.ndarray,
        quantity1_R: np.ndarray,
        quantity2_L: np.ndarray,
        quantity2_R: np.ndarray,
        fname: str | None = None,
        texture: str = "x",
    ) -> None:
        fig = go.Figure()
    
        plot_rois = list(ROI_colors.keys())
        
        dx = 0.18           # horizontal offset from ROI center
        bar_width = 0.32    # must be <= 2*dx to avoid overlap
        roi_center_x = []   # for ticks
        
        t_stats = []
        p_values = []
        for roi_idx, roi in enumerate(ROI_colors):
            if roi in excluded_ROIs:
                continue
        
            if roi == "Reliable":
                raise NotImplementedError("Reliable ROI not implemented")

            roi_bool_L = boolean_masks['L'][roi].astype(bool)
            roi_bool_R = boolean_masks['R'][roi].astype(bool)
            standard_vals_L = np.array([np.nanmedian(quantity1_L[i, subject_masks_L[i] & roi_bool_L]) for i in range(N_SUBJECTS)])
            standard_vals_R = np.array([np.nanmedian(quantity1_R[i, subject_masks_R[i] & roi_bool_R]) for i in range(N_SUBJECTS)])
            robust_vals_L    = np.array([np.nanmedian(quantity2_L[i, subject_masks_L[i] & roi_bool_L]) for i in range(N_SUBJECTS)])
            robust_vals_R    = np.array([np.nanmedian(quantity2_R[i, subject_masks_R[i] & roi_bool_R]) for i in range(N_SUBJECTS)])
            standard_vals = (standard_vals_L + standard_vals_R) / 2
            robust_vals = (robust_vals_L + robust_vals_R) / 2

            # paired t-test
            t_stat, p_value = scipy.stats.ttest_rel(standard_vals, robust_vals)
            
            t_stats.append(t_stat)
            p_values.append(p_value)
        
            standard_mean = float(np.nanmean(standard_vals))
            robust_mean    = float(np.nanmean(robust_vals))
        
            x_non = roi_idx - dx
            x_rob = roi_idx + dx
            roi_center_x.append(roi_idx)
        
            # lines connecting each subject (standard -> robust)
            for nv, rv in zip(standard_vals, robust_vals):
                fig.add_trace(go.Scatter(
                    x=[x_non, x_rob],
                    y=[nv, rv],
                    mode='lines',
                    line=dict(color='rgba(50,50,50,0.8)', width=0.75),
                    hoverinfo='skip',
                    showlegend=False
                ))
        
            # subject dots
            fig.add_trace(go.Scatter(
                x=[x_non] * len(standard_vals),
                y=standard_vals,
                mode='markers',
                marker=dict(color=ROI_colors[roi], size=6, line=dict(width=0.25, color='black')),
                name=f"{roi}_standard",
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=[x_rob] * len(robust_vals),
                y=robust_vals,
                mode='markers',
                marker=dict(color=ROI_colors[roi], size=6, line=dict(width=0.25, color='black')),
                name=f"{roi}_robust",
                showlegend=False
            ))
        
            # bars (vertical)
            fig.add_trace(go.Bar(
                x=[x_non],
                y=[standard_mean],
                marker=dict(color=ROI_colors[roi], line=dict(color='black', width=1)),
                width=bar_width,
                showlegend=False
            ))
            fig.add_trace(go.Bar(
                x=[x_rob],
                y=[robust_mean],
                marker=dict(color=ROI_colors[roi], line=dict(color='black', width=1)),
                width=bar_width,
                showlegend=False,
                marker_pattern_shape=texture
            ))
        
        # layout
        fig.update_layout(
            width=300 * 2,
            height=300 * 1.5,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=0, b=0),
            font=dict(size=10, family="Times New Roman"),
            barmode='overlay',  # bars are at different x offsets, so overlay is fine
            yaxis=dict(
                zeroline=True,
                zerolinecolor="black",
                gridcolor="lightgrey",
                tickvals=[0, 0.25, 0.5, 0.75],
                showticklabels=False,
                range=[-0.1, 0.9],
                title=None
            ),
            xaxis=dict(
                tickvals=list(range(len(ROI_colors))),
                ticktext=list(ROI_colors.keys()),
                showticklabels=True,
                tickangle=45,
                range=[-0.6, len(ROI_colors)-0.4],  # breathing room at edges
                title=None,
                visible=False
            )
        )
        if fname is not None:
            fig.write_image(fname, scale=12, width=100 * 2.75, height=100 * 3.25)
        fig.update_layout(
            xaxis=dict(visible=True)
        )
        corrected = scipy.stats.false_discovery_control(np.array(p_values),method="by")
        for roi, t_stat, p_value, corrected_p_value in zip(ROI_colors.keys(), t_stats, p_values, corrected):
            print(f"Subject-level paired t-test ({roi}): t={t_stat:.3f}, p={p_value:.3f}, corrected p={corrected_p_value:.3f}\n")

    observed_standard_pred_acc_fsaverage_L = np.array(
        [
            stat2grid(
                masked_corr(D_group_natural_observed_L, D_group_natural_standard1_L, axis=1)[i],
                natsounds,
            )
            for i in range(N_SUBJECTS)
        ]
    )
    observed_robust_pred_acc_fsaverage_L = np.array(
        [
            stat2grid(
                masked_corr(D_group_natural_observed_L, D_group_natural_robust1_L, axis=1)[i],
                natsounds,
            )
            for i in range(N_SUBJECTS)
        ]
    )

    observed_standard_pred_acc_fsaverage_R = np.array(
        [
            stat2grid(
                masked_corr(D_group_natural_observed_R, D_group_natural_standard1_R, axis=1)[i],
                natsounds,
                hemi="R"
            )
            for i in range(N_SUBJECTS)
        ]
    )
    observed_robust_pred_acc_fsaverage_R = np.array(
        [
            stat2grid(
                masked_corr(D_group_natural_observed_R, D_group_natural_robust1_R, axis=1)[i],
                natsounds,
                hemi="R"
            )
            for i in range(N_SUBJECTS)
        ]
    )
    corrs_between_models_fsaverage_L = np.array(
        [
            stat2grid(
                masked_corr(D_group_natural_robust1_L, D_group_natural_standard2_L, axis=1)[i],
                natsounds,
                hemi="L"
            )
            for i in trange(N_SUBJECTS)
        ]
    )
    ceiling_corrs_fsaverage_L = np.array(
        [
            stat2grid(
                masked_corr(D_group_natural_robust1_L, D_group_natural_robust2_L, axis=1)[i],
                natsounds,
                hemi="L"
            )
            for i in trange(N_SUBJECTS)
        ]
    )
    corrs_between_models_fsaverage_R = np.array(
        [
            stat2grid(
                masked_corr(D_group_natural_robust1_R, D_group_natural_standard2_R, axis=1)[i],
                natsounds,
                hemi="R"
            )
            for i in trange(N_SUBJECTS)
        ]
    )
    ceiling_corrs_fsaverage_R = np.array(
        [
            stat2grid(
                masked_corr(D_group_natural_robust1_R, D_group_natural_robust2_R, axis=1)[i],
                natsounds,
                hemi="R"
            )
            for i in trange(N_SUBJECTS)
        ]
    )



    print("Subject-level paired t-test (prediction accuracy):")
    roi_summary_figure(
        observed_robust_pred_acc_fsaverage_L,
        observed_robust_pred_acc_fsaverage_R,
        observed_standard_pred_acc_fsaverage_L,
        observed_standard_pred_acc_fsaverage_R,
        fname=str(args.output_dir / "roi_summary_pred_acc_nonrobust_robust.png"),
    )
    print("Subject-level paired t-test (correlation between predictions and ceiling):")
    roi_summary_figure(
        corrs_between_models_fsaverage_L,
        corrs_between_models_fsaverage_R,
        ceiling_corrs_fsaverage_L,
        ceiling_corrs_fsaverage_R,
        fname=str(args.output_dir / "roi_summary_corr_between_ceiling.png"),
        texture=".",
    )

    #-------------------------------------------------------------------------------
    # Figure: ROI surface map and ROI-based summaries
    #-------------------------------------------------------------------------------

    ROI_colors = {
        "Medial Heschl's gyrus":"rgb(49, 40, 123)",
        "Lateral Heschl's gyrus":"rgb(92, 132, 193)",
        "Planum polare":"rgb(187, 238, 102)",
        "Planum temporale":"rgb(233, 123, 33)",
        "Superior temporal gyrus":"rgb(134, 30, 26)",
    }

    # Create a list of colors in order
    
    def rgba_string_to_mpl(rgba_string):
        """Convert 'rgba(245,34,10,1)' to matplotlib RGBA tuple"""
        # Remove 'rgba(' and ')'
        values = rgba_string.strip('rgba()').split(',')    
        # Convert to floats
        r, g, b = [float(v) for v in values]
        return (r/255.0, g/255.0, b/255.0)
        
    colors = [rgba_string_to_mpl(ROI_colors[roi]) for roi in ROI_colors]
    
    # Create the colormap
    cmap = ListedColormap(colors)
    
    # Define boundaries for discrete values (0, 1, 2, 3, 4, 5)
    bounds = [0, 1, 2, 3, 4, 5]
    norm = BoundaryNorm(bounds, cmap.N)
    
    # Value mapping
    value_mapping = {
        roi: i for (i,roi) in enumerate(ROI_colors)
    }
    
    # print("Value to ROI mapping:")
    # for value, roi_name in value_mapping.items():
    #     print(f"  {value}: {roi_name}")
    
    colors_roi_map = np.full(163842,np.nan)
    for roi in value_mapping:
        roi_bool = boolean_masks['L'][roi]
        colors_roi_map[roi_bool] = value_mapping[roi]
        
    fig = plotting.plot_surf_stat_map(
            inflated_L,
            colors_roi_map,
            bg_map=bg_L,
            engine="plotly",
            cmap=cmap,
            colorbar=False,
            hemi="left"
            )
    base_name_roi_map = "ROI_map_L"
    fig.figure.write_image(str(args.output_dir / f"{base_name_roi_map}.png"), scale=4)
    # fig.figure.write_html(str(args.output_dir / f"{base_name_roi_map}.html"))

    # value_mapping


    
    # colors_roi_map = np.full(163842,np.nan)
    # for roi in figure3_selected_ROIs:
    #     roi_bool = boolean_masks['L'][roi]
    #     colors_roi_map[roi_bool] = value_mapping[roi]
        
    # from nilearn import datasets, plotting, surface
    # fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage')
    # inflated_L = fsaverage.infl_left
    # inflated_R = fsaverage.infl_right
    # curv_L = nilearn.surface.load_surf_data(fsaverage.curv_left)
    # curv_R = nilearn.surface.load_surf_data(fsaverage.curv_right)
    # bg_L = np.array([0.9 if c > 0 else 0.5 for c in curv_L])
    # bg_R = np.array([0.9 if c > 0 else 0.5 for c in curv_R])
    
    
    # fig = plotting.plot_surf_stat_map(
    #         inflated_L,
    #         colors_roi_map,
    #         bg_map=bg_L,
    #         engine="plotly",
    #         cmap=cmap,
    #         colorbar=False,
    #         hemi="left",
    #         vmin=0,vmax=5,
    #         )
    # fig.figure.write_html(str(args.output_dir / "plot_7.html"))
    # fig.figure.write_image("figure3/ROI_map_L.png")

    #-------------------------------------------------------------------------------
    # End Figure: ROI surface map and ROI-based summaries
    #-------------------------------------------------------------------------------

    #-------------------------------------------------------------------------------
    # Figure: ROI prediction accuracy boxplots (standard vs robust)
    #-------------------------------------------------------------------------------

    # fig = go.Figure()
    # for roi in ROI_colors:
    #     if roi in excluded_ROIs:
    #         continue
    #     mask_ = boolean_masks['L'][roi]
        
    #     # Add non-robust box plot
    #     fig.add_trace(go.Box(
    #         x0=roi,  # Use x instead of x0 for proper grouping
    #         y=observed_standard_pred_acc_fsaverage[:,mask_].mean(0),
    #         name=f"{roi}_standard",
    #         fillcolor=ROI_colors[roi],
    #         line=dict(color="black"),
    #         showlegend=False,
    #         offsetgroup=0,  # Group boxes together
    #         legendgroup=roi  # Group in legend
    #     ))
        
    #     # Add robust box plot
    #     fig.add_trace(go.Box(
    #         x0=roi,  # Use x instead of x0 for proper grouping
    #         y=observed_robust_pred_acc_fsaverage[:,mask_].mean(0),
    #         name=f"{roi}_robust",
    #         fillcolor=ROI_colors[roi],
    #         line=dict(color="black"),
    #         showlegend=False,
    #         offsetgroup=1,  # Group boxes together
    #         legendgroup=roi  # Group in legend
    #     ))
    
    #     # fig.add_trace(go.Scatter(
    #     #     x=[roi]*20,  # Use x instead of x0 for proper grouping
    #     #     y=observed_robust_pred_acc_fsaverage[:,mask_].mean(1),
    #     #     showlegend=False,
    #     #     mode='markers',
    #     # ))
        
    
    # fig.update_layout(
    #     width=300 * 2,
    #     height=300 * 1.5,
    #     plot_bgcolor="rgba(0,0,0,0)",
    #     paper_bgcolor="rgba(0,0,0,0)",
    #     boxmode='group',  # This groups boxes by x-axis position
    #     yaxis=dict(
    #         zeroline=True,
    #         zerolinecolor="black",
    #         gridcolor="lightgrey",
    #         tickvals=[0, 0.25, 0.5, 0.75],
    #         showticklabels=False,
    #         range=[-.1,.9],
    #     ),
    #     xaxis=dict(
    #         showticklabels=False,  # Show ROI names
    #         tickangle=45  # Rotate labels for better readability
    #     ),
    #     margin=dict(l=0, r=0, t=0, b=0),
    #     font=dict(
    #         size=10,
    #         family="Times New Roman"
    #     )
    # )
    # base_name_roi_pred = "roi_summary_pred_acc_nonrobust_robust"
    # fig.write_image(str(args.output_dir / f"{base_name_roi_pred}.png"), scale=12, width=2 * 300, height=1.5 * 300)
    # fig.update_layout(
    #     xaxis=dict(
    #         showticklabels=True,  # Show ROI names
    #         tickangle=45  # Rotate labels for better readability
    #     ),
    # )
    # # fig.write_html(str(args.output_dir / f"{base_name_roi_pred}.html", include_plotlyjs=False), include_plotlyjs=False)

    # #-------------------------------------------------------------------------------
    # # End Figure: ROI prediction accuracy boxplots (standard vs robust)
    # #-------------------------------------------------------------------------------

    # #-------------------------------------------------------------------------------
    # # Figure: ROI correlations boxplots (between-model vs ceiling)
    # #-------------------------------------------------------------------------------

    # fig = go.Figure()
    # for roi in ROI_colors:
    #     if roi in excluded_ROIs:
    #         continue
    #     mask_ = boolean_masks['L'][roi]
        
    #     # Add non-robust box plot
    #     fig.add_trace(go.Box(
    #         x0=roi,  # Use x instead of x0 for proper grouping
    #         y=corrs_between_models_fsaverage[:,mask_].mean(0),
    #         name=f"{roi}_standard",
    #         fillcolor=ROI_colors[roi],
    #         line=dict(color="black"),
    #         showlegend=False,
    #         offsetgroup=0,  # Group boxes together
    #         legendgroup=roi  # Group in legend
    #     ))
        
    #     # Add robust box plot
    #     fig.add_trace(go.Box(
    #         x0=roi,  # Use x instead of x0 for proper grouping
    #         y=ceiling_corrs_fsaverage[:,mask_].mean(0),
    #         name=f"{roi}_robust",
    #         fillcolor=ROI_colors[roi],
    #         line=dict(color="black"),
    #         showlegend=False,
    #         offsetgroup=1,  # Group boxes together
    #         legendgroup=roi  # Group in legend
    #     ))
    
    #     # fig.add_trace(go.Scatter(
    #     #     x=[roi]*20,  # Use x instead of x0 for proper grouping
    #     #     y=observed_robust_pred_acc_fsaverage[:,mask_].mean(1),
    #     #     showlegend=False,
    #     #     mode='markers',
    #     # ))
        
    
    # fig.update_layout(
    #     width=300 * 2,
    #     height=300 * 1.5,
    #     plot_bgcolor="rgba(0,0,0,0)",
    #     paper_bgcolor="rgba(0,0,0,0)",
    #     boxmode='group',  # This groups boxes by x-axis position
        
    #     yaxis=dict(
    #         zeroline=True,
    #         zerolinecolor="black",
    #         gridcolor="lightgrey",
    #         tickvals=[0, 0.25, 0.5, 0.75],
    #         showticklabels=False,
    #         range=[-.1,.9],
            
    #     ),
    #     font=dict(
    #         size=10,
    #         family="Times New Roman"
    #     ),
    #     xaxis=dict(
    #         showticklabels=False,  # Show ROI names
    #         tickangle=45  # Rotate labels for better readability
    #     ),
    #     margin=dict(l=0, r=0, t=0, b=0),
        
    # )
    # base_name_roi_corr = "roi_summary_corr_between_ceiling"
    # fig.write_image(str(args.output_dir / f"{base_name_roi_corr}.png"), scale=12, width=2 * 300, height=1.5 * 300)
    # # fig.write_html(str(args.output_dir / f"{base_name_roi_corr}.html", include_plotlyjs=False), include_plotlyjs=False)

    # #-------------------------------------------------------------------------------
    # # End Figure: ROI correlations boxplots (between-model vs ceiling)
    # #-------------------------------------------------------------------------------

    # #-------------------------------------------------------------------------------
    # # Figure: Variance histograms for individual robust vs standard models
    # #-------------------------------------------------------------------------------

    # fig, ax = plt.subplots(1, 1)
    # ax.hist(D_individual_natural_robust1.var(0), bins=100, alpha=0.5, label="robust1")
    # ax.hist(D_individual_natural_standard1.var(0), bins=100, alpha=0.5, label="standard1")
    # ax.legend()
    # # ax.hist(D_individual_natural_robust2.var(0),bins=100)
    # # ax.hist(D_individual_natural_standard2.var(0),bins=100)
    # variance_fig_path = args.output_dir / "variance_histograms.png"
    # fig.savefig(variance_fig_path, dpi=300, bbox_inches="tight")

    # #-------------------------------------------------------------------------------
    # # End Figure: Variance histograms for individual robust vs standard models
    # #-------------------------------------------------------------------------------

    # --- Individual subject S29 (index -2 in the group arrays) ---
    # Prediction accuracy brain maps
    plot_brain(
        standard_prediction_accuracy_fsaverage_L[-2],
        vmin=-1, vmax=1, cmap="RdBu_r",
        fname=str(args.output_dir / "standard_prediction_accuracy_individual_S29_L.png"),
    )
    plot_brain(
        AR_prediction_accuracy_fsaverage_L[-2],
        vmin=-1, vmax=1, cmap="RdBu_r",
        fname=str(args.output_dir / "AR_prediction_accuracy_individual_S29_L.png"),
    )
    plot_brain(
        standard_prediction_accuracy_fsaverage_R[-2],
        vmin=-1, vmax=1, cmap="RdBu_r",
        fname=str(args.output_dir / "standard_prediction_accuracy_individual_S29_R.png"),
        hemi="R",
    )
    plot_brain(
        AR_prediction_accuracy_fsaverage_R[-2],
        vmin=-1, vmax=1, cmap="RdBu_r",
        fname=str(args.output_dir / "AR_prediction_accuracy_individual_S29_R.png"),
        hemi="R",
    )

    # Prediction accuracy scatter (S29)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[-1, 1], y=[-1, 1], mode='lines',
        line=dict(color="cyan", dash="dash"), showlegend=False))
    _s29_acc_x = np.concatenate([standard_prediction_accuracy_L[-2, :], standard_prediction_accuracy_R[-2, :]])
    _s29_acc_y = np.concatenate([AR_prediction_accuracy_L[-2, :], AR_prediction_accuracy_R[-2, :]])
    _s29_acc_valid = np.isfinite(_s29_acc_x) & np.isfinite(_s29_acc_y)
    fig.add_trace(go.Scatter(
        x=_s29_acc_x[_s29_acc_valid],
        y=_s29_acc_y[_s29_acc_valid],
        mode='markers', marker=dict(size=DOT_SIZE, opacity=DOT_OPACITY, color="black"),
        showlegend=False))
    fig.update_xaxes(scaleanchor="y", scaleratio=1, zeroline=True,
        tickvals=[-.5, 0, .5, 1], showticklabels=False,
        gridcolor="lightgrey", zerolinecolor="black")
    fig.update_yaxes(scaleanchor="x", scaleratio=1, zeroline=True,
        tickvals=[-.5, 0, .5, 1], showticklabels=False,
        gridcolor="lightgrey", zerolinecolor="black")
    fig.update_layout(width=.75 * 100, height=.75 * 100,
        xaxis_range=[-.5, 1.1], yaxis_range=[-.5, 1.1],
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0))
    fig.write_image(str(args.output_dir / "individual_prediction_accuracy_S29.png"),
        scale=12, width=.75 * 100, height=.75 * 100)

    # Similarity/ceiling scatters (S29) — load individual data
    _s29_r1  = slice_and_dice(data, 'robust1',   test_individual='S29')
    _s29_r2  = slice_and_dice(data, 'robust2',   test_individual='S29')
    _s29_nr1 = slice_and_dice(data, 'standard1', test_individual='S29')
    _s29_nr2 = slice_and_dice(data, 'standard2', test_individual='S29')
    c_corrs_S29_L        = masked_corr(_s29_r1[0], _s29_nr2[0], axis=0)
    c_corrs_S29_R        = masked_corr(_s29_r1[1], _s29_nr2[1], axis=0)
    ceiling_robust_S29_L    = masked_corr(_s29_r1[0], _s29_r2[0], axis=0)
    ceiling_robust_S29_R    = masked_corr(_s29_r1[1], _s29_r2[1], axis=0)
    ceiling_standard_S29_L = masked_corr(_s29_nr1[0], _s29_nr2[0], axis=0)
    ceiling_standard_S29_R = masked_corr(_s29_nr1[1], _s29_nr2[1], axis=0)

    # Between-prediction and ceiling brain maps (S29)
    plot_brain(
        stat2grid(c_corrs_S29_L, natsounds, hemi='L'),
        vmin=-1, vmax=1, cmap="PiYG",
        fname=str(args.output_dir / "between_predictions_individual_S29_L.png"),
    )
    plot_brain(
        stat2grid(c_corrs_S29_R, natsounds, hemi='R'),
        vmin=-1, vmax=1, cmap="PiYG",
        fname=str(args.output_dir / "between_predictions_individual_S29_R.png"),
        hemi="R",
    )
    plot_brain(
        stat2grid(ceiling_robust_S29_L, natsounds, hemi='L'),
        vmin=-1, vmax=1, cmap="PiYG",
        fname=str(args.output_dir / "ceiling_individual_robust_S29_L.png"),
    )
    plot_brain(
        stat2grid(ceiling_robust_S29_R, natsounds, hemi='R'),
        vmin=-1, vmax=1, cmap="PiYG",
        fname=str(args.output_dir / "ceiling_individual_robust_S29_R.png"),
        hemi="R",
    )
    plot_brain(
        stat2grid(ceiling_standard_S29_L, natsounds, hemi='L'),
        vmin=-1, vmax=1, cmap="PiYG",
        fname=str(args.output_dir / "ceiling_individual_nonrobust_S29_L.png"),
    )
    plot_brain(
        stat2grid(ceiling_standard_S29_R, natsounds, hemi='R'),
        vmin=-1, vmax=1, cmap="PiYG",
        fname=str(args.output_dir / "ceiling_individual_nonrobust_S29_R.png"),
        hemi="R",
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[-.5, 1.1], y=[-.5, 1.1], mode='lines',
        line=dict(color="cyan", dash="dash"), showlegend=False))
    _s29_robust_ceiling_x = np.concatenate([ceiling_robust_S29_L, ceiling_robust_S29_R])
    _s29_robust_ceiling_y = np.concatenate([c_corrs_S29_L, c_corrs_S29_R])
    _s29_robust_ceiling_valid = np.isfinite(_s29_robust_ceiling_x) & np.isfinite(_s29_robust_ceiling_y)
    fig.add_trace(go.Scatter(
        x=_s29_robust_ceiling_x[_s29_robust_ceiling_valid],
        y=_s29_robust_ceiling_y[_s29_robust_ceiling_valid],
        mode='markers', marker=dict(size=DOT_SIZE, opacity=DOT_OPACITY, color="black"),
        showlegend=False))
    fig.update_xaxes(scaleanchor="y", scaleratio=1, zeroline=True,
        tickvals=[-1, -.5, 0, .5, 1], showticklabels=False,
        gridcolor="lightgrey", zerolinecolor="black")
    fig.update_yaxes(scaleanchor="x", scaleratio=1, zeroline=True,
        tickvals=[-1, -.5, 0, .5, 1], showticklabels=False,
        gridcolor="lightgrey", zerolinecolor="black")
    fig.update_layout(width=.75 * 100, height=.75 * 100,
        xaxis_range=[-1, 1.1], yaxis_range=[-1, 1.1],
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0))
    fig.write_image(str(args.output_dir / "individual_similarity_predictions_vs_ceiling_robust_S29.png"),
        scale=12, width=.75 * 100, height=.75 * 100)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[-1, 1.1], y=[-1, 1.1], mode='lines',
        line=dict(color="cyan", dash="dash"), showlegend=False))
    _s29_standard_ceiling_x = np.concatenate([ceiling_standard_S29_L, ceiling_standard_S29_R])
    _s29_standard_ceiling_y = np.concatenate([c_corrs_S29_L, c_corrs_S29_R])
    _s29_standard_ceiling_valid = np.isfinite(_s29_standard_ceiling_x) & np.isfinite(_s29_standard_ceiling_y)
    fig.add_trace(go.Scatter(
        x=_s29_standard_ceiling_x[_s29_standard_ceiling_valid],
        y=_s29_standard_ceiling_y[_s29_standard_ceiling_valid],
        mode='markers', marker=dict(size=DOT_SIZE, opacity=DOT_OPACITY, color="black"),
        showlegend=False))
    fig.update_xaxes(scaleanchor="y", scaleratio=1, zeroline=True,
        tickvals=[-1, -.5, 0, .5, 1], showticklabels=False,
        gridcolor="lightgrey", zerolinecolor="black")
    fig.update_yaxes(scaleanchor="x", scaleratio=1, zeroline=True,
        tickvals=[-1, -.5, 0, .5, 1], showticklabels=False,
        gridcolor="lightgrey", zerolinecolor="black")
    fig.update_layout(width=.75 * 100, height=.75 * 100,
        xaxis_range=[-1, 1.1], yaxis_range=[-1, 1.1],
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0))
    fig.write_image(str(args.output_dir / "individual_similarity_predictions_vs_ceiling_nonrobust_S29.png"),
        scale=12, width=.75 * 100, height=.75 * 100)


if __name__ == '__main__':
    main()

