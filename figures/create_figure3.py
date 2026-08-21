#!/usr/bin/env python3
"""
Create Figure3

Converted from create_figure3.ipynb
"""

import argparse
from pathlib import Path
import nilearn
import scipy.stats

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('../data'),
                      help='Base data directory')
    parser.add_argument('--masks-dir', type=Path, default=Path('../masks'),
                      help='Base masks directory')
    parser.add_argument('--components-dir', type=Path, default=Path('../components'),
                      help='Base components directory')
    parser.add_argument('--output-dir', type=Path, default=Path('output_create_figure3'),
                      help='Output directory for figures')
    parser.add_argument('--log-file', type=Path, default=None,
                      help='Path to stats log file (default: <output-dir>/stats.log)')
    return parser.parse_args()

def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import sys
    sys.path.append('../scripts')

    class _Tee:
        def __init__(self, *streams): self._s = streams
        def write(self, d): [s.write(d) for s in self._s]
        def flush(self): [s.flush() for s in self._s]

    _log_path = args.log_file if args.log_file is not None else args.output_dir / 'stats.log'
    _log_fh = open(_log_path, 'w')
    sys.stdout = _Tee(sys.__stdout__, _log_fh)

    print(f"Output directory: {args.output_dir}")

    import numpy as np
    from lib.data import load_data, load_dict_h5
    from lib.utils import mycorr, demean, z_score, identity, concordance_correlation_coefficient
    from lib.utils import plot_brain, set_plot_area, color_lookup
    from lib.utils import NPD_color, natural_color
    from lib.analysis import calculate_split_half_reliability
    from lib.analysis.reliability import reliability_b2021
    import plotly.graph_objects as go

    data = load_data(
        filename=str(args.data_dir / 'new-subjects-clean/observed_and_predicted_data_new_subjects_smoothed_stratified.npz'),
        components=str(args.components_dir / "components_reordered_stratified.h5"),
    )
    masks = load_dict_h5(str(args.masks_dir / "ROI_masks.h5"))
    np.unicode_ = np.str_
    stim_names = np.array(data['stimuli_names'])


    data.keys()

    selected_indices = np.concatenate([np.arange(60), np.arange(120, 180)])

    #---------------------------------------------------------------------------
    # Load data for both hemispheres
    #---------------------------------------------------------------------------
    D_observed_individual = {}
    D_observed_individual2 = {}
    D_predicted_robust_individual = {}
    D_predicted_standard_individual = {}
    D_predicted_robust_individual2 = {}
    D_predicted_standard_individual2 = {}
    D_observed_group = {}
    D_predicted_robust_group = {}
    D_predicted_standard_group = {}

    for hemi in ['L', 'R']:
        D_observed_individual[hemi] = data['observed']['sub-MRI009'][hemi].mean(0)
        D_observed_individual2[hemi] = data['observed']['sub-MRI004'][hemi].mean(0)
        D_predicted_robust_individual[hemi] = data['predictions']['sub-MRI009']['robust'][hemi]
        D_predicted_standard_individual[hemi] = data['predictions']['sub-MRI009']['standard'][hemi]
        D_predicted_robust_individual2[hemi] = data['predictions']['sub-MRI004']['robust'][hemi]
        D_predicted_standard_individual2[hemi] = data['predictions']['sub-MRI004']['standard'][hemi]
        D_observed_group[hemi] = np.stack([data['observed'][s][hemi].mean(0) for s in data['observed'].keys()], 0)
        D_predicted_robust_group[hemi] = np.stack([data['predictions'][s]['robust'][hemi] for s in data['observed'].keys()], 0)
        D_predicted_standard_group[hemi] = np.stack([data['predictions'][s]['standard'][hemi] for s in data['observed'].keys()], 0)

    npd_indices = np.arange(60)
    natural_indices = np.arange(120, 180)

    corrs = {}
    for hemi in ['L', 'R']:
        corrs[hemi] = {}
        corrs[hemi]['observed_robust_individual_natural'] = mycorr(D_observed_individual[hemi][natural_indices], D_predicted_robust_individual[hemi][natural_indices], 0)
        corrs[hemi]['observed_robust_individual2_natural'] = mycorr(D_observed_individual2[hemi][natural_indices], D_predicted_robust_individual2[hemi][natural_indices], 0)
        corrs[hemi]['observed_standard_individual_natural'] = mycorr(D_observed_individual[hemi][natural_indices], D_predicted_standard_individual[hemi][natural_indices], 0)
        corrs[hemi]['observed_standard_individual2_natural'] = mycorr(D_observed_individual2[hemi][natural_indices], D_predicted_standard_individual2[hemi][natural_indices], 0)
        corrs[hemi]['observed_robust_individual_npd'] = mycorr(D_observed_individual[hemi][npd_indices], D_predicted_robust_individual[hemi][npd_indices], 0)
        corrs[hemi]['observed_robust_individual2_npd'] = mycorr(D_observed_individual2[hemi][npd_indices], D_predicted_robust_individual2[hemi][npd_indices], 0)
        corrs[hemi]['observed_standard_individual_npd'] = mycorr(D_observed_individual[hemi][npd_indices], D_predicted_standard_individual[hemi][npd_indices], 0)
        corrs[hemi]['observed_standard_individual2_npd'] = mycorr(D_observed_individual2[hemi][npd_indices], D_predicted_standard_individual2[hemi][npd_indices], 0)
        corrs[hemi]['observed_robust_group_natural'] = mycorr(D_observed_group[hemi][:, natural_indices], D_predicted_robust_group[hemi][:, natural_indices], 1)
        corrs[hemi]['observed_standard_group_natural'] = mycorr(D_observed_group[hemi][:, natural_indices], D_predicted_standard_group[hemi][:, natural_indices], 1)
        corrs[hemi]['observed_robust_group_npd'] = mycorr(D_observed_group[hemi][:, npd_indices], D_predicted_robust_group[hemi][:, npd_indices], 1)
        corrs[hemi]['observed_standard_group_npd'] = mycorr(D_observed_group[hemi][:, npd_indices], D_predicted_standard_group[hemi][:, npd_indices], 1)

    #---------------------------------------------------------------------------
    # Load masks for both hemispheres
    #---------------------------------------------------------------------------
    experiment1_mask = {}
    experiment3_mask = {}
    for hemi in ['L', 'R']:
        experiment1_mask[hemi] = np.load(str(args.masks_dir / f"old_subjects_reliability_x_anatomical_fsaverage_{hemi}.npy"))
        experiment1_mask[hemi] = np.where(np.isnan(experiment1_mask[hemi]), False, True)
        experiment3_mask[hemi] = np.load(str(args.masks_dir / f"new_subjects_split_half_reliability_40_{hemi}.npy"))
        experiment3_mask[hemi] = np.where(np.isnan(experiment3_mask[hemi]), False, True)

    boolean_masks = load_dict_h5(str(args.masks_dir / "ROI_masks.h5"))
    for mask in boolean_masks['L']:
        if np.isnan(boolean_masks['L'][mask]).any():
            boolean_masks['L'][mask] = np.where(np.isnan(boolean_masks['L'][mask]), False, True)
        if np.isnan(boolean_masks['R'][mask]).any():
            boolean_masks['R'][mask] = np.where(np.isnan(boolean_masks['R'][mask]), False, True)

    subjects_list = list(data['observed'].keys())
    subject_masks = {}
    for s in subjects_list:
        subject_masks[s] = {}
        for hemi in ['L', 'R']:
            d = data['observed'][s][hemi]
            d0 = d[[0, 2, 4]].mean(0) if d.shape[0] == 6 else d[0]
            d1 = d[[1, 3, 5]].mean(0) if d.shape[0] == 6 else d[1]
            smask = reliability_b2021(d0[60:180], d1[60:180]) > 0.4
            subject_masks[s][hemi] = smask & experiment3_mask[hemi]

    opacity = 1
    ROI_colors = {
        "Medial Heschl's gyrus": "rgb(49, 40, 123)",
        "Lateral Heschl's gyrus": "rgb(92, 132, 193)",
        'Planum polare': "rgb(187, 238, 102)",
        'Planum temporale': "rgb(233, 123, 33)",
        'Superior temporal gyrus': "rgb(134, 30, 26)"
    }

    selected_ROIs = list(ROI_colors.keys())
    excluded_ROIs = []

    #---------------------------------------------------------------------------
    # ROI variance bar plot (combine both hemispheres)
    #---------------------------------------------------------------------------
    y_positions = {roi: i for i, roi in enumerate(selected_ROIs) if roi not in excluded_ROIs}
    roi_idx = 0
    # Use experiment 3 mask to compute global rescale factor
    rescale = max(
        np.quantile(D_observed_group['L'][..., experiment3_mask['L']], 0.999),
        np.quantile(D_observed_group['R'][..., experiment3_mask['R']], 0.999),
    )
    fig = go.Figure()
    for roi in selected_ROIs:
        if roi in excluded_ROIs:
            continue

        # Combine L and R
        natural_variances_list = []
        npd_variances_list = []
        for hemi in ['L', 'R']:
            roi_bool = boolean_masks[hemi][roi].astype(bool)
            nat_var = np.array([
                np.var(D_observed_group[hemi][i, 120:][:, subject_masks[s][hemi] & roi_bool])
                for i, s in enumerate(subjects_list)
            ])
            con_var = np.array([
                np.var(D_observed_group[hemi][i, :60][:, subject_masks[s][hemi] & roi_bool])
                for i, s in enumerate(subjects_list)
            ])
            nat_var /= rescale
            con_var /= rescale
            natural_variances_list.append(nat_var)
            npd_variances_list.append(con_var)

        natural_variances = np.concatenate(natural_variances_list, axis=0)
        npd_variances = np.concatenate(npd_variances_list, axis=0)
        natural_variances_mean = np.nanmean(natural_variances)
        npd_variances_mean = np.nanmean(npd_variances)

        for i in range(len(natural_variances)):
            fig.add_trace(go.Scatter(
                y=[natural_variances[i], npd_variances[i]],
                x=[roi_idx - 0.2, roi_idx + 0.2],
                mode='lines',
                line=dict(color='rgba(1,1,1,1)', width=0.75),
                showlegend=False, hoverinfo='skip'
            ))

        fig.add_trace(go.Scatter(
            y=natural_variances,
            x=[roi_idx - 0.2] * len(natural_variances),
            mode='markers',
            marker=dict(color=ROI_colors[roi], size=6, line=dict(width=0.25, color='black')),
            name=f"{roi}_standard", showlegend=False
        ))
        fig.add_trace(go.Scatter(
            y=npd_variances,
            x=[roi_idx + 0.2] * len(npd_variances),
            mode='markers',
            marker=dict(color=ROI_colors[roi], size=6, line=dict(width=0.25, color='black')),
            name=f"{roi}_robust", showlegend=False
        ))

        x_non = roi_idx - 0.2
        x_rob = roi_idx + 0.2
        bar_thickness = 0.35

        fig.add_trace(go.Bar(
            y=[natural_variances_mean], x=[x_non], orientation='v',
            marker=dict(color=ROI_colors[roi], line=dict(color='black', width=1)),
            width=bar_thickness, showlegend=False, base=0,
        ))
        fig.add_trace(go.Bar(
            y=[npd_variances_mean], x=[x_rob], orientation='v',
            marker=dict(color=ROI_colors[roi], line=dict(color='black', width=1)),
            width=bar_thickness, showlegend=False, base=0, marker_pattern_shape="x"
        ))
        roi_idx += 1

    fig.update_layout(
        width=100 * 2.75, height=100 * 3.25,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(
            zeroline=True, zerolinecolor="black",
            gridcolor="rgba(128,128,128,0.3)", showgrid=True,
            showticklabels=False, nticks=4
        ),
        xaxis=dict(
            showticklabels=False,
            ticktext=[r for r in selected_ROIs if r not in excluded_ROIs],
            tickangle=0
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        font=dict(size=10, family="Times New Roman")
    )
    supplemental_dir = args.output_dir / "supplemental"
    supplemental_dir.mkdir(parents=True, exist_ok=True)
    png_path = supplemental_dir / "roi_response_variances.png"
    fig.write_image(str(png_path), scale=12, height=100 * 2.75, width=100 * 3.25)
    fig.write_html(str(png_path.with_suffix(".html")), include_plotlyjs="directory", include_mathjax="cdn")

    #---------------------------------------------------------------------------
    # ROI median bar plot (combine both hemispheres)
    #---------------------------------------------------------------------------
    fig = go.Figure()
    for roi in selected_ROIs:
        if roi in excluded_ROIs:
            continue

        natural_medians_list = []
        npd_medians_list = []
        for hemi in ['L', 'R']:
            roi_bool = boolean_masks[hemi][roi].astype(bool)
            nat_med = np.array([
                np.median(D_observed_group[hemi][i, 120:][:, subject_masks[s][hemi] & roi_bool])
                for i, s in enumerate(subjects_list)
            ])
            con_med = np.array([
                np.median(D_observed_group[hemi][i, :60][:, subject_masks[s][hemi] & roi_bool])
                for i, s in enumerate(subjects_list)
            ])
            nat_med /= rescale
            con_med /= rescale
            natural_medians_list.append(nat_med)
            npd_medians_list.append(con_med)

        natural_medians = np.concatenate(natural_medians_list, axis=0)
        npd_medians = np.concatenate(npd_medians_list, axis=0)
        natural_medians_mean = np.nanmean(natural_medians)
        npd_medians_mean = np.nanmean(npd_medians)

        for i in range(len(natural_medians)):
            fig.add_trace(go.Scatter(
                y=[natural_medians[i], npd_medians[i]],
                x=[roi_idx - 0.2, roi_idx + 0.2],
                mode='lines',
                line=dict(color='rgba(1,1,1,1)', width=0.75),
                showlegend=False, hoverinfo='skip'
            ))

        fig.add_trace(go.Scatter(
            y=natural_medians,
            x=[roi_idx - 0.2] * len(natural_medians),
            mode='markers',
            marker=dict(color=ROI_colors[roi], size=6, line=dict(width=0.25, color='black')),
            name=f"{roi}_standard", showlegend=False
        ))
        fig.add_trace(go.Scatter(
            y=npd_medians,
            x=[roi_idx + 0.2] * len(npd_medians),
            mode='markers',
            marker=dict(color=ROI_colors[roi], size=6, line=dict(width=0.25, color='black')),
            name=f"{roi}_robust", showlegend=False
        ))

        x_non = roi_idx - 0.2
        x_rob = roi_idx + 0.2
        bar_thickness = 0.35

        fig.add_trace(go.Bar(
            y=[natural_medians_mean], x=[x_non], orientation='v',
            marker=dict(color=ROI_colors[roi], line=dict(color='black', width=1)),
            width=bar_thickness, showlegend=False, base=0,
        ))
        fig.add_trace(go.Bar(
            y=[npd_medians_mean], x=[x_rob], orientation='v',
            marker=dict(color=ROI_colors[roi], line=dict(color='black', width=1)),
            width=bar_thickness, showlegend=False, base=0, marker_pattern_shape="x"
        ))
        roi_idx += 1

    fig.update_layout(
        width=100 * 2.75, height=100 * 3.25,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(
            zerolinecolor="black",
            gridcolor="rgba(128,128,128,0.3)", showgrid=True,
            showticklabels=False, nticks=4
        ),
        xaxis=dict(
            showticklabels=False,
            ticktext=[r for r in selected_ROIs if r not in excluded_ROIs],
            tickangle=0, zeroline=True,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        font=dict(size=10, family="Times New Roman")
    )

    png_path = supplemental_dir / "roi_response_medians.png"
    fig.write_image(str(png_path), scale=12, height=100 * 2.75, width=100 * 3.25)
    fig.write_html(str(png_path.with_suffix(".html")), include_plotlyjs="directory", include_mathjax="cdn")

    #---------------------------------------------------------------------------
    # Per-hemisphere: variance computations and brain maps
    #---------------------------------------------------------------------------
    variance_data = {}  # store pre-masked variance arrays for combined scatter plots
    for hemi in ['L', 'R']:
        exp1_mask = experiment1_mask[hemi]

        # Voxelwise variance arrays (pre-masked by exp1_mask)
        variance_data[hemi] = dict(
            observed_natural_individual=np.var(D_observed_individual[hemi][natural_indices][:, exp1_mask], 0),
            observed_npd_individual=np.var(D_observed_individual[hemi][npd_indices][:, exp1_mask], 0),
            robust_natural_individual=np.var(D_predicted_robust_individual[hemi][natural_indices][:, exp1_mask], 0),
            robust_npd_individual=np.var(D_predicted_robust_individual[hemi][npd_indices][:, exp1_mask], 0),
            standard_natural_individual=np.var(D_predicted_standard_individual[hemi][natural_indices][:, exp1_mask], 0),
            standard_npd_individual=np.var(D_predicted_standard_individual[hemi][npd_indices][:, exp1_mask], 0),
            observed_natural_individual2=np.var(D_observed_individual2[hemi][natural_indices][:, exp1_mask], 0),
            observed_npd_individual2=np.var(D_observed_individual2[hemi][npd_indices][:, exp1_mask], 0),
            robust_natural_individual2=np.var(D_predicted_robust_individual2[hemi][natural_indices][:, exp1_mask], 0),
            robust_npd_individual2=np.var(D_predicted_robust_individual2[hemi][npd_indices][:, exp1_mask], 0),
            standard_natural_individual2=np.var(D_predicted_standard_individual2[hemi][natural_indices][:, exp1_mask], 0),
            standard_npd_individual2=np.var(D_predicted_standard_individual2[hemi][npd_indices][:, exp1_mask], 0),
            observed_natural_group=np.var(D_observed_group[hemi][:, natural_indices][:, :, exp1_mask], 1).mean(0),
            observed_npd_group=np.var(D_observed_group[hemi][:, npd_indices][:, :, exp1_mask], 1).mean(0),
            robust_natural_group=np.var(D_predicted_robust_group[hemi][:, natural_indices][:, :, exp1_mask], 1).mean(0),
            robust_npd_group=np.var(D_predicted_robust_group[hemi][:, npd_indices][:, :, exp1_mask], 1).mean(0),
            standard_natural_group=np.var(D_predicted_standard_group[hemi][:, natural_indices][:, :, exp1_mask], 1).mean(0),
            standard_npd_group=np.var(D_predicted_standard_group[hemi][:, npd_indices][:, :, exp1_mask], 1).mean(0),
        )

        #-----------------------------------------------------------------------
        # Brain maps (per hemisphere)
        #-----------------------------------------------------------------------
        c = corrs[hemi]

        all_data = np.stack([
            c['observed_robust_group_natural'].mean(0),
            c['observed_standard_group_natural'].mean(0),
            c['observed_robust_group_npd'].mean(0),
            c['observed_standard_group_npd'].mean(0),
            c['observed_robust_individual_natural'],
            c['observed_standard_individual_natural'],
            c['observed_robust_individual_npd'],
            c['observed_standard_individual_npd'],
        ])
        q1, q99 = np.quantile(all_data[..., exp1_mask], (0.001, 0.999))

        plotting_mask = np.where(exp1_mask, 1, np.nan)
        plot_brain(c['observed_robust_group_natural'].mean(0), plotting_mask, vmin=-.8, vmax=.8,
            fname=str(args.output_dir / f"brainmap_pred_acc_robust_group_natural_{hemi}.png"), hemi=hemi)
        plot_brain(c['observed_standard_group_natural'].mean(0), plotting_mask, vmin=-.8, vmax=.8,
            fname=str(args.output_dir / f"brainmap_pred_acc_nonrobust_group_natural_{hemi}.png"), hemi=hemi)
        plot_brain(c['observed_robust_group_npd'].mean(0), plotting_mask, vmin=-.8, vmax=.8,
            fname=str(args.output_dir / f"brainmap_pred_acc_robust_group_controversial_{hemi}.png"), hemi=hemi)
        plot_brain(c['observed_standard_group_npd'].mean(0), plotting_mask, vmin=-.8, vmax=.8,
            fname=str(args.output_dir / f"brainmap_pred_acc_nonrobust_group_controversial_{hemi}.png"), hemi=hemi)

        plot_brain(c['observed_robust_individual_natural'], plotting_mask, vmin=-1, vmax=1,
            fname=str(args.output_dir / f"brainmap_pred_acc_robust_individual_natural_{hemi}.png"), hemi=hemi)
        plot_brain(c['observed_standard_individual_natural'], plotting_mask, vmin=-1, vmax=1,
            fname=str(args.output_dir / f"brainmap_pred_acc_nonrobust_individual_natural_{hemi}.png"), hemi=hemi)
        plot_brain(c['observed_robust_individual_npd'], plotting_mask, vmin=-1, vmax=1,
            fname=str(args.output_dir / f"brainmap_pred_acc_robust_individual_controversial_{hemi}.png"), hemi=hemi)
        plot_brain(c['observed_standard_individual_npd'], plotting_mask, vmin=-1, vmax=1,
            fname=str(args.output_dir / f"brainmap_pred_acc_nonrobust_individual_controversial_{hemi}.png"), hemi=hemi)

        plot_brain(c['observed_robust_individual2_natural'], plotting_mask, vmin=-1, vmax=1,
            fname=str(args.output_dir / f"brainmap_pred_acc_robust_individual2_natural_{hemi}.png"), hemi=hemi)
        plot_brain(c['observed_standard_individual2_natural'], plotting_mask, vmin=-1, vmax=1,
            fname=str(args.output_dir / f"brainmap_pred_acc_nonrobust_individual2_natural_{hemi}.png"), hemi=hemi)
        plot_brain(c['observed_robust_individual2_npd'], plotting_mask, vmin=-1, vmax=1,
            fname=str(args.output_dir / f"brainmap_pred_acc_robust_individual2_controversial_{hemi}.png"), hemi=hemi)
        plot_brain(c['observed_standard_individual2_npd'], plotting_mask, vmin=-1, vmax=1,
            fname=str(args.output_dir / f"brainmap_pred_acc_nonrobust_individual2_controversial_{hemi}.png"), hemi=hemi)

        #-----------------------------------------------------------------------
        # CCC statistics (per hemisphere)
        #-----------------------------------------------------------------------
        for pair, title in zip([
            (c['observed_robust_individual_natural'], c['observed_standard_individual_natural']),
            (c['observed_robust_individual2_natural'], c['observed_standard_individual2_natural']),
            (c['observed_robust_group_natural'].mean(0), c['observed_standard_group_natural'].mean(0)),
            (c['observed_robust_individual_npd'], c['observed_standard_individual_npd']),
            (c['observed_robust_individual2_npd'], c['observed_standard_individual2_npd']),
            (c['observed_robust_group_npd'].mean(0), c['observed_standard_group_npd'].mean(0)),
            (c['observed_robust_individual_natural'], c['observed_robust_individual_npd']),
            (c['observed_robust_individual2_natural'], c['observed_robust_individual2_npd']),
            (c['observed_robust_group_natural'].mean(0), c['observed_robust_group_npd'].mean(0)),
            (c['observed_standard_individual_natural'], c['observed_standard_individual_npd']),
            (c['observed_standard_individual2_natural'], c['observed_standard_individual2_npd']),
            (c['observed_standard_group_natural'].mean(0), c['observed_standard_group_npd'].mean(0)),
        ], [
            "Robust and standard natural predictions (individual)",
            "Robust and standard natural predictions (individual2)",
            "Robust and standard natural predictions (group)",
            "Robust and standard npd predictions (individual)",
            "Robust and standard npd predictions (individual2)",
            "Robust and standard npd predictions (group)",
            "Robust and npd natural predictions (individual)",
            "Robust and npd natural predictions (individual2)",
            "Robust and npd natural predictions (group)",
            "Standard and npd natural predictions (individual)",
            "Standard and npd natural predictions (individual2)",
            "Standard and npd natural predictions (group)",
        ]):
            ccc = concordance_correlation_coefficient(pair[0][exp1_mask], pair[1][exp1_mask])
            print(f"[{hemi}] {title}:")
            for k, v in ccc.items():
                print(f"  {k}: {v:.3f}")
            print()

    #---------------------------------------------------------------------------
    # Combined scatter plots (both hemispheres concatenated)
    #---------------------------------------------------------------------------
    exp1_mask_L = experiment1_mask['L']
    exp1_mask_R = experiment1_mask['R']

    fig_dir = supplemental_dir / "observed_variances"
    fig_dir.mkdir(parents=True, exist_ok=True)

    def variance_scatter(x, y, fname=None):
        x_norm = x.copy()
        y_norm = y.copy()
        data_min = y_norm.min()
        data_max = y_norm.max()
        x_norm = (x_norm - data_min) / (data_max - data_min)
        y_norm = (y_norm - data_min) / (data_max - data_min)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[-1, 1], y=[-1, 1], mode='lines',
            line=dict(color="cyan", dash="dash"), showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[-1, 1], y=[-5, 5], mode='lines',
            line=dict(color="red", dash="dash"), showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=x_norm, y=y_norm, mode='markers',
            marker=dict(size=2, opacity=.04, color="black"),
            showlegend=False
        ))
        fig.update_xaxes(
            scaleanchor="y", scaleratio=1, zeroline=True,
            tickvals=[-1, -.5, 0, .5, 1], showticklabels=False,
            gridcolor="lightgrey", zerolinecolor="black"
        )
        fig.update_yaxes(
            scaleanchor="x", scaleratio=1, zeroline=True,
            tickvals=[-.5, 0, .5, 1], showticklabels=False,
            gridcolor="lightgrey", zerolinecolor="black"
        )
        fig.update_layout(
            width=.75*100, height=.75*100,
            xaxis_range=[-.1, 1.1], yaxis_range=[-.1, 1.1],
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=0, b=0),
        )
        if fname is not None:
            fig.write_image(fname, scale=12, width=.75 * 100, height=.75 * 100)

    def _cat(key):
        """Concatenate pre-masked variance arrays from both hemispheres."""
        return np.concatenate([variance_data['L'][key], variance_data['R'][key]])

    # natural on x axis (per rule)
    variance_scatter(_cat('observed_natural_individual'), _cat('observed_npd_individual'),
        fname=str(fig_dir / "variance_individual_observed_natural_vs_controversial.png"))
    variance_scatter(_cat('observed_natural_individual2'), _cat('observed_npd_individual2'),
        fname=str(fig_dir / "variance_individual2_observed_natural_vs_controversial.png"))
    variance_scatter(_cat('observed_natural_group'), _cat('observed_npd_group'),
        fname=str(fig_dir / "variance_group_observed_natural_vs_controversial.png"))
    variance_scatter(_cat('robust_natural_individual'), _cat('robust_npd_individual'),
        fname=str(fig_dir / "variance_individual_robust_natural_vs_controversial.png"))
    variance_scatter(_cat('standard_natural_individual'), _cat('standard_npd_individual'),
        fname=str(fig_dir / "variance_individual_nonrobust_natural_vs_controversial.png"))
    variance_scatter(_cat('robust_natural_individual2'), _cat('robust_npd_individual2'),
        fname=str(fig_dir / "variance_individual2_robust_natural_vs_controversial.png"))
    variance_scatter(_cat('standard_natural_individual2'), _cat('standard_npd_individual2'),
        fname=str(fig_dir / "variance_individual2_nonrobust_natural_vs_controversial.png"))
    variance_scatter(_cat('robust_natural_group'), _cat('robust_npd_group'),
        fname=str(fig_dir / "variance_group_robust_natural_vs_controversial.png"))
    variance_scatter(_cat('standard_natural_group'), _cat('standard_npd_group'),
        fname=str(fig_dir / "variance_group_nonrobust_natural_vs_controversial.png"))

    individual_scatter_xaxis_range = [-.5, 1.1]
    individual_scatter_yaxis_range = [-.5, 1.1]

    def corr_scatter(x, y, fname=None, xaxis_range=[-.2, 1.1], yaxis_range=[-.2, 1.1]):
        """x and y are pre-masked and concatenated across hemispheres."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=y, mode='markers',
            marker=dict(size=2, opacity=.04, color="black"), showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[-1, 1], y=[-1, 1], mode='lines',
            line=dict(color="cyan", dash="dash"), showlegend=False
        ))
        fig.update_xaxes(scaleanchor="y", scaleratio=1, zeroline=True,
            tickvals=[-1, -.5, 0, .5, 1], showticklabels=False,
            gridcolor="lightgrey", zerolinecolor="black")
        fig.update_yaxes(scaleanchor="x", scaleratio=1, zeroline=True,
            tickvals=[-.5, 0, .5, 1], showticklabels=False,
            gridcolor="lightgrey", zerolinecolor="black")
        fig.update_layout(
            width=.75*100, height=.75*100,
            xaxis_range=xaxis_range, yaxis_range=yaxis_range,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=0, b=0),
        )
        if fname is not None:
            fig.write_image(fname, scale=12, height=.75*100, width=.75*100)

    def _ccat(key):
        """Concatenate exp1_mask-filtered corr arrays from both hemispheres."""
        return np.concatenate([corrs['L'][key][exp1_mask_L], corrs['R'][key][exp1_mask_R]])

    def _gccat(key):
        """Same but for group-level corrs (.mean(0) first)."""
        return np.concatenate([corrs['L'][key].mean(0)[exp1_mask_L], corrs['R'][key].mean(0)[exp1_mask_R]])

    # same stimuli different model — standard on x axis (per rule)
    corr_scatter(_ccat('observed_standard_individual_natural'), _ccat('observed_robust_individual_natural'),
        fname=str(args.output_dir / "scatter_pred_acc_nonrobust_robust_individual_natural.png"),
        xaxis_range=individual_scatter_xaxis_range, yaxis_range=individual_scatter_yaxis_range)
    corr_scatter(_ccat('observed_standard_individual2_natural'), _ccat('observed_robust_individual2_natural'),
        fname=str(args.output_dir / "scatter_pred_acc_nonrobust_robust_individual2_natural.png"),
        xaxis_range=individual_scatter_xaxis_range, yaxis_range=individual_scatter_yaxis_range)
    corr_scatter(_gccat('observed_standard_group_natural'), _gccat('observed_robust_group_natural'),
        fname=str(args.output_dir / "scatter_pred_acc_nonrobust_robust_group_natural.png"))
    corr_scatter(_ccat('observed_standard_individual_npd'), _ccat('observed_robust_individual_npd'),
        fname=str(args.output_dir / "scatter_pred_acc_nonrobust_robust_individual_controversial.png"),
        xaxis_range=individual_scatter_xaxis_range, yaxis_range=individual_scatter_yaxis_range)
    corr_scatter(_ccat('observed_standard_individual2_npd'), _ccat('observed_robust_individual2_npd'),
        fname=str(args.output_dir / "scatter_pred_acc_nonrobust_robust_individual2_controversial.png"),
        xaxis_range=individual_scatter_xaxis_range, yaxis_range=individual_scatter_yaxis_range)
    corr_scatter(_gccat('observed_standard_group_npd'), _gccat('observed_robust_group_npd'),
        fname=str(args.output_dir / "scatter_pred_acc_nonrobust_robust_group_controversial.png"))

    # same model different stimuli — natural on x axis (per rule)
    corr_scatter(_ccat('observed_robust_individual_natural'), _ccat('observed_robust_individual_npd'),
        fname=str(args.output_dir / "scatter_robust_pred_accuracy_natural_v_controversial_individual.png"),
        xaxis_range=individual_scatter_xaxis_range, yaxis_range=individual_scatter_yaxis_range)
    corr_scatter(_ccat('observed_robust_individual2_natural'), _ccat('observed_robust_individual2_npd'),
        fname=str(args.output_dir / "scatter_robust_pred_accuracy_natural_v_controversial_individual2.png"),
        xaxis_range=individual_scatter_xaxis_range, yaxis_range=individual_scatter_yaxis_range)
    corr_scatter(_gccat('observed_robust_group_natural'), _gccat('observed_robust_group_npd'),
        fname=str(args.output_dir / "scatter_robust_pred_accuracy_natural_v_controversial_group.png"))
    corr_scatter(_ccat('observed_standard_individual_natural'), _ccat('observed_standard_individual_npd'),
        fname=str(args.output_dir / "scatter_nonrobust_pred_accuracy_natural_v_controversial_individual.png"),
        xaxis_range=individual_scatter_xaxis_range, yaxis_range=individual_scatter_yaxis_range)
    corr_scatter(_ccat('observed_standard_individual2_natural'), _ccat('observed_standard_individual2_npd'),
        fname=str(args.output_dir / "scatter_nonrobust_pred_accuracy_natural_v_controversial_individual2.png"),
        xaxis_range=individual_scatter_xaxis_range, yaxis_range=individual_scatter_yaxis_range)
    corr_scatter(_gccat('observed_standard_group_natural'), _gccat('observed_standard_group_npd'),
        fname=str(args.output_dir / "scatter_nonrobust_pred_accuracy_natural_v_controversial_group.png"))

    #---------------------------------------------------------------------------
    # Large combined scatter plots (both hemispheres, standard on x axis)
    #---------------------------------------------------------------------------
    def corr_scatter_large(x, y, fname=None):
        """x and y are pre-masked and concatenated across hemispheres."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=y, mode='markers',
            marker=dict(size=8, opacity=.2, color="black"), showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[-1, 1], y=[-1, 1], mode='lines',
            line=dict(color="cyan", dash="dash"), showlegend=False
        ))
        fig.update_xaxes(scaleanchor="y", scaleratio=1, zeroline=True,
            tickvals=[-1, -.5, 0, .5, 1], showticklabels=False,
            gridcolor="lightgrey", zerolinecolor="black")
        fig.update_yaxes(scaleanchor="x", scaleratio=1, zeroline=True,
            tickvals=[-.5, 0, .5, 1], showticklabels=False,
            gridcolor="lightgrey", zerolinecolor="black")
        fig.update_layout(
            width=.85*300, height=.85*300,
            xaxis_range=[-.65, 1.1], yaxis_range=[-.65, 1.1],
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=0, b=0),
        )
        if fname is not None:
            fig.write_image(fname, scale=12)

    # standard on x axis (per rule)
    corr_scatter_large(_ccat('observed_standard_individual_natural'), _ccat('observed_robust_individual_natural'),
        fname=str(args.output_dir / "scatter_nonrobust_vs_robust_individual_natural_large.png"))
    corr_scatter_large(_gccat('observed_standard_group_natural'), _gccat('observed_robust_group_natural'),
        fname=str(args.output_dir / "scatter_nonrobust_vs_robust_group_natural_large.png"))
    corr_scatter_large(_ccat('observed_standard_individual_npd'), _ccat('observed_robust_individual_npd'),
        fname=str(args.output_dir / "scatter_nonrobust_vs_robust_individual_controversial_large.png"))
    corr_scatter_large(_gccat('observed_standard_group_npd'), _gccat('observed_robust_group_npd'),
        fname=str(args.output_dir / "scatter_nonrobust_vs_robust_group_controversial_large.png"))

    #---------------------------------------------------------------------------
    # ROI summary: prediction accuracy npd (combine both hemispheres)
    #---------------------------------------------------------------------------
    ROI_colors = {
        "Medial Heschl's gyrus": "rgb(49, 40, 123)",
        "Lateral Heschl's gyrus": "rgb(92, 132, 193)",
        'Planum polare': "rgb(187, 238, 102)",
        'Planum temporale': "rgb(233, 123, 33)",
        'Superior temporal gyrus': "rgb(134, 30, 26)"
    }

    selected_ROIs = list(ROI_colors.keys())
    excluded_ROIs = []
    fig = go.Figure()

    y_positions = {roi: i for i, roi in enumerate(selected_ROIs) if roi not in excluded_ROIs}
    roi_idx = 0
    t_stats = []
    p_values = []
    for roi in selected_ROIs:
        if roi in excluded_ROIs:
            continue

        robust_data_list = []
        standard_data_list = []
        for hemi in ['L', 'R']:
            roi_bool = boolean_masks[hemi][roi].astype(bool)
            robust_data_list.append(np.array([
                np.median(corrs[hemi]['observed_robust_group_npd'][i, subject_masks[s][hemi] & roi_bool])
                for i, s in enumerate(subjects_list)
            ]))
            standard_data_list.append(np.array([
                np.median(corrs[hemi]['observed_standard_group_npd'][i, subject_masks[s][hemi] & roi_bool])
                for i, s in enumerate(subjects_list)
            ]))

        robust_data = np.nanmean(np.stack(robust_data_list, axis=0), axis=0)
        standard_data = np.nanmean(np.stack(standard_data_list, axis=0), axis=0)

        robust_mean = np.nanmean(robust_data)
        standard_mean = np.nanmean(standard_data)

        for i in range(len(robust_data)):
            fig.add_trace(go.Scatter(
                x=[standard_data[i], robust_data[i]],
                y=[roi_idx + 0.2, roi_idx - 0.2],
                mode='lines', line=dict(color='rgba(1,1,1,1)', width=0.75),
                showlegend=False, hoverinfo='skip'
            ))

        fig.add_trace(go.Scatter(
            x=standard_data, y=[roi_idx + 0.2] * len(standard_data),
            mode='markers',
            marker=dict(color=ROI_colors[roi], size=6, line=dict(width=0.25, color='black')),
            name=f"{roi}_standard", showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=robust_data, y=[roi_idx - 0.2] * len(robust_data),
            mode='markers',
            marker=dict(color=ROI_colors[roi], size=6, line=dict(width=0.25, color='black')),
            name=f"{roi}_robust", showlegend=False
        ))

        t_stat, p_value = scipy.stats.ttest_rel(standard_data, robust_data, nan_policy='omit')
        t_stats.append(t_stat)
        p_values.append(p_value)

        y_non = roi_idx + 0.2
        y_rob = roi_idx - 0.2
        bar_thickness = 0.35

        fig.add_trace(go.Bar(
            x=[standard_mean], y=[y_non], orientation='h',
            marker=dict(color=ROI_colors[roi], line=dict(color='black', width=1)),
            width=bar_thickness, showlegend=False, base=0, marker_pattern_shape="x"
        ))
        fig.add_trace(go.Bar(
            x=[robust_mean], y=[y_rob], orientation='h',
            marker=dict(color=ROI_colors[roi], line=dict(color='black', width=1)),
            width=bar_thickness, showlegend=False, base=0,
        ))

        roi_idx += 1

    fig.update_layout(
        width=100 * 2.75, height=100 * 3.25,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            zeroline=True, zerolinecolor="black",
            gridcolor="rgba(128,128,128,0.3)", showgrid=True,
            tickvals=[0, 0.25, 0.5, 0.75], showticklabels=False,
            range=[-.2, .9],
        ),
        yaxis=dict(
            showticklabels=False,
            tickvals=list(range(len([r for r in selected_ROIs if r not in excluded_ROIs]))),
            ticktext=[r for r in selected_ROIs if r not in excluded_ROIs],
            tickangle=0
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        font=dict(size=10, family="Times New Roman")
    )
    fig.write_image(str(args.output_dir / "roi_summary_pred_acc_nonrobust_robust_controversial.png"), scale=12, width=100 * 2.75, height=100 * 3.25)
    p_arr = np.array(p_values, dtype=float)
    valid = ~np.isnan(p_arr)
    corrected = np.full(len(p_arr), np.nan)
    if valid.any():
        corrected[valid] = scipy.stats.false_discovery_control(p_arr[valid], method="by")
    for roi, t_stat, p_value, corrected_p_value in zip(selected_ROIs, t_stats, p_values, corrected):
        print(f"Subject-level paired t-test ({roi}): t={t_stat:.3f}, p={p_value:.3f}, corrected p={corrected_p_value:.3f}\n")

    #---------------------------------------------------------------------------
    # ROI map visualization (both hemispheres)
    #---------------------------------------------------------------------------
    from nilearn import datasets, plotting, surface
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm

    ROI_colors = {
        "Medial Heschl's gyrus": "rgb(49, 40, 123)",
        "Lateral Heschl's gyrus": "rgb(92, 132, 193)",
        'Planum polare': "rgb(187, 238, 102)",
        'Planum temporale': "rgb(233, 123, 33)",
        'Superior temporal gyrus': "rgb(134, 30, 26)"
    }

    def rgba_string_to_mpl(rgba_string):
        values = rgba_string.strip('rgba()').split(',')
        r, g, b = [float(v) for v in values]
        return (r/255.0, g/255.0, b/255.0)

    colors = [rgba_string_to_mpl(ROI_colors[roi]) for roi in ROI_colors]
    cmap = ListedColormap(colors)
    bounds = [0, 1, 2, 3, 4, 5]
    norm = BoundaryNorm(bounds, cmap.N)

    value_mapping = {roi: i for (i, roi) in enumerate(ROI_colors)}

    print("Value to ROI mapping:")
    for value, roi_name in value_mapping.items():
        print(f"  {value}: {roi_name}")

    fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage')
    inflated_L = fsaverage.infl_left
    inflated_R = fsaverage.infl_right
    curv_L = nilearn.surface.load_surf_data(fsaverage.curv_left)
    curv_R = nilearn.surface.load_surf_data(fsaverage.curv_right)
    bg_L = np.array([0.9 if c_ > 0 else 0.5 for c_ in curv_L])
    bg_R = np.array([0.9 if c_ > 0 else 0.5 for c_ in curv_R])

    for hemi in ['L', 'R']:
        colors_roi_map = np.full(163842, np.nan)
        for roi in value_mapping:
            mask_x_reliable = np.logical_and(boolean_masks[hemi]['subject_reliability_NH2015_30'], boolean_masks[hemi][roi])
            colors_roi_map[mask_x_reliable] = value_mapping[roi]

        inflated = inflated_L if hemi == 'L' else inflated_R
        bg = bg_L if hemi == 'L' else bg_R
        hemi_str = "left" if hemi == 'L' else "right"

        fig = plotting.plot_surf_stat_map(
            inflated, colors_roi_map, bg_map=bg,
            engine="plotly", cmap=cmap, colorbar=False, hemi=hemi_str,
        )
        png_path = args.output_dir / f"ROI_map_{hemi}.png"
        fig.figure.write_image(str(png_path), scale=4)

    #---------------------------------------------------------------------------
    # Box plots / violins (combine both hemispheres)
    #---------------------------------------------------------------------------
    npd_diff = {}
    for hemi in ['L', 'R']:
        npd_diff[hemi] = corrs[hemi]['observed_robust_group_npd'] - corrs[hemi]['observed_standard_group_npd']

    excluded_ROIs = []
    selected_ROIs = list(ROI_colors.keys())

    fig = go.Figure()
    for roi in selected_ROIs:
        if roi in excluded_ROIs:
            print(f"Excluding {roi}")
            continue
        # Combine L and R voxels
        robust_vals = np.concatenate([
            corrs['L']['observed_robust_group_npd'][:, boolean_masks['L'][roi]].mean(0),
            corrs['R']['observed_robust_group_npd'][:, boolean_masks['R'][roi]].mean(0),
        ])
        standard_vals = np.concatenate([
            corrs['L']['observed_standard_group_npd'][:, boolean_masks['L'][roi]].mean(0),
            corrs['R']['observed_standard_group_npd'][:, boolean_masks['R'][roi]].mean(0),
        ])

        fig.add_trace(go.Box(
            y0=roi, x=robust_vals,
            name=f"{roi}_standard", fillcolor=ROI_colors[roi],
            line=dict(color="black"), showlegend=False,
            offsetgroup=-.1, legendgroup=roi
        ))
        fig.add_trace(go.Box(
            y0=roi, x=standard_vals,
            name=f"{roi}_standard", fillcolor=ROI_colors[roi],
            line=dict(color="black"), showlegend=False,
            offsetgroup=.1, legendgroup=roi
        ))

    fig.update_layout(
        width=300 * 2, height=300 * 1.5,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        boxmode='group',
        xaxis=dict(
            zeroline=True, zerolinecolor="black",
            gridcolor="lightgrey", tickvals=[0, 0.25, 0.5, 0.75],
            showticklabels=False, range=[-.2, .9],
        ),
        yaxis=dict(showticklabels=False, tickangle=45),
        margin=dict(l=0, r=0, t=0, b=0),
        font=dict(size=10, family="Times New Roman")
    )
    fig.update_layout(yaxis=dict(showticklabels=True, tickangle=45))
    fig.write_html(str(args.output_dir / "roi_boxplots_robust_vs_nonrobust_controversial.html"), include_plotlyjs="directory", include_mathjax="cdn")

    fig = go.Figure()
    for roi in boolean_masks['L'].keys():
        diff_vals = np.concatenate([
            npd_diff['L'][:, boolean_masks['L'][roi]].mean(0),
            npd_diff['R'][:, boolean_masks['R'][roi]].mean(0),
        ])
        fig.add_trace(go.Violin(y=diff_vals, name=roi, spanmode='hard'))
    fig.write_html(str(args.output_dir / "roi_violin_controversial_diff_all_rois.html"), include_plotlyjs="directory", include_mathjax="cdn")

    fig = go.Figure()
    excluded_ROIs = ["Anterior non-primary", "subject_split_half_reliability_50"]
    for i, roi in enumerate(boolean_masks['L'].keys()):
        if roi in excluded_ROIs:
            continue
        diff_vals = np.concatenate([
            npd_diff['L'][:, boolean_masks['L'][roi]].mean(0),
            npd_diff['R'][:, boolean_masks['R'][roi]].mean(0),
        ])
        fig.add_trace(go.Box(y=diff_vals, name=roi))
    fig.update_layout(boxgap=.1, width=300 * 2.5, height=300 * 1)
    fig.write_html(str(args.output_dir / "roi_boxplots_controversial_diff_selected_rois.html"), include_plotlyjs="directory", include_mathjax="cdn")


if __name__ == '__main__':
    main()
