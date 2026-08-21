#!/usr/bin/env python3
"""
Create Figure2

Converted from create_figure2.ipynb
"""

import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('../data'),
                      help='Base data directory')
    parser.add_argument('--masks-dir', type=Path, default=Path('../masks'),
                      help='Base masks directory')
    parser.add_argument('--output-dir', type=Path, default=Path('output_create_figure2'),
                      help='Output directory for figures')
    return parser.parse_args()

def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    import sys
    sys.path.append('../scripts')

    import numpy as np
    from lib.utils import mycorr, concordance_correlation_coefficient, degenerate_entries
    from lib.data import load_original_subjects_data, load_dict_h5
    from lib.utils.visualization import stat2grid, plot_brain
    import plotly.graph_objects as go
    import nilearn
    from nilearn import datasets, plotting

    dot_size = 2
    dot_opacity = .04

    data = load_original_subjects_data(str(args.data_dir / 'original-subjects/observed_and_predicted_data_original_subjects_hemi.npz'))

    natsounds = load_dict_h5(str(args.data_dir / 'original-subjects/natsounds.h5'))

    def slice_and_dice(data, kind='observed', test_individual='S30'):
        test_subjects = [f"S{i}" for i in range(11, 31)]
        npd_indices = np.arange(60)
        natural_indices = np.arange(60, 115)

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

        D_individual_natural_L = D_individual_L[natural_indices]
        D_individual_npd_L = D_individual_L[npd_indices]
        D_individual_natural_R = D_individual_R[natural_indices]
        D_individual_npd_R = D_individual_R[npd_indices]

        D_group_natural_L = D_group_L[:, natural_indices]
        D_group_npd_L = D_group_L[:, npd_indices]
        D_group_natural_R = D_group_R[:, natural_indices]
        D_group_npd_R = D_group_R[:, npd_indices]

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

    nan_mask_L = np.isnan(D_group_natural_standard1_L[:, -55:]).sum((0, 1)) == 0
    nan_mask_R = np.isnan(D_group_natural_standard1_R[:, -55:]).sum((0, 1)) == 0

    #---------------------------------------------------------------------------
    # Correlations between predictions (per hemisphere) — compute + brain maps
    #---------------------------------------------------------------------------
    stored = {}
    for hemi in ['L', 'R']:
        D_individual_natural_robust1 = locals()[f'D_individual_natural_robust1_{hemi}']
        D_individual_natural_standard1 = locals()[f'D_individual_natural_standard1_{hemi}']
        D_group_natural_robust1 = locals()[f'D_group_natural_robust1_{hemi}']
        D_group_natural_standard1 = locals()[f'D_group_natural_standard1_{hemi}']
        D_individual_npd_robust1 = locals()[f'D_individual_npd_robust1_{hemi}']
        D_individual_npd_standard1 = locals()[f'D_individual_npd_standard1_{hemi}']
        D_group_npd_robust1 = locals()[f'D_group_npd_robust1_{hemi}']
        D_group_npd_standard1 = locals()[f'D_group_npd_standard1_{hemi}']

        corrs_between_natural_predictions_individual = mycorr(D_individual_natural_robust1, D_individual_natural_standard1, axis=0)
        corrs_between_natural_predictions_group = mycorr(D_group_natural_robust1, D_group_natural_standard1, axis=1)
        corrs_between_npd_predictions_individual = mycorr(D_individual_npd_robust1, D_individual_npd_standard1, axis=0)
        corrs_between_npd_predictions_group = mycorr(D_group_npd_robust1, D_group_npd_standard1, axis=1)

        # Mask degenerate (zero-variance / NaN-input) entries with NaN, per statistic
        # (natural vs npd) and per level (individual vs group -- group is masked
        # per-subject so one degenerate subject doesn't blank out a whole voxel's
        # group average; nanmean below then averages over the remaining subjects).
        corrs_between_natural_predictions_individual = np.where(
            degenerate_entries(D_individual_natural_robust1, D_individual_natural_standard1),
            np.nan, corrs_between_natural_predictions_individual,
        )
        corrs_between_npd_predictions_individual = np.where(
            degenerate_entries(D_individual_npd_robust1, D_individual_npd_standard1),
            np.nan, corrs_between_npd_predictions_individual,
        )
        corrs_between_natural_predictions_group = np.where(
            degenerate_entries(D_group_natural_robust1, D_group_natural_standard1, axis=1),
            np.nan, corrs_between_natural_predictions_group,
        )
        corrs_between_npd_predictions_group = np.where(
            degenerate_entries(D_group_npd_robust1, D_group_npd_standard1, axis=1),
            np.nan, corrs_between_npd_predictions_group,
        )

        individual_valid_mask = (
            np.isfinite(corrs_between_natural_predictions_individual)
            & np.isfinite(corrs_between_npd_predictions_individual)
        )

        corrs_between_natural_predictions_individual_fsaverage = stat2grid(
            corrs_between_natural_predictions_individual, natsounds, hemi=hemi,
        )
        corrs_between_natural_predictions_group_fsaverage = stat2grid(
            np.nanmean(corrs_between_natural_predictions_group, axis=0), natsounds, hemi=hemi,
        )
        corrs_between_npd_predictions_individual_fsaverage = stat2grid(
            corrs_between_npd_predictions_individual, natsounds, hemi=hemi,
        )
        corrs_between_npd_predictions_group_fsaverage = stat2grid(
            np.nanmean(corrs_between_npd_predictions_group, axis=0), natsounds, hemi=hemi,
        )

        corrs_between_natural_predictions_all_fsaverage = np.array(
            [stat2grid(corrs_between_natural_predictions_group[i], natsounds, hemi=hemi) for i in range(20)]
        )
        corrs_between_npd_predictions_all_fsaverage = np.array(
            [stat2grid(corrs_between_npd_predictions_group[i], natsounds, hemi=hemi) for i in range(20)]
        )

        print(f"[{hemi}] Median correlation between natural predictions (group):")
        print(f"  {np.nanmedian(np.nanmean(corrs_between_natural_predictions_all_fsaverage, axis=0)):.3f}")
        print(f"[{hemi}] Median correlation between npd predictions (group):")
        print(f"  {np.nanmedian(np.nanmean(corrs_between_npd_predictions_all_fsaverage, axis=0)):.3f}")

        #-----------------------------------------------------------------------
        # Brain maps (per hemisphere)
        #-----------------------------------------------------------------------
        plot_brain(
            corrs_between_natural_predictions_individual_fsaverage,
            vmin=-1, vmax=1,
            fname=str(args.output_dir / f"natural_individual_brainmap_{hemi}_S30.png"),
            cmap="PiYG", hemi=hemi,
        )
        plot_brain(
            corrs_between_npd_predictions_individual_fsaverage,
            vmin=-1, vmax=1,
            fname=str(args.output_dir / f"controversial_individual_brainmap_{hemi}_S30.png"),
            cmap="PiYG", hemi=hemi,
        )
        plot_brain(
            corrs_between_natural_predictions_group_fsaverage,
            vmin=-1, vmax=1,
            fname=str(args.output_dir / f"natural_group_brainmap_{hemi}.png"),
            cmap="PiYG", hemi=hemi,
        )
        plot_brain(
            corrs_between_npd_predictions_group_fsaverage,
            vmin=-1, vmax=1,
            fname=str(args.output_dir / f"controversial_group_brainmap_{hemi}.png"),
            cmap="PiYG", hemi=hemi,
        )

        #-----------------------------------------------------------------------
        # CCC statistics (per hemisphere)
        #-----------------------------------------------------------------------
        ccc_corrs_natural_vs_npd_group = concordance_correlation_coefficient(
            np.nanmean(corrs_between_natural_predictions_group, axis=0),
            np.nanmean(corrs_between_npd_predictions_group, axis=0),
        )
        print(f"[{hemi}] CCC correlation between natural and npd predictions (group):")
        for k, v in ccc_corrs_natural_vs_npd_group.items():
            print(f"  {k}: {v:.3f}")

        ccc_corrs_natural_vs_npd_individual = concordance_correlation_coefficient(
            corrs_between_natural_predictions_individual[individual_valid_mask],
            corrs_between_npd_predictions_individual[individual_valid_mask],
        )
        print(f"[{hemi}] CCC correlation between natural and npd predictions (individual):")
        for k, v in ccc_corrs_natural_vs_npd_individual.items():
            print(f"  {k}: {v:.3f}")

        # Store computed correlation arrays for combined scatter plots
        nan_mask = locals()[f'nan_mask_{hemi}']
        stored[hemi] = dict(
            nat_individual=corrs_between_natural_predictions_individual,
            nat_group=np.nanmean(corrs_between_natural_predictions_group, axis=0),
            con_individual=corrs_between_npd_predictions_individual,
            con_group=np.nanmean(corrs_between_npd_predictions_group, axis=0),
            nan_mask=nan_mask,
            individual_valid_mask=individual_valid_mask,
            robust_natural_individual_vars=np.var(D_individual_natural_robust1, 0),
            robust_npd_individual_vars=np.var(D_individual_npd_robust1, 0),
            standard_natural_individual_vars=np.var(D_individual_natural_standard1, 0),
            standard_npd_individual_vars=np.var(D_individual_npd_standard1, 0),
            robust_natural_group_vars=np.var(D_group_natural_robust1, 1).mean(0),
            robust_npd_group_vars=np.var(D_group_npd_robust1, 1).mean(0),
            standard_natural_group_vars=np.var(D_group_natural_standard1, 1).mean(0),
            standard_npd_group_vars=np.var(D_group_npd_standard1, 1).mean(0),
        )

    #---------------------------------------------------------------------------
    # Combined scatter plots (both hemispheres)
    #---------------------------------------------------------------------------
    def corr_scatter(x, y, fname=None):
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=[-1, 1], y=[-1, 1], mode='lines',
                line=dict(color="cyan", dash="dash"), showlegend=False
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x, y=y, mode='markers',
                marker=dict(size=dot_size, opacity=dot_opacity, color="black"),
                showlegend=False
            )
        )
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
            xaxis_range=[-.65, 1.1], yaxis_range=[-.65, 1.1],
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=0, b=0),
        )
        if fname is not None:
            fig.write_image(fname, scale=12, height=.75*100, width=.75*100)

    # natural on x axis (per rule), npd on y axis
    individual_valid = np.concatenate([stored['L']['individual_valid_mask'], stored['R']['individual_valid_mask']])
    corr_scatter(
        np.concatenate([stored['L']['nat_individual'], stored['R']['nat_individual']])[individual_valid],
        np.concatenate([stored['L']['con_individual'], stored['R']['con_individual']])[individual_valid],
        fname=str(args.output_dir / "natural_corrs_vs_controversial_corrs_individual_corr_S30.png"),
    )
    nat_group_all = np.concatenate([stored['L']['nat_group'], stored['R']['nat_group']])
    con_group_all = np.concatenate([stored['L']['con_group'], stored['R']['con_group']])
    group_valid = np.isfinite(nat_group_all) & np.isfinite(con_group_all)
    corr_scatter(
        nat_group_all[group_valid],
        con_group_all[group_valid],
        fname=str(args.output_dir / "natural_corrs_vs_controversial_corrs_group_corr.png"),
    )

    #---------------------------------------------------------------------------
    # Combined variance scatters (both hemispheres)
    #---------------------------------------------------------------------------
    supplemental_dir = args.output_dir / "supplemental"
    supplemental_dir.mkdir(parents=True, exist_ok=True)

    def variance_scatter(x, y, fname=None):
        x_norm = x.copy()
        y_norm = y.copy()
        data_min = y_norm.min()
        data_max = y_norm.max()
        x_norm = (x_norm - data_min) / (data_max - data_min)
        y_norm = (y_norm - data_min) / (data_max - data_min)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=[-1, 1], y=[-1, 1], mode='lines',
                line=dict(color="cyan", dash="dash"), showlegend=False
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[-1, 1], y=[-5, 5], mode='lines',
                line=dict(color="red", dash="dash"), showlegend=False
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_norm, y=y_norm, mode='markers',
                marker=dict(size=dot_size, opacity=dot_opacity, color="black"),
                showlegend=False
            )
        )
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
            width=.85*300, height=.85*300,
            xaxis_range=[-.1, 1.1], yaxis_range=[-.1, 1.1],
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=0, b=0),
        )
        if fname is not None:
            fig.write_image(fname, scale=12, width=.75*100, height=.75*100)

    # natural on x axis (per rule)
    variance_scatter(
        np.concatenate([stored['L']['robust_natural_group_vars'][nan_mask_L], stored['R']['robust_natural_group_vars'][nan_mask_R]]),
        np.concatenate([stored['L']['robust_npd_group_vars'][nan_mask_L], stored['R']['robust_npd_group_vars'][nan_mask_R]]),
        fname=str(supplemental_dir / "variance_group_robust_natural_vs_controversial.png"),
    )
    variance_scatter(
        np.concatenate([stored['L']['standard_natural_group_vars'][nan_mask_L], stored['R']['standard_natural_group_vars'][nan_mask_R]]),
        np.concatenate([stored['L']['standard_npd_group_vars'][nan_mask_L], stored['R']['standard_npd_group_vars'][nan_mask_R]]),
        fname=str(supplemental_dir / "variance_group_nonrobust_natural_vs_controversial.png"),
    )
    variance_scatter(
        np.concatenate([stored['L']['robust_natural_individual_vars'][nan_mask_L], stored['R']['robust_natural_individual_vars'][nan_mask_R]]),
        np.concatenate([stored['L']['robust_npd_individual_vars'][nan_mask_L], stored['R']['robust_npd_individual_vars'][nan_mask_R]]),
        fname=str(supplemental_dir / "variance_individual_robust_natural_vs_controversial_S30.png"),
    )
    variance_scatter(
        np.concatenate([stored['L']['standard_natural_individual_vars'][nan_mask_L], stored['R']['standard_natural_individual_vars'][nan_mask_R]]),
        np.concatenate([stored['L']['standard_npd_individual_vars'][nan_mask_L], stored['R']['standard_npd_individual_vars'][nan_mask_R]]),
        fname=str(supplemental_dir / "variance_individual_nonrobust_natural_vs_controversial_S30.png"),
    )

    # --- Individual subject S29 ---
    _s29_r1 = slice_and_dice(data, 'robust1', test_individual='S29')
    _s29_nr1 = slice_and_dice(data, 'standard1', test_individual='S29')
    stored_S29 = {}
    for hemi in ['L', 'R']:
        D_nat_r1  = _s29_r1[0 if hemi == 'L' else 1]
        D_nat_nr1 = _s29_nr1[0 if hemi == 'L' else 1]
        D_cont_r1  = _s29_r1[2 if hemi == 'L' else 3]
        D_cont_nr1 = _s29_nr1[2 if hemi == 'L' else 3]
        nat_corrs  = mycorr(D_nat_r1,  D_nat_nr1, axis=0)
        cont_corrs = mycorr(D_cont_r1, D_cont_nr1, axis=0)
        nat_corrs  = np.where(degenerate_entries(D_nat_r1, D_nat_nr1), np.nan, nat_corrs)
        cont_corrs = np.where(degenerate_entries(D_cont_r1, D_cont_nr1), np.nan, cont_corrs)
        s29_valid_mask = np.isfinite(nat_corrs) & np.isfinite(cont_corrs)
        plot_brain(
            stat2grid(nat_corrs,  natsounds, hemi=hemi),
            vmin=-1, vmax=1,
            fname=str(args.output_dir / f"natural_individual_brainmap_{hemi}_S29.png"),
            cmap="PiYG", hemi=hemi,
        )
        plot_brain(
            stat2grid(cont_corrs, natsounds, hemi=hemi),
            vmin=-1, vmax=1,
            fname=str(args.output_dir / f"controversial_individual_brainmap_{hemi}_S29.png"),
            cmap="PiYG", hemi=hemi,
        )
        stored_S29[hemi] = dict(
            nat_individual=nat_corrs,
            con_individual=cont_corrs,
            valid_mask=s29_valid_mask,
            robust_natural_individual_vars=np.var(D_nat_r1,  0),
            robust_npd_individual_vars=np.var(D_cont_r1,  0),
            standard_natural_individual_vars=np.var(D_nat_nr1, 0),
            standard_npd_individual_vars=np.var(D_cont_nr1, 0),
        )
    s29_valid = np.concatenate([stored_S29['L']['valid_mask'], stored_S29['R']['valid_mask']])
    corr_scatter(
        np.concatenate([stored_S29['L']['nat_individual'], stored_S29['R']['nat_individual']])[s29_valid],
        np.concatenate([stored_S29['L']['con_individual'], stored_S29['R']['con_individual']])[s29_valid],
        fname=str(args.output_dir / "natural_corrs_vs_controversial_corrs_individual_corr_S29.png"),
    )
    variance_scatter(
        np.concatenate([stored_S29['L']['robust_natural_individual_vars'][nan_mask_L],
                        stored_S29['R']['robust_natural_individual_vars'][nan_mask_R]]),
        np.concatenate([stored_S29['L']['robust_npd_individual_vars'][nan_mask_L],
                        stored_S29['R']['robust_npd_individual_vars'][nan_mask_R]]),
        fname=str(supplemental_dir / "variance_individual_robust_natural_vs_controversial_S29.png"),
    )
    variance_scatter(
        np.concatenate([stored_S29['L']['standard_natural_individual_vars'][nan_mask_L],
                        stored_S29['R']['standard_natural_individual_vars'][nan_mask_R]]),
        np.concatenate([stored_S29['L']['standard_npd_individual_vars'][nan_mask_L],
                        stored_S29['R']['standard_npd_individual_vars'][nan_mask_R]]),
        fname=str(supplemental_dir / "variance_individual_nonrobust_natural_vs_controversial_S29.png"),
    )


if __name__ == '__main__':
    main()

