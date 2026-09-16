#!/usr/bin/env python3
"""
RSA Components

Fits an SVD-based component subspace on natural sounds, projects measured and
predicted responses (robust / standard) into that space, and visualises the
results as UMAP embeddings and distance summary plots.

Outputs (annotated = with legend/title, clean = no text or legend):
  umap_observed_{annotated,clean}.{png,svg}
  umap_npd_comparison_{annotated,clean}.{png,svg}
  umap_nat_comparison_{annotated,clean}.{png,svg}
  distances_{annotated,clean}.{png,svg}
  *.html  (interactive, always annotated)
"""

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path,
                        default=Path('../data/new-subjects-clean'))
    parser.add_argument('--masks-dir', type=Path,
                        default=Path('../masks'))
    parser.add_argument('--components-dir', type=Path,
                        default=Path('../components'))
    parser.add_argument('--output-dir', type=Path,
                        default=Path('output_create_figure4_consolidated'))
    parser.add_argument('--n-components', type=int, default=6,
                        help='Number of SVD components for subspace')
    parser.add_argument('--log-file', type=Path, default=None,
                        help='Path to stats log file (default: <output-dir>/stats.log)')
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import sys

    class _Tee:
        def __init__(self, *streams): self._s = streams
        def write(self, d): [s.write(d) for s in self._s]
        def flush(self): [s.flush() for s in self._s]

    _log_path = args.log_file if args.log_file is not None else args.output_dir / 'stats.log'
    _log_fh = open(_log_path, 'w')
    sys.stdout = _Tee(sys.__stdout__, _log_fh)
    import numpy as np
    import plotly.graph_objects as go
    import plotly.io as pio
    from scipy.spatial.distance import pdist, squareform
    from scipy.stats import ttest_ind, ttest_rel
    from umap import UMAP

    sys.path.insert(0, str(Path(__file__).parent / '../scripts'))
    from lib.data import load_data
    from lib.utils import color_lookup, mycorr
    from lib.analysis.reliability import reliability_b2021

    ROBUST_COLOR   = '#E66100'
    STANDARD_COLOR = '#5D3A9B'
    N_COMPS        = args.n_components
    DPI            = 300
    SCALE          = 12

    # ── Load data ────────────────────────────────────────────────────────────────

    def prepare_data(data_type='observed', masks=None):
        masks_out = [] if masks is None else masks
        D_all = []
        for i, subject in enumerate(subjects):
            if data_type == 'observed':
                hemi_data = {}
                for hemi in ['L', 'R']:
                    d = data['observed'][subject][hemi]
                    if d.shape[0] == 6:
                        d0 = d[[0, 2, 4]].mean(0)
                        d1 = d[[1, 3, 5]].mean(0)
                    else:
                        d0, d1 = d[0], d[1]
                    if masks is None:
                        smask = reliability_b2021(d0[60:180], d1[60:180]) > 0.4
                        smask = smask & anatomical_mask[hemi]
                        if i >= len(masks_out):
                            masks_out.append({})
                        masks_out[i][hemi] = smask
                    else:
                        smask = masks[i][hemi]
                    hemi_data[hemi] = np.stack([d0, d1])[..., smask]
                D_all.append(np.concatenate([hemi_data['L'], hemi_data['R']], axis=-1))
            else:
                assert masks is not None
                hemi_preds = []
                for hemi in ['L', 'R']:
                    even = data['predictions'][subject][f'{data_type}_even'][hemi]
                    odd  = data['predictions'][subject][f'{data_type}_odd'][hemi]
                    hemi_preds.append(np.stack([even, odd], 0)[..., masks[i][hemi]])
                D_all.append(np.concatenate(hemi_preds, axis=-1))
        return [d.astype(np.float64) for d in D_all], masks_out

    data = load_data(
        str(args.data_dir / 'observed_and_predicted_data_new_subjects_smoothed_stratified.npz'),
        components=str(args.components_dir / 'components_reordered_stratified.h5'),
        include_prediction_reps=True,
    )
    subjects   = list(data['observed'].keys())
    components = data['components']
    print(f'{len(subjects)} subjects loaded')

    anatomical_mask = {}
    for hemi in ['L', 'R']:
        raw = np.load(str(args.masks_dir /
                          f'old_subjects_reliability_x_anatomical_fsaverage_{hemi}.npy'))
        anatomical_mask[hemi] = ~np.isnan(raw)

    stim_names    = np.array(list(components['stim_names']))
    cat_labels    = np.array(components['category_labels'])
    valid_indices = np.concatenate([np.arange(60), np.arange(120, 180)])

    # ── Prepare responses ────────────────────────────────────────────────────────

    D_all_raw,          subject_masks = prepare_data()
    D_all_robust_raw,   _             = prepare_data('robust',    masks=subject_masks)
    D_all_standard_raw, _             = prepare_data('standard', masks=subject_masks)

    D_meas = [d.mean(0) for d in D_all_raw]
    D_rob  = [d.mean(0) for d in D_all_robust_raw]
    D_std  = [d.mean(0) for d in D_all_standard_raw]

    # ── Center across subjects ───────────────────────────────────────────────────

    def center(arrays, axis=-1):
        concat      = np.concatenate(arrays, axis=axis)
        concat_mean = concat.mean(axis=axis, keepdims=True)
        return [a.copy() - a.mean(axis, keepdims=True) + concat_mean for a in arrays]

    D_meas_centered      = center(D_meas)
    D_rob_centered       = center(D_rob)
    D_std_centered       = center(D_std)
    D_meas_centered_reps = center(D_all_raw, axis=-1)

    D_meas_concat               = np.concatenate(D_meas_centered, axis=-1)
    D_rob_concat                = np.concatenate(D_rob_centered,  axis=-1)
    D_std_concat                = np.concatenate(D_std_centered,  axis=-1)
    D_meas_centered_reps_concat = np.concatenate(D_meas_centered_reps, axis=-1)

    # ── SVD subspace (fit on natural sounds only) ────────────────────────────────

    _, _, v = np.linalg.svd(D_meas_concat[60:120], full_matrices=False)
    W_meas = v[:N_COMPS, :]   # (N_COMPS, n_voxels)

    R_meas = D_meas_centered_reps_concat[0] @ np.linalg.pinv(W_meas)
    R_rob  = D_rob_concat                   @ np.linalg.pinv(W_meas)
    R_std  = D_std_concat                   @ np.linalg.pinv(W_meas)

    # ── UMAP (fit on measured, transform predictions) ────────────────────────────

    umap_model = UMAP(n_components=2, metric='euclidean',
                      random_state=11235, n_neighbors=5, min_dist=0.25)
    umap_meas_embed = umap_model.fit_transform(R_meas[valid_indices])
    umap_rob_embed  = umap_model.transform(R_rob[valid_indices])
    umap_std_embed  = umap_model.transform(R_std[valid_indices])

    def expand_lims(vals, pct=0.12):
        lo, hi = float(vals.min()), float(vals.max())
        pad = pct * (hi - lo)
        return [lo - pad, hi + pad]

    all_x = np.concatenate([umap_meas_embed[:, 0],
                             umap_rob_embed[:, 0],
                             umap_std_embed[:, 0]])
    all_y = np.concatenate([umap_meas_embed[:, 1],
                             umap_rob_embed[:, 1],
                             umap_std_embed[:, 1]])
    xlims = expand_lims(all_x)
    ylims = expand_lims(all_y)

    # ── Hover / colour helpers ───────────────────────────────────────────────────

    nat_labels = cat_labels[valid_indices[60:]]
    nat_names  = stim_names[valid_indices[60:]]
    npd_names  = stim_names[valid_indices[:60]]

    def format_name(name):
        if name.startswith('NPD_'):
            return name.replace('NPD_', 'NPD #')
        parts = name.split('_', 1)
        if len(parts) > 1:
            return parts[1].replace('_', ' ').title()
        return name.replace('_', ' ').title()

    hover_npd = [format_name(n) for n in npd_names]
    hover_nat = [format_name(n) for n in nat_names]

    # ── Save helpers ─────────────────────────────────────────────────────────────

    _umap_layout = dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(size=10, family='Times New Roman'),
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False, range=xlims),
        yaxis=dict(visible=False, range=ylims),
    )

    def save_umap(fig, name, width=DPI, height=DPI):
        out = args.output_dir
        fig.update_layout(
            legend=dict(visible=True, bgcolor='rgba(0,0,0,0)',
                        borderwidth=0, font=dict(size=8)),
            title=dict(font=dict(size=11, family='Times New Roman')),
            width=600, height=600,
        )
        fig.write_html(str(out / f'{name}.html'),
                       include_plotlyjs='directory', include_mathjax='cdn')
        pio.write_image(fig, str(out / f'{name}_annotated.png'),
                        width=width, height=height, scale=SCALE)
        pio.write_image(fig, str(out / f'{name}_annotated.svg'),
                        width=width, height=height, scale=1)
        fig.update_layout(showlegend=False, title=None)
        pio.write_image(fig, str(out / f'{name}_clean.png'),
                        width=width, height=height, scale=SCALE)
        pio.write_image(fig, str(out / f'{name}_clean.svg'),
                        width=width, height=height, scale=1)

    def save_box(fig, name, width=260, height=180):
        out = args.output_dir
        fig.update_layout(showlegend=True, width=width * 2, height=height * 2)
        fig.write_html(str(out / f'{name}.html'),
                       include_plotlyjs='directory', include_mathjax='cdn')
        pio.write_image(fig, str(out / f'{name}_annotated.png'),
                        width=width, height=height, scale=SCALE)
        pio.write_image(fig, str(out / f'{name}_annotated.svg'),
                        width=width, height=height, scale=1)
        fig.update_layout(
            showlegend=False,
            title=None,
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False, title=None),
        )
        pio.write_image(fig, str(out / f'{name}_clean.png'),
                        width=width, height=height, scale=SCALE)
        pio.write_image(fig, str(out / f'{name}_clean.svg'),
                        width=width, height=height, scale=1)

    # ── Plot 1: Observed responses (natural + NPD) ───────────────────────────────

    fig1 = go.Figure()
    for cat in np.unique(nat_labels):
        mask = nat_labels == cat
        idx  = np.where(mask)[0]
        fig1.add_trace(go.Scatter(
            x=umap_meas_embed[60 + idx, 0],
            y=umap_meas_embed[60 + idx, 1],
            mode='markers', name=cat,
            hovertext=[hover_nat[i] for i in idx],
            marker=dict(symbol='circle', size=6, color=color_lookup[cat],
                        line=dict(width=0.5, color='black')),
        ))
    fig1.add_trace(go.Scatter(
        x=umap_meas_embed[:60, 0], y=umap_meas_embed[:60, 1],
        mode='markers', name='NPD',
        hovertext=hover_npd,
        marker=dict(symbol='cross', size=6, color='rgba(0,0,0,0)',
                    line=dict(width=1.2, color='black')),
    ))
    fig1.update_layout(**_umap_layout, title=dict(text='Observed responses'))
    save_umap(fig1, 'umap_observed')
    print('Saved umap_observed')

    # ── Plot 2: NPD predictions (robust vs standard) ─────────────────────────────

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=umap_meas_embed[60:, 0], y=umap_meas_embed[60:, 1],
        mode='markers', name='Natural (measured)', hovertext=hover_nat,
        marker=dict(symbol='circle', size=6, color='lightgrey',
                    line=dict(width=0, color='lightgrey')),
    ))
    for embeds, color, name in [
        (umap_rob_embed, ROBUST_COLOR,   'Robust NPD'),
        (umap_std_embed, STANDARD_COLOR, 'Standard NPD'),
    ]:
        fig2.add_trace(go.Scatter(
            x=embeds[:60, 0], y=embeds[:60, 1],
            mode='markers', name=name, hovertext=hover_npd,
            marker=dict(symbol='cross', size=6, color='rgba(0,0,0,0)',
                        line=dict(width=1.0, color=color)),
        ))
    for i in range(60):
        fig2.add_trace(go.Scatter(
            x=[umap_rob_embed[i, 0], umap_std_embed[i, 0]],
            y=[umap_rob_embed[i, 1], umap_std_embed[i, 1]],
            mode='lines', name=npd_names[i], hovertext=hover_npd[i],
            line=dict(width=0.75, color='#777777'),
            hoverinfo='text', showlegend=False, opacity=1.0,
        ))
    fig2.update_layout(**_umap_layout,
                       title=dict(text='NPD predictions: robust vs standard'))
    save_umap(fig2, 'umap_npd_comparison')
    print('Saved umap_npd_comparison')

    # ── Plot 3: Natural predictions (robust vs standard) ─────────────────────────

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=umap_meas_embed[60:, 0], y=umap_meas_embed[60:, 1],
        mode='markers', name='Natural (measured)', hovertext=hover_nat,
        marker=dict(symbol='circle', size=6, color='lightgrey',
                    line=dict(width=0, color='lightgrey')),
    ))
    for embeds, color, name in [
        (umap_rob_embed, ROBUST_COLOR,   'Robust natural'),
        (umap_std_embed, STANDARD_COLOR, 'Standard natural'),
    ]:
        fig3.add_trace(go.Scatter(
            x=embeds[60:, 0], y=embeds[60:, 1],
            mode='markers', name=name, hovertext=hover_nat,
            marker=dict(symbol='circle', size=6, color='rgba(0,0,0,0)',
                        line=dict(width=1.0, color=color)),
        ))
    for i in range(60):
        fig3.add_trace(go.Scatter(
            x=[umap_rob_embed[60 + i, 0], umap_std_embed[60 + i, 0]],
            y=[umap_rob_embed[60 + i, 1], umap_std_embed[60 + i, 1]],
            mode='lines', name=nat_names[i], hovertext=hover_nat[i],
            line=dict(width=0.75, color='#777777'),
            hoverinfo='text', showlegend=False, opacity=1.0,
        ))
    fig3.update_layout(**_umap_layout,
                       title=dict(text='Natural predictions: robust vs standard'))
    save_umap(fig3, 'umap_nat_comparison')
    print('Saved umap_nat_comparison')

    # ── Plot 4: Distances (robust vs standard predictions) ───────────────────────

    dist_npd = np.linalg.norm(R_rob[:60,  :] - R_std[:60,  :], axis=-1)
    dist_nat = np.linalg.norm(R_rob[-60:, :] - R_std[-60:, :], axis=-1)

    box_w       = 0.25
    box_line_w  = 1.0   # thin: box edges encode quartiles, so keep them crisp
    g_nat, g_npd = 0.0, 0.4

    _box_layout = dict(
        boxmode='overlay',   # boxes sit at their explicit x0, no auto-offset
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(size=10, family='Times New Roman'),
        margin=dict(l=4, r=4, t=4, b=2),
        xaxis=dict(
            tickvals=[g_nat, g_npd],
            ticktext=['Natural', 'NPD'],
            showgrid=False, zeroline=False, showticklabels=True,
            range=[-0.3, 0.7],
        ),
        yaxis=dict(
            showgrid=True, gridcolor='rgba(128,128,128,0.3)',
            zeroline=True, zerolinecolor='black',
            showticklabels=True,
            rangemode='tozero',   # anchor both panels at 0
        ),
        showlegend=False,
    )

    # Box-and-whisker (no bar plots — Nature Neuroscience style): box spans the
    # inter-quartile range, centre line is the median, whiskers extend to the
    # most extreme point within 1.5 x IQR, and points beyond that are drawn.
    fig4 = go.Figure()
    for x_pos, vals, label, fill in [
        (g_nat, dist_nat, 'Natural', 'rgba(180,180,180,0.5)'),
        (g_npd, dist_npd, 'NPD',     'rgba(255,255,255,0.5)'),
    ]:
        fig4.add_trace(go.Box(
            x0=x_pos, y=vals, name=label,
            fillcolor=fill,
            line=dict(color='black', width=box_line_w),
            marker=dict(color='black', size=2, opacity=0.6),
            boxpoints='outliers', width=box_w, whiskerwidth=0.5,
            showlegend=False,
        ))

    fig4.update_layout(**_box_layout,
                       yaxis_title='Distance (robust − standard)',
                       title=dict(text='Robust vs standard prediction distance'))
    # Gridlines at 1000/2000/3000 only; 0 carries the black zeroline instead.
    fig4.update_yaxes(tick0=0, dtick=1000)
    save_box(fig4, 'distances', width=182, height=260)      # w/h = 0.70
    print('Saved distances')

    # ── Plot 4b: Accuracy (predictions vs measured) ──────────────────────────────

    dist_rob_npd = np.linalg.norm(R_rob[:60,  :] - R_meas[:60,  :], axis=-1)
    dist_std_npd = np.linalg.norm(R_std[:60,  :] - R_meas[:60,  :], axis=-1)
    dist_rob_nat = np.linalg.norm(R_rob[-60:, :] - R_meas[-60:, :], axis=-1)
    dist_std_nat = np.linalg.norm(R_std[-60:, :] - R_meas[-60:, :], axis=-1)

    sub_w = 0.14                   # width of each sub-box
    sub_gap = 0.02                 # small gap between the two boxes in a group
    sub_dx = (sub_w + sub_gap) / 2
    g_nat_b, g_npd_b = 0.0, 0.42  # group centers — gap between groups >> 0 within

    # Box-and-whisker, as in Plot 4: IQR box, median line, whiskers to the most
    # extreme point within 1.5 x IQR, outliers drawn individually.
    fig4b = go.Figure()
    for x_pos, rob_vals, std_vals in [
        (g_nat_b, dist_rob_nat, dist_std_nat),
        (g_npd_b, dist_rob_npd, dist_std_npd),
    ]:
        for dx, vals, color in [
            (-sub_dx, rob_vals, ROBUST_COLOR),
            (+sub_dx, std_vals, STANDARD_COLOR),
        ]:
            fig4b.add_trace(go.Box(
                x0=x_pos + dx, y=vals,
                fillcolor=color,
                line=dict(color='black', width=box_line_w),
                marker=dict(color='black', size=2, opacity=0.6),
                boxpoints='outliers', width=sub_w, whiskerwidth=0.5,
                showlegend=False,
            ))

    fig4b.update_layout(**_box_layout,
                        yaxis_title='Distance (predicted − measured)',
                        title=dict(text='Prediction accuracy'))
    fig4b.update_layout(xaxis=dict(
        tickvals=[g_nat_b, g_npd_b],  # ticks at group centers
        ticktext=['Natural', 'NPD'],
        showgrid=False, zeroline=False, showticklabels=True,
        range=[-0.22, 0.64],
    ))
    save_box(fig4b, 'distances_accuracy', width=192, height=240)  # w/h = 0.80
    print('Saved distances_accuracy')

    # ── RSM computation ──────────────────────────────────────────────────────────

    def compute_rsm(d):
        return 1 - squareform(pdist(d, 'correlation'))

    def spearman_brown(r):
        return 2 * r / (1 + r)

    def upper_tri(mat, k=1):
        return mat[np.triu_indices_from(mat, k=k)]

    rsm_group_avg = np.stack([compute_rsm(d[valid_indices]) for d in D_meas]).mean(0)

    npd_npd_block = upper_tri(rsm_group_avg[:60, :60])
    npd_nat_block = rsm_group_avg[:60, 60:].flatten()
    nat_nat_block = upper_tri(rsm_group_avg[60:, 60:])

    # Per-stimulus Spearman-Brown corrected split-half reliability (noise ceiling)
    corrs_within = np.stack([
        spearman_brown(mycorr(D_all_raw[i][0][valid_indices],
                              D_all_raw[i][1][valid_indices], axis=-1))
        for i in range(len(D_all_raw))
    ]).mean(0)   # (120,) — distribution used as violin, not collapsed to a scalar

    # ── Plot 5: RSM heatmap ───────────────────────────────────────────────────────

    n = rsm_group_avg.shape[0]
    mid = n / 2

    fig5 = go.Figure()
    fig5.add_trace(go.Heatmap(z=rsm_group_avg, colorscale='Reds', showscale=False))
    fig5.add_shape(type='line', x0=mid, y0=0,   x1=mid, y1=n,   line=dict(color='black'))
    fig5.add_shape(type='line', x0=0,   y0=mid, x1=n,   y1=mid, line=dict(color='black'))
    fig5.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(size=10, family='Times New Roman'),
        margin=dict(l=0, r=0, t=24, b=0),
        xaxis=dict(range=[0, n], constrain='domain', visible=False),
        yaxis=dict(range=[0, n], autorange='reversed', scaleanchor='x',
                   scaleratio=1, constrain='domain', visible=False),
        title=dict(text='Group average RSM'),
        width=DPI, height=DPI,
    )

    out = args.output_dir
    fig5.write_html(str(out / 'rsm_heatmap.html'),
                    include_plotlyjs='directory', include_mathjax='cdn')
    pio.write_image(fig5, str(out / 'rsm_heatmap_annotated.png'),
                    width=DPI, height=DPI, scale=SCALE)
    pio.write_image(fig5, str(out / 'rsm_heatmap_annotated.svg'),
                    width=DPI, height=DPI, scale=1)
    fig5.update_layout(title=None)
    pio.write_image(fig5, str(out / 'rsm_heatmap_clean.png'),
                    width=DPI, height=DPI, scale=SCALE)
    pio.write_image(fig5, str(out / 'rsm_heatmap_clean.svg'),
                    width=DPI, height=DPI, scale=1)
    print('Saved rsm_heatmap')

    # ── Plot 5b: RSM colorbar ─────────────────────────────────────────────────────

    zmin_val = float(np.nanmin(rsm_group_avg))
    zmax_val = float(np.nanmax(rsm_group_avg))
    cbar_gradient = np.linspace(zmin_val, zmax_val, 256).reshape(-1, 1)

    fig5b = go.Figure()
    fig5b.add_trace(go.Heatmap(
        z=cbar_gradient,
        colorscale='Reds',
        zmin=zmin_val,
        zmax=zmax_val,
        showscale=False,
    ))
    fig5b.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(size=10, family='Times New Roman'),
        margin=dict(l=4, r=4, t=24, b=4),
        xaxis=dict(visible=False),
        yaxis=dict(
            tickvals=[0, 64, 128, 192, 255],
            ticktext=[f'{v:.2f}' for v in np.linspace(zmin_val, zmax_val, 5)],
            side='right', showgrid=False, zeroline=False,
        ),
        title=dict(text='Similarity'),
        width=80, height=260,
    )

    fig5b.write_html(str(out / 'rsm_colorbar.html'),
                     include_plotlyjs='directory', include_mathjax='cdn')
    pio.write_image(fig5b, str(out / 'rsm_colorbar_annotated.png'),
                    width=80, height=260, scale=SCALE)
    pio.write_image(fig5b, str(out / 'rsm_colorbar_annotated.svg'),
                    width=80, height=260, scale=1)
    fig5b.update_layout(
        title=None,
        yaxis=dict(visible=False),
    )
    pio.write_image(fig5b, str(out / 'rsm_colorbar_clean.png'),
                    width=80, height=260, scale=SCALE)
    pio.write_image(fig5b, str(out / 'rsm_colorbar_clean.svg'),
                    width=80, height=260, scale=1)
    print('Saved rsm_colorbar')

    # ── Plot 6: RSM block violin plots with noise ceiling ────────────────────────

    fig6 = go.Figure()
    fig6.add_trace(go.Violin(y=npd_nat_block, name='NPD-NAT',
                             fillcolor='lightgrey', line_color='black',
                             spanmode='hard', points=False))
    fig6.add_trace(go.Violin(y=nat_nat_block, name='NAT-NAT',
                             fillcolor='lightgrey', line_color='black',
                             spanmode='hard', points=False))
    fig6.add_trace(go.Violin(y=corrs_within, name='Noise ceiling',
                             fillcolor='dimgrey', line_color='black',
                             spanmode='hard', points=False))
    fig6.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(size=10, family='Times New Roman'),
        margin=dict(l=4, r=4, t=24, b=4),
        legend=dict(visible=True, bgcolor='rgba(0,0,0,0)', borderwidth=0),
        yaxis=dict(
            showgrid=True, gridcolor='rgba(128,128,128,0.3)',
            zeroline=False, showticklabels=True,
            title='Pairwise similarity (r)',
        ),
        xaxis=dict(showticklabels=True),
        title=dict(text='Group average RSM — block distributions'),
        width=260, height=220,
    )

    fig6.write_html(str(out / 'rsm_violins.html'),
                    include_plotlyjs='directory', include_mathjax='cdn')
    pio.write_image(fig6, str(out / 'rsm_violins_annotated.png'),
                    width=260, height=220, scale=SCALE)
    pio.write_image(fig6, str(out / 'rsm_violins_annotated.svg'),
                    width=260, height=220, scale=1)
    fig6.update_layout(
        showlegend=False, title=None,
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False, title=None),
    )
    pio.write_image(fig6, str(out / 'rsm_violins_clean.png'),
                    width=260, height=220, scale=SCALE)
    pio.write_image(fig6, str(out / 'rsm_violins_clean.svg'),
                    width=260, height=220, scale=1)
    print('Saved rsm_violins')

    # ── Statistics ───────────────────────────────────────────────────────────────

    print('\n── Statistical tests ────────────────────────────────────────────────────')

    t, p = ttest_ind(nat_nat_block, npd_nat_block)
    print(f'NAT-NAT vs NPD-NAT similarity (independent t-test):  t={t:.3f}, p={p:.4g}')

    t, p = ttest_ind(dist_nat, dist_npd)
    print(f'Robust-standard distance, NAT vs NPD (independent t-test):  t={t:.3f}, p={p:.4g}')

    t, p = ttest_rel(dist_rob_nat, dist_std_nat)
    print(f'Prediction accuracy, robust vs standard — NAT (paired t-test):  t={t:.3f}, p={p:.4g}')

    t, p = ttest_rel(dist_rob_npd, dist_std_npd)
    print(f'Prediction accuracy, robust vs standard — NPD (paired t-test):  t={t:.3f}, p={p:.4g}')

    print(f'All figures written to {args.output_dir}')


if __name__ == '__main__':
    main()
