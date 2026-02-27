# -*- coding: utf-8 -*-
"""
analysis.py

Statistical analysis and figure generation for:
"Neural network-based encoding in free-viewing fMRI with gaze-aware models"
Gözükara et al., 2026

Produces all main and supplementary figures from the paper.
Paths are configured centrally in config.py.
"""

# ============================================================
# Imports
# Install dependencies: pip install siibra nilearn nibabel seaborn
# ============================================================

import os
import pickle
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import siibra

from nilearn import plotting, image, surface, datasets
from scipy import stats
from scipy.stats import wilcoxon, zscore, ttest_rel, ttest_1samp
from statsmodels.stats.multitest import multipletests
import statsmodels.stats.multitest as multitest

# ============================================================
# Path configuration
# ============================================================

from config import (RESULTS_DIR, WEIGHTS_DIR, FIGURES_DIR,
                    REMODNAV_DIR, AAL_MASK, REF_NIFTI)

PRF_RESULTS_DIR = RESULTS_DIR  # pRF results are written to the same results directory

# ============================================================
# Participants and ROIs
# ============================================================

sub_list = ['sub-01','sub-02','sub-03','sub-04','sub-09','sub-10',
            'sub-14','sub-15','sub-16','sub-17','sub-18','sub-19','sub-20']

# Julich atlas ROI groupings (bilateral)
roi_list_nested = [
    ['hOc1 left',  'hOc1 right'],
    ['hOc2 left',  'hOc2 right'],
    ['hOc3v left', 'hOc3v right'],
    ['hOc4lp left','hOc4lp right',
     'hOc4la left','hOc4la right',
     'hOc5 left',  'hOc5 right'],
    ['FG1 left','FG1 right',
     'FG2 left','FG2 right',
     'FG3 left','FG3 right',
     'FG4 left','FG4 right'],
    ['STS1 left','STS1 right',
     'STS2 left','STS2 right'],
]

roi_list = ['v1', 'v2', 'v3', 'LO', 'FG', 'STS']

# ============================================================
# Setup: atlas and AAL mask
# ============================================================

# Load AAL mask
aal_mask_img = nib.load(AAL_MASK)
aal_affine   = aal_mask_img.affine
aal_mask     = aal_mask_img.get_fdata()
aal_mask[aal_mask > 0] = 1  # correct interpolation artefacts, make binary

# Load reference functional image for atlas resampling
searchlight_ref_img = nib.load(REF_NIFTI)

# Build Julich atlas ROI coordinate masks
atlas = siibra.atlases.get("human")
space = atlas.spaces.get("MNI Colin 27")

roi_coords_list = []
for roi_names in roi_list_nested:
    roi_coords_mask = np.zeros_like(searchlight_ref_img.get_fdata()[:, :, :, 0])
    for roi_name in roi_names:
        roi      = atlas.get_region(roi_name, parcellation='julich 2.9')
        roi_mask = roi.get_regional_mask(space, maptype="labelled").fetch()
        resampled_mask = image.resample_img(
            roi_mask,
            target_affine=searchlight_ref_img.affine,
            target_shape=searchlight_ref_img.shape[:3],
            interpolation='nearest',
            force_resample=True,
            copy_header=True
        )
        roi_coords_mask[np.where(resampled_mask.get_fdata() > 0)] += 1
    roi_coords_list.append(np.where(roi_coords_mask > 0))

# ============================================================
# Helper functions
# ============================================================

def results_to_nifti(results, aal_mask, sig_mask=None):
    """Map a 1D voxel results array back into MNI volume space."""
    idx = np.where(aal_mask > 0)
    vol = np.zeros((61, 73, 61))
    if sig_mask is None:
        vol[idx[0], idx[1], idx[2]] = results
    else:
        vol[idx[0][sig_mask], idx[1][sig_mask], idx[2][sig_mask]] = results[sig_mask]
    return nib.Nifti1Image(vol, aal_affine)


def nifti_to_surface(nifti_image):
    """Project a volumetric NIfTI image onto the fsaverage surface."""
    fsaverage = datasets.fetch_surf_fsaverage()
    texture_left  = surface.vol_to_surf(nifti_image, fsaverage.pial_left,
                                        interpolation='nearest', radius=3.0, n_samples=20)
    texture_right = surface.vol_to_surf(nifti_image, fsaverage.pial_right,
                                        interpolation='nearest', radius=3.0, n_samples=20)
    # Set near-zero values to NaN for surface transparency
    for tex in [texture_left, texture_right]:
        tex[(tex >= -0.0005) & (tex <= 0.0005)] = np.nan
    return texture_left, texture_right


def long_plot(texture_left, texture_right, title='Title', saving=True,
              min_val=-1, max_val=1, views=['lateral', 'medial'],
              cmap='Spectral_r', norm=None, show=True):
    """Plot a surface stat map in 4 panels (lateral/medial × left/right)."""
    fsaverage = datasets.fetch_surf_fsaverage()
    fig, axes = plt.subplots(1, 4, figsize=(15, 3), subplot_kw={'projection': '3d'})
    fig.suptitle(title, fontsize=12, x=0.45)

    from_left_to_right = [
        (views[0], 'left'), (views[0], 'right'),
        (views[1], 'left'), (views[1], 'right')
    ]
    if norm is None:
        norm = plt.Normalize(vmin=min_val, vmax=max_val)

    for i, (view, hemi) in enumerate(from_left_to_right):
        texture = texture_left if hemi == 'left' else texture_right
        plotting.plot_surf_stat_map(
            fsaverage.infl_left  if hemi == 'left' else fsaverage.infl_right,
            texture, hemi=hemi, view=view, colorbar=False,
            bg_map=fsaverage.sulc_left if hemi == 'left' else fsaverage.sulc_right,
            cmap=cmap, axes=axes[i], vmax=max_val, vmin=min_val, bg_on_data=True
        )

    fig.subplots_adjust(top=1, right=0.85, wspace=0, hspace=0.00002, left=0.05, bottom=0)
    cbar_ax = fig.add_axes([0.95, 0.05, 0.02, 0.8])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([min_val, max_val])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Pearson r', rotation=270, labelpad=15)

    if saving:
        plt.savefig(os.path.join(FIGURES_DIR, 'PrecisionFigure_' + title),
                    transparent=True, dpi=300)
    if show:
        plt.show()


def julich_roi_results(nifti):
    """Extract per-ROI voxel values from a NIfTI image using Julich atlas masks."""
    roi_results = []
    for roi_coords in roi_coords_list:
        vals = nifti.get_fdata()[roi_coords]
        if np.sum(vals) == 0:
            roi_results.append(np.zeros(vals.shape[0]))
        else:
            roi_results.append(vals[vals > 0])
    return roi_results


def add_stat_annotations(ax, data, x_col, y_col, hue_col, spacing_multiplier=0.1):
    """
    Add Bonferroni-corrected pairwise t-test significance brackets to a violin plot.
    spacing_multiplier controls vertical spacing between annotation brackets.
    """
    rois   = data[x_col].unique()
    models = data[hue_col].unique()

    valid_data = data[~np.isnan(data[y_col])]
    y_max      = valid_data[y_col].max()
    y_min      = valid_data[y_col].min()
    y_increment = (y_max - y_min) * spacing_multiplier

    # Collect all pairwise p-values
    pair_pvals = []
    pair_tests = []
    for roi in rois:
        pairs = [(models[i], models[j])
                 for i in range(len(models)) for j in range(i + 1, len(models))]
        for m1, m2 in pairs:
            s1 = data[(data[x_col] == roi) & (data[hue_col] == m1)][y_col].dropna()
            s2 = data[(data[x_col] == roi) & (data[hue_col] == m2)][y_col].dropna()
            if len(s1) > 1 and len(s2) > 1:
                if len(s1) == len(s2):
                    _, p = stats.ttest_rel(s1, s2, alternative='two-sided')
                else:
                    _, p = stats.ttest_ind(s1, s2, alternative='two-sided')
                pair_pvals.append(p)
                pair_tests.append((roi, m1, m2))

    if not pair_pvals:
        return

    _, corrected_pvals, _, _ = multipletests(pair_pvals, method='bonferroni')

    sig_markers = {0.001: '***', 0.01: '**', 0.05: '*'}

    # Compute violin x-positions
    num_models   = len(models)
    violin_width = 0.8 / num_models
    roi_positions = {roi: i for i, roi in enumerate(rois)}
    violin_positions = {}
    for roi in rois:
        offsets = np.linspace(-(num_models - 1) / 2 * violin_width,
                               (num_models - 1) / 2 * violin_width, num_models)
        for i, model in enumerate(models):
            violin_positions[(roi, model)] = roi_positions[roi] + offsets[i]

    annotation_heights = {roi: y_max + y_increment for roi in rois}

    for (roi, m1, m2), p_val in zip(pair_tests, corrected_pvals):
        if p_val >= 0.05:
            continue
        sig = next(marker for thresh, marker in sorted(sig_markers.items()) if p_val < thresh)
        x1 = violin_positions[(roi, m1)]
        x2 = violin_positions[(roi, m2)]
        y  = annotation_heights[roi]
        annotation_heights[roi] += y_increment
        bar_h = y_increment * 0.3
        ax.plot([x1, x1, x2, x2], [y, y + bar_h, y + bar_h, y], color='black', linewidth=2)
        ax.text((x1 + x2) / 2, y + bar_h, sig, ha='center', va='bottom', fontsize=16)


def nested_list_to_df(mode_data_frame_inputs, model_names, roi_names):
    """
    Convert nested list of ROI results [model][roi][values] into a long-form DataFrame
    with columns: Value, ROI, Model.
    """
    dfs = []
    for model_results, model_name in zip(mode_data_frame_inputs, model_names):
        df = pd.DataFrame(np.array(model_results).T, columns=roi_names)
        df['Model'] = model_name
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    return combined.melt(id_vars='Model', var_name='ROI', value_name='Value')


def spatial_entropy_2d(heatmap):
    """Shannon entropy of a 2D spatial distribution."""
    prob_dist = heatmap / heatmap.sum()
    prob_dist = prob_dist[prob_dist > 0]
    return -np.sum(prob_dist * np.log2(prob_dist))


# ============================================================
# Load results
# ============================================================

print('Loading encoding model results...')

baseline_performance         = np.zeros((13, 19629))
no_PRF_baseline_performance  = np.zeros((13, 19629))
noPRF_centerfix_performance  = np.zeros((13, 19629))
PCA_baseline_performance     = np.zeros((13, 19629))

for sub_count, sub in enumerate(sub_list):
    baseline_data = np.load(
        os.path.join(RESULTS_DIR, f'{sub}_hyperlayer_fast_baseline_results.npy'),
        allow_pickle=True)
    baseline_performance[sub_count] = baseline_data[:, 0]

    gaze_data = np.load(
        os.path.join(RESULTS_DIR, f'{sub}_noPRF_hyperlayer_fast_baseline_results.npy'),
        allow_pickle=True)
    no_PRF_baseline_performance[sub_count] = gaze_data[:, 0]

    center_data = np.load(
        os.path.join(RESULTS_DIR, f'{sub}_centergaze_hyperlayer_fast_baseline_results.npy'),
        allow_pickle=True)
    noPRF_centerfix_performance[sub_count] = center_data[:, 0]

    pca_data = np.load(
        os.path.join(RESULTS_DIR, f'{sub}_PCA_hyperlayer_fast_baseline_results.npy'),
        allow_pickle=True)
    PCA_baseline_performance[sub_count] = pca_data[:, 0]

# pRF+Gaze model (only available for 11 subjects)
prf_gaze_subs = ['sub-01','sub-02','sub-03','sub-04','sub-09','sub-10',
                  'sub-14','sub-15','sub-17','sub-19','sub-20']
precision_performance = np.zeros((13, 19629))
for sub_count, sub in enumerate(prf_gaze_subs):
    path = os.path.join(PRF_RESULTS_DIR, f'precision_results_dictionary_list_{sub}.pkl')
    with open(path, 'rb') as f:
        sub_results = pickle.load(f)
    precision_performance[sub_count] = np.array(
        [v['test_corr'] for v in sub_results])

# Weights and gaze distributions (used in Figs 3a-d)
print('Loading weights and gaze distributions...')
with open(os.path.join(WEIGHTS_DIR, 'all_subs_weights_over_space.pkl'), 'rb') as f:
    all_subs_weights_over_space = pickle.load(f)
with open(os.path.join(WEIGHTS_DIR, 'all_subs_gaze_dists.pkl'), 'rb') as f:
    all_subs_gaze_dists = pickle.load(f)

print('All data loaded.')

# ============================================================
# Figure 2a — Performance histograms
# ============================================================

print('Plotting Figure 2a...')

fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
bin_edges = np.linspace(-0.05, 0.25, 100)
palette   = ['yellow', 'orange', 'red', 'darkgrey', 'grey']

ax.hist(np.mean(no_PRF_baseline_performance, axis=0),
        alpha=0.7, label='GazeOnly',    bins=bin_edges, color=palette[0], zorder=-1)
ax.hist(np.mean(no_PRF_baseline_performance, axis=0),
        alpha=0.7, label='PRF+Gaze',   bins=bin_edges, color=palette[1])
ax.hist(np.mean(baseline_performance, axis=0),
        alpha=0.7, label='Baseline',   bins=bin_edges, color=palette[2], zorder=-2)
ax.hist(np.mean(noPRF_centerfix_performance, axis=0),
        alpha=0.7, label='CenterOnly', bins=bin_edges, color=palette[3], zorder=-3)
ax.hist(np.mean(PCA_baseline_performance, axis=0),
        alpha=0.7, label='PCABaseline',bins=bin_edges, color=palette[4], zorder=-4)

ax.set_xlabel('Pearson r', fontsize=20)
ax.set_ylabel('Voxel Count', fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=18)
for spine in ['bottom', 'left']:
    ax.spines[spine].set_linewidth(4)
ax.spines[['top', 'right']].set_visible(False)
leg = ax.legend()
for text in leg.get_texts():
    text.set_fontsize(18)

fig.savefig(os.path.join(FIGURES_DIR, 'Fig2a_Performance_Histograms.png'),
            transparent=True, dpi=300)
plt.show()

# ============================================================
# Figure 2b/c — Brain maps (group average, FDR corrected)
# ============================================================

print('Plotting Figure 2b (brain maps)...')

model_results_dict = {
    'GazeOnly':    no_PRF_baseline_performance,
    'CenterOnly':  noPRF_centerfix_performance,
    'Baseline':    baseline_performance,
    'PCABaseline': PCA_baseline_performance,
}

for model_name, model_results in model_results_dict.items():
    fz = 0.5 * np.log((1 + model_results) / (1 - model_results))
    res = ttest_1samp(fz, popmean=0, axis=0)
    fdr_ps = stats.false_discovery_control(res.pvalue)
    group_avg = np.mean(model_results, axis=0)
    nifti = results_to_nifti(group_avg, aal_mask, sig_mask=fdr_ps < 0.05)
    tex_l, tex_r = nifti_to_surface(nifti)
    long_plot(tex_l, tex_r, title=f'Fig2b_{model_name}',
              saving=True, min_val=0, max_val=0.3, cmap='YlOrRd_r')
    print(f'{model_name}: {(fdr_ps < 0.05).sum() / len(fdr_ps):.1%} voxels significant')

print('Plotting Figure 2c (difference maps)...')

gaze_base_difference   = no_PRF_baseline_performance - baseline_performance
gaze_center_difference = no_PRF_baseline_performance - noPRF_centerfix_performance
center_base_difference = baseline_performance - noPRF_centerfix_performance

for diff, name in zip(
    [gaze_base_difference, gaze_center_difference, center_base_difference],
    ['Fig2c_GazeMinusBaseline', 'Fig2c_GazeMinusCenter', 'Fig2c_CenterMinusBaseline']
):
    fz  = 0.5 * np.log((1 + diff) / (1 - diff))
    res = ttest_1samp(fz, popmean=0, axis=0)
    fdr_ps = stats.false_discovery_control(res.pvalue)
    group_avg = np.mean(diff, axis=0)
    nifti = results_to_nifti(group_avg, aal_mask, sig_mask=fdr_ps < 0.05)
    tex_l, tex_r = nifti_to_surface(nifti)
    long_plot(tex_l, tex_r, title=name,
              saving=True, min_val=-0.1, max_val=0.1, cmap='Spectral_r')

# ============================================================
# Figure 2d — Group violin plots
# ============================================================

print('Plotting Figure 2d (group violins)...')

models_to_plot = [no_PRF_baseline_performance, baseline_performance,
                  noPRF_centerfix_performance, PCA_baseline_performance]
model_names    = ['GazeOnly', 'Baseline', 'CenterOnly', 'PCABaseline']
palette_violins = ['yellow', 'red', 'darkgrey', 'grey']

mode_data = []
for model_results in models_to_plot:
    idx = np.where(aal_mask > 0)
    vol = np.zeros((13, 61, 73, 61))
    vol[:, idx[0], idx[1], idx[2]] = model_results
    vol[vol == 0] = np.nan
    roi_results = [np.nanmean(vol[:, c[0], c[1], c[2]], axis=1)
                   for c in roi_coords_list]
    mode_data.append(roi_results)

dfs = [pd.DataFrame(np.array(r).T, columns=roi_list) for r in mode_data]
for df, name in zip(dfs, model_names):
    df['Model'] = name
df_long = pd.concat(dfs, ignore_index=True).melt(
    id_vars='Model', var_name='ROI', value_name='Pearson r')

fig, ax = plt.subplots(figsize=(25, 10), dpi=300)
sns.violinplot(data=df_long, x='ROI', y='Pearson r',
               inner='stick', hue='Model', gap=0.0,
               density_norm='count', palette=palette_violins,
               alpha=1, split=False, linewidth=3, linecolor='black', ax=ax)
ax.set_ylim(-0.1, 0.45)
for spine in ['bottom', 'left']:
    ax.spines[spine].set_linewidth(6)
ax.spines[['top', 'right']].set_visible(False)
ax.set_xlabel('ROI', fontsize=32)
ax.set_ylabel('Pearson r', fontsize=32)
ax.tick_params(axis='both', which='major', labelsize=28)
add_stat_annotations(ax, df_long, 'ROI', 'Pearson r', 'Model', spacing_multiplier=0.1)
fig.text(0.05, 0.02, '* p<0.05, ** p<0.01, *** p<0.001 (Bonferroni-corrected)', fontsize=20)
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, 'Fig2d_Group_Violins.png'), transparent=True, dpi=300)
plt.show()

# ============================================================
# Figure 3a — Gaze and weight distribution heatmaps
# ============================================================

print('Plotting Figure 3a (heatmaps)...')

fig, axs = plt.subplots(13, 3, figsize=(5, 10))
n = 1

for sub_idx, sub in enumerate(sub_list):
    sub_gaze_dist        = all_subs_gaze_dists[sub_idx]
    sub_weights_over_space = all_subs_weights_over_space[sub_idx]

    sub_baseline = np.load(
        os.path.join(RESULTS_DIR, f'{sub}_hyperlayer_fast_baseline_results.npy'),
        allow_pickle=True)
    sorted_voxels = np.argsort(sub_baseline[:, 0])

    axs[sub_idx, 0].imshow(sub_gaze_dist.T, interpolation='none',
                            cmap='Spectral_r', origin='upper')
    axs[sub_idx, 1].imshow(
        np.nanmean(sub_weights_over_space[:, :, sorted_voxels[:n]], axis=2),
        interpolation='none', cmap='Spectral_r')
    axs[sub_idx, 2].imshow(
        np.nanmean(sub_weights_over_space[:, :, np.flip(sorted_voxels)[:n]], axis=2),
        interpolation='none', cmap='Spectral_r')

for ax in axs.flatten():
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

fig.savefig(os.path.join(FIGURES_DIR, 'Fig3a_Gaze_Weight_Heatmaps.png'),
            transparent=True, dpi=300)
plt.show()

# ============================================================
# Figure 3b — Gaze–weight correlation brain maps
# ============================================================

print('Plotting Figure 3b (gaze-weight correlation brain maps)...')

all_subs_gaze_weight_corrs = []

for sub_idx, (sub_gaze_dist, sub_weights) in enumerate(
        zip(all_subs_gaze_dists, all_subs_weights_over_space)):
    features_flat = sub_weights.reshape(-1, 19629)
    gaze_flat     = sub_gaze_dist.T.flatten()
    X = features_flat - features_flat.mean(axis=0)
    y = gaze_flat     - gaze_flat.mean()
    corrs = np.dot(X.T, y) / (np.sqrt(np.sum(X ** 2, axis=0)) * np.sqrt(np.sum(y ** 2)))
    all_subs_gaze_weight_corrs.append(corrs)

    nifti    = results_to_nifti(corrs, aal_mask)
    tex_l, tex_r = nifti_to_surface(nifti)
    long_plot(tex_l, tex_r,
              title=f'Fig3b_{sub}_GazeWeightCorr',
              saving=True, min_val=-0.6, max_val=0.6, cmap='seismic')

# Group median map
median_corrs = np.median(np.array(all_subs_gaze_weight_corrs), axis=0)
nifti = results_to_nifti(median_corrs, aal_mask)
tex_l, tex_r = nifti_to_surface(nifti)
long_plot(tex_l, tex_r, title='Fig3b_GroupMedian_GazeWeightCorr',
          saving=True, min_val=-0.6, max_val=0.6, cmap='seismic')

# ============================================================
# Figure 3c (top) — Performance vs. gaze–weight correlation
# ============================================================

print('Plotting Figure 3c top (performance vs gaze-weight correlation)...')

mean_gaze_weight_corr = np.nanmean(np.array(all_subs_gaze_weight_corrs), axis=1)
mean_gaze_perf        = np.nanmean(no_PRF_baseline_performance, axis=1)
mean_base_perf        = np.nanmean(baseline_performance,  axis=1)

fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
ax.scatter(mean_gaze_weight_corr, mean_base_perf,
           label='Baseline', color='red',    edgecolors='black', s=150)
ax.scatter(mean_gaze_weight_corr, mean_gaze_perf,
           label='Gaze',     color='yellow', edgecolors='black', s=150)

slope, intercept, r, p, _ = stats.linregress(mean_gaze_weight_corr, mean_base_perf)
ax.axline([0, intercept], slope=slope, color='darkred', ls='--', zorder=0,
          label='Least Squares Fit')
print(f'Baseline: r={r:.3f}, p={p:.4f}')

slope, intercept, r, p, _ = stats.linregress(mean_gaze_weight_corr, mean_gaze_perf)
ax.axline([0, intercept], slope=slope, color='orange', ls='--', zorder=0,
          label='Least Squares Fit')
print(f'Gaze: r={r:.3f}, p={p:.4f}')

for i in range(len(mean_base_perf)):
    ax.plot([mean_gaze_weight_corr[i], mean_gaze_weight_corr[i]],
            [mean_base_perf[i], mean_gaze_perf[i]],
            'k-', alpha=0.3, linewidth=0.5, zorder=0)

ax.set_xlabel('Spatial Feature - Gaze Correlation\n(Pearson R)', fontsize=24)
ax.set_ylabel('Model Performance\n(Pearson R)', fontsize=24)
ax.tick_params(axis='both', which='major', labelsize=18)
fig.legend(fontsize=16)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, 'Fig3c_top_Performance_vs_GazeWeightCorr.png'),
            transparent=True, dpi=300)
plt.show()

# ============================================================
# Figure 3c (bottom) — Number of fixations vs. performance
# ============================================================

print('Plotting Figure 3c bottom (fixations vs performance)...')

per_sub_fixation_num = []
for sub in sub_list:
    n_fixations = 0
    for run_num in range(1, 9):
        tsv_path = os.path.join(REMODNAV_DIR, f'{sub}-run{run_num}.tsv')
        gaze_data = pd.read_csv(tsv_path, sep='\t')
        n_fixations += gaze_data['label'].value_counts().get('FIXA', 0)
    per_sub_fixation_num.append(n_fixations)

per_sub_fixation_num = np.array(per_sub_fixation_num, dtype=float)
per_sub_fixation_num -= per_sub_fixation_num.min()
per_sub_fixation_num /= per_sub_fixation_num.max()

fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
ax.scatter(per_sub_fixation_num, mean_gaze_perf,
           label='Gaze',     color='yellow', edgecolors='black', s=150)
ax.scatter(per_sub_fixation_num, mean_base_perf,
           label='Baseline', color='red',    edgecolors='black', s=150)

slope, intercept, r, p, _ = stats.linregress(per_sub_fixation_num, mean_gaze_perf)
ax.axline([0, intercept], slope=slope, color='orange',  ls='--', zorder=0)
print(f'Gaze vs fixations:     r={r:.3f}, p={p:.4f}')

slope, intercept, r, p, _ = stats.linregress(per_sub_fixation_num, mean_base_perf)
ax.axline([0, intercept], slope=slope, color='darkred', ls='--', zorder=0)
print(f'Baseline vs fixations: r={r:.3f}, p={p:.4f}')

for i in range(len(mean_base_perf)):
    ax.plot([per_sub_fixation_num[i], per_sub_fixation_num[i]],
            [mean_base_perf[i], mean_gaze_perf[i]],
            'k-', alpha=0.3, linewidth=0.5, zorder=0)

ax.set_xlabel('Normalized Number of Fixations', fontsize=24)
ax.set_ylabel('Performance\n(Pearson R)', fontsize=24)
ax.tick_params(axis='both', which='major', labelsize=18)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, 'Fig3c_bottom_NumFixations_vs_Performance.png'),
            transparent=True, dpi=300)
plt.show()

# ============================================================
# Figure 3d — Shannon entropy of baseline model weight distributions
# ============================================================

print('Plotting Figure 3d (weight distribution entropy)...')

all_subs_weight_entropies = []
for sub_idx in range(13):
    sub_weights = all_subs_weights_over_space[sub_idx]
    entropies   = [spatial_entropy_2d(sub_weights[:, :, v]) for v in range(19629)]
    all_subs_weight_entropies.append(zscore(entropies))

mean_entropies = np.nanmean(np.array(all_subs_weight_entropies), axis=0)
nifti = results_to_nifti(mean_entropies, aal_mask)
tex_l, tex_r = nifti_to_surface(nifti)
long_plot(tex_l, tex_r, title='Fig3d_Weight_Entropy',
          saving=True, min_val=-2, max_val=2, cmap='seismic')

# ============================================================
# Supplementary Figure 1/2/3 — Single subject brain maps
# ============================================================

print('Plotting Supplementary Figures 1/2/3 (single subject brain maps)...')

for sub_count, sub in enumerate(sub_list):
    sub_color_max = np.max([
        no_PRF_baseline_performance[sub_count].max(),
        noPRF_centerfix_performance[sub_count].max(),
        baseline_performance[sub_count].max(),
        PCA_baseline_performance[sub_count].max(),
    ])
    for model_results, model_name in zip(
        [no_PRF_baseline_performance[sub_count],
         noPRF_centerfix_performance[sub_count],
         baseline_performance[sub_count],
         PCA_baseline_performance[sub_count]],
        ['GazeOnly', 'CenterOnly', 'Baseline', 'PCABaseline']
    ):
        n_trs = 3111
        t_stat = model_results * np.sqrt((n_trs - 2) / (1 - model_results ** 2))
        p_vals = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n_trs - 2))
        fdr_ps = stats.false_discovery_control(p_vals)
        nifti  = results_to_nifti(model_results, aal_mask, sig_mask=fdr_ps < 0.05)
        tex_l, tex_r = nifti_to_surface(nifti)
        long_plot(tex_l, tex_r,
                  title=f'SuppFig_{sub}_{model_name}',
                  saving=True, min_val=0, max_val=sub_color_max, cmap='YlOrRd_r')

    # Gaze minus baseline difference map
    diff   = no_PRF_baseline_performance[sub_count] - baseline_performance[sub_count]
    t_stat = diff * np.sqrt((n_trs - 2) / (1 - diff ** 2))
    p_vals = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n_trs - 2))
    fdr_ps = stats.false_discovery_control(p_vals)
    nifti  = results_to_nifti(diff, aal_mask, sig_mask=fdr_ps < 0.05)
    tex_l, tex_r = nifti_to_surface(nifti)
    long_plot(tex_l, tex_r,
              title=f'SuppFig3_{sub}_GazeMinusBaseline',
              saving=True, min_val=-0.35, max_val=0.35, cmap='seismic')

# ============================================================
# Supplementary Figure 5 — Single subject violin plots
# ============================================================

print('Plotting Supplementary Figure 5 (single subject violins)...')

fig, axs = plt.subplots(13, 1, figsize=(15, 4 * 13), dpi=400, sharex=True)

for sub_count, sub in enumerate(sub_list):
    sub_models = [no_PRF_baseline_performance[sub_count],
                  baseline_performance[sub_count],
                  noPRF_centerfix_performance[sub_count],
                  PCA_baseline_performance[sub_count]]
    sub_model_names = ['GazeOnly', 'Baseline', 'CenterOnly', 'PCABaseline']

    sub_mode_data = []
    for model_results in sub_models:
        vol = np.zeros((61, 73, 61))
        idx = np.where(aal_mask > 0)
        vol[idx[0], idx[1], idx[2]] = model_results
        vol[vol == 0] = np.nan
        roi_results = [vol[c[0], c[1], c[2]] for c in roi_coords_list]
        sub_mode_data.append(roi_results)

    df = nested_list_to_df(sub_mode_data, sub_model_names, roi_list)

    sns.violinplot(x='ROI', y='Value', hue='Model', data=df,
                   split=False, inner='quartile', gap=0.0,
                   density_norm='count',
                   palette=['yellow', 'red', 'darkgrey', 'grey'],
                   alpha=1, linewidth=3, linecolor='black',
                   ax=axs[sub_count])

    axs[sub_count].set_title(f'{sub} Comparison of Models across 6 ROIs', fontsize=16)
    axs[sub_count].set_xlabel(
        'Region of Interest (ROI)' if sub_count == 12 else '', fontsize=14)
    axs[sub_count].set_ylabel('Pearson r', fontsize=14)
    if sub_count > 0:
        axs[sub_count].legend_.remove()

fig.tight_layout()
plt.subplots_adjust(hspace=0.25)
fig.savefig(os.path.join(FIGURES_DIR, 'SuppFig5_SingleSub_Violins.png'),
            transparent=True, dpi=300)

# ============================================================
# Supplementary Figure 6 — pRF + Gaze violin plots
# ============================================================

print('Plotting Supplementary Figure 6 (pRF violin plots)...')

prf_models_to_plot = [precision_performance, no_PRF_baseline_performance,
                      baseline_performance, noPRF_centerfix_performance,
                      PCA_baseline_performance]
prf_model_names    = ['PRF+Gaze', 'GazeOnly', 'Baseline', 'CenterOnly', 'PCABaseline']

prf_mode_data = []
for model_results in prf_models_to_plot:
    idx = np.where(aal_mask > 0)
    vol = np.zeros((13, 61, 73, 61))
    vol[:, idx[0], idx[1], idx[2]] = model_results
    vol[vol == 0] = np.nan
    roi_results = [np.nanmean(vol[:, c[0], c[1], c[2]], axis=1)
                   for c in roi_coords_list]
    prf_mode_data.append(roi_results)

dfs = [pd.DataFrame(np.array(r).T, columns=roi_list) for r in prf_mode_data]
for df, name in zip(dfs, prf_model_names):
    df['Model'] = name
df_long_prf = pd.concat(dfs, ignore_index=True).melt(
    id_vars='Model', var_name='ROI', value_name='Pearson r')

fig, ax = plt.subplots(figsize=(25, 10), dpi=300)
sns.violinplot(data=df_long_prf, x='ROI', y='Pearson r',
               inner='stick', hue='Model', gap=0.0,
               density_norm='count',
               palette=['orange', 'yellow', 'red', 'darkgrey', 'grey'],
               alpha=1, split=False, linewidth=3, linecolor='black', ax=ax)
ax.set_ylim(-0.1, 0.45)
for spine in ['bottom', 'left']:
    ax.spines[spine].set_linewidth(6)
ax.spines[['top', 'right']].set_visible(False)
ax.set_xlabel('ROI', fontsize=32)
ax.set_ylabel('Pearson r', fontsize=32)
ax.tick_params(axis='both', which='major', labelsize=28)
add_stat_annotations(ax, df_long_prf, 'ROI', 'Pearson r', 'Model', spacing_multiplier=0.1)
fig.text(0.05, 0.02, '* p<0.05, ** p<0.01, *** p<0.001 (Bonferroni-corrected)', fontsize=20)
plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, 'SuppFig6_pRF_Violins.png'),
            transparent=True, dpi=300)
plt.show()

print('All figures saved to', FIGURES_DIR)
