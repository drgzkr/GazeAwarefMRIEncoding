#!/usr/bin/env python3
"""
generate_weights_and_gaze.py

Generates the two precomputed pkl files required by analysis.py:

    derivatives/weights_and_gaze/all_subs_weights_over_space.pkl
    derivatives/weights_and_gaze/all_subs_gaze_dists.pkl

Run this after completing the full encoding pipeline (Steps 1-5), specifically
after precision_baseline.py has saved per-subject weight files.

Required pipeline outputs
-------------------------
For each subject in sub_list:
  - derivatives/results/{sub}_hyperlayer_fast_baseline_weights.npy
      Shape: (164864, 19629)  i.e. (7*16*1472, 19629)
      Saved by precision_baseline.py via fastRidge.get_weights()

  - derivatives/fixations/{sub}-run{run}.npy
      Shape: (N_fixations, 3)  columns: [frame_id, x_pixel, y_pixel]
      Saved by remodnav_frame_selection.py (Step 1)

Output formats
--------------
all_subs_weights_over_space : list of 13 arrays, each shape (7, 16, 19629)
    For each subject and voxel, the L2 norm of the weight vector over the
    channel dimension (1472 channels), leaving the 7x16 spatial grid intact.
    This is the spatial weight distribution used in Figures 3a, 3b, 3d.

all_subs_gaze_dists : list of 13 arrays, each shape (16, 7)
    For each subject, a 2D histogram of fixation density over the 7x16
    feature grid, stored in x-major order (16, 7) so that .T gives (7, 16)
    for display with imshow. This matches the convention in analysis.py where
    sub_gaze_dist.T is passed to imshow and sub_gaze_dist.T.flatten() is used
    for the 112-element correlation vector.
"""

import os
import pickle
import numpy as np
from tqdm import tqdm
from config import RESULTS_DIR, FIXATIONS_DIR, WEIGHTS_DIR

# ------------------------------------------------------------------ #
# Constants — must match the encoding pipeline
# ------------------------------------------------------------------ #

sub_list = [
    'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-09', 'sub-10',
    'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20',
]

N_VOXELS   = 19629
FEAT_H     = 7       # spatial height of hyperlayer feature map
FEAT_W     = 16      # spatial width  of hyperlayer feature map
N_CHANNELS = 1472    # 64+128+256+512+512
assert FEAT_H * FEAT_W * N_CHANNELS == 164864

# Screen resolution of the StudyForrest movie stimulus (pixels)
SCREEN_W = 1280
SCREEN_H = 544

# ------------------------------------------------------------------ #
# 1.  Spatial weight distributions  (7 x 16 x 19629 per subject)
# ------------------------------------------------------------------ #
# precision_baseline.py calls fastRidge.get_weights() which returns
#
#     W = X_train.T @ alpha      shape: (164864, 19629)
#
# where 164864 = 7 * 16 * 1472 and the feature vector was formed by
#
#     np.reshape(sub_features, (N_frames, 164864))
#
# The reshape uses C-order (row-major), so the flat index maps as:
#
#     flat_idx = channel * (7*16) + row * 16 + col
#
# i.e. axis order in the original array is (N, C, H, W), flattened
# across C, H, W.  Reshaping W back gives (C=1472, H=7, W=16, V=19629).
# The L2 norm over C collapses to (H=7, W=16, V=19629).

print("=" * 60)
print("Step 1 — Computing spatial weight distributions")
print("=" * 60)

all_subs_weights_over_space = []

for sub in tqdm(sub_list, desc="Subjects (weights)"):
    weight_path = os.path.join(
        RESULTS_DIR, f"{sub}_hyperlayer_fast_baseline_weights.npy"
    )
    if not os.path.exists(weight_path):
        raise FileNotFoundError(
            f"Weight file not found: {weight_path}\n"
            "Run precision_baseline.py first."
        )

    # Shape: (164864, 19629)
    learned_ws = np.load(weight_path, allow_pickle=True)

    if learned_ws.shape != (N_CHANNELS * FEAT_H * FEAT_W, N_VOXELS):
        raise ValueError(
            f"Unexpected weight shape for {sub}: {learned_ws.shape}. "
            f"Expected ({N_CHANNELS * FEAT_H * FEAT_W}, {N_VOXELS})."
        )

    # Reshape to (C, H, W, V) then take L2 norm over C -> (H, W, V)
    # The encoding feature vector was formed by reshaping (N, C, H, W) -> (N, C*H*W)
    # so the flat layout is C-contiguous over [C, H, W].
    ws_spatial = learned_ws.reshape(N_CHANNELS, FEAT_H, FEAT_W, N_VOXELS)
    ws_l2 = np.linalg.norm(ws_spatial, axis=0)   # (7, 16, 19629)

    all_subs_weights_over_space.append(ws_l2)
    print(f"  {sub}: weight map shape {ws_l2.shape}, "
          f"mean L2 = {ws_l2.mean():.4f}")

# ------------------------------------------------------------------ #
# 2.  Gaze density distributions  (16 x 7 per subject)
# ------------------------------------------------------------------ #
# Fixation files from Step 1 contain columns [frame_id, x_pixel, y_pixel].
# We map pixel coordinates onto the 7x16 feature grid (same mapping used
# in precision_gaze.py) and accumulate a 2D histogram.
#
# Storage convention: (16, 7) i.e. x-major, matching the .T usage in
# analysis.py:
#   imshow(sub_gaze_dist.T, ...)   -> displays as (7 rows, 16 cols)
#   sub_gaze_dist.T.flatten()      -> 112-element vector, row-major over H then W

print("\n" + "=" * 60)
print("Step 2 — Computing gaze density distributions")
print("=" * 60)

all_subs_gaze_dists = []

for sub in tqdm(sub_list, desc="Subjects (gaze)"):
    # Accumulate all fixation coordinates across 8 runs
    all_x = []
    all_y = []

    for run in ['1', '2', '3', '4', '5', '6', '7', '8']:
        fix_path = os.path.join(FIXATIONS_DIR, f"{sub}-run{run}.npy")
        if not os.path.exists(fix_path):
            raise FileNotFoundError(
                f"Fixation file not found: {fix_path}\n"
                "Run remodnav_frame_selection.py (Step 1) first."
            )
        fixations = np.load(fix_path, allow_pickle=True)  # (N, 3)
        all_x.append(fixations[:, 1])   # x pixel coordinate
        all_y.append(fixations[:, 2])   # y pixel coordinate

    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)

    # Convert pixel coordinates to feature-grid indices
    # (same clipping / mapping as in precision_gaze.py)
    gaze_x = all_x / (SCREEN_W / FEAT_W)   # -> [0, FEAT_W)
    gaze_y = all_y / (SCREEN_H / FEAT_H)   # -> [0, FEAT_H)

    gaze_x = np.clip(np.floor(gaze_x).astype(int), 0, FEAT_W - 1)
    gaze_y = np.clip(np.floor(gaze_y).astype(int), 0, FEAT_H - 1)

    # 2D histogram: bins are feature-grid cells
    # hist shape: (FEAT_W, FEAT_H) = (16, 7) because x -> cols, y -> rows
    hist, _, _ = np.histogram2d(
        gaze_x, gaze_y,
        bins=[np.arange(FEAT_W + 1), np.arange(FEAT_H + 1)],
    )
    # Normalise to a probability distribution
    hist = hist / hist.sum()

    all_subs_gaze_dists.append(hist)   # (16, 7)
    print(f"  {sub}: gaze dist shape {hist.shape}, "
          f"total fixations = {len(all_x)}")

# ------------------------------------------------------------------ #
# 3.  Save
# ------------------------------------------------------------------ #

os.makedirs(WEIGHTS_DIR, exist_ok=True)

weights_path = os.path.join(WEIGHTS_DIR, 'all_subs_weights_over_space.pkl')
gaze_path    = os.path.join(WEIGHTS_DIR, 'all_subs_gaze_dists.pkl')

with open(weights_path, 'wb') as f:
    pickle.dump(all_subs_weights_over_space, f)
print(f"\nSaved: {weights_path}")

with open(gaze_path, 'wb') as f:
    pickle.dump(all_subs_gaze_dists, f)
print(f"Saved: {gaze_path}")

# ------------------------------------------------------------------ #
# 4.  Quick sanity checks
# ------------------------------------------------------------------ #

print("\n" + "=" * 60)
print("Sanity checks")
print("=" * 60)

# Check shapes
assert len(all_subs_weights_over_space) == 13
assert all_subs_weights_over_space[0].shape == (FEAT_H, FEAT_W, N_VOXELS), \
    f"Got {all_subs_weights_over_space[0].shape}"

assert len(all_subs_gaze_dists) == 13
assert all_subs_gaze_dists[0].shape == (FEAT_W, FEAT_H), \
    f"Got {all_subs_gaze_dists[0].shape}"

# Check analysis.py reshape is consistent:
#   features_flat = sub_weights.reshape(-1, 19629)  -> (112, 19629)
#   gaze_flat     = sub_gaze_dist.T.flatten()        -> (112,)
w_flat = all_subs_weights_over_space[0].reshape(-1, N_VOXELS)
g_flat = all_subs_gaze_dists[0].T.flatten()
assert w_flat.shape[0] == g_flat.shape[0] == FEAT_H * FEAT_W, \
    f"Flat dimension mismatch: weights {w_flat.shape[0]}, gaze {g_flat.shape[0]}"

print(f"  weights_over_space[0].shape  : {all_subs_weights_over_space[0].shape}  ✓")
print(f"  gaze_dists[0].shape          : {all_subs_gaze_dists[0].shape}  ✓")
print(f"  Flat dim for correlation      : {w_flat.shape[0]}  ✓")
print(f"  Gaze dist sums to 1          : {all_subs_gaze_dists[0].sum():.6f}  ✓")
print("\nDone. Both pkl files are ready for analysis.py.")
