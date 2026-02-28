"""
memory_footprint.py
====================
Computes and reports the memory footprint of the baseline and gaze-aware
encoding model pipelines described in:

Gözükara et al. (2025). Neural network-based encoding in free-viewing
fMRI with gaze-aware models.

All variables are directly from the paper (Methods section).
No data files are required to run this script.

Usage
-----
    python memory_footprint.py

Requirements
------------
    numpy (any recent version)

Output
------
Prints a summary table to stdout and saves a LaTeX-formatted version
to memory_footprint_table.tex in the current directory.
"""

import numpy as np

# ── 1. Dataset constants (from Methods) ─────────────────────────────────────
N_VOLUMES = 3599    # total fMRI TRs across all 8 runs
N_VOXELS  = 19629   # voxels in the visual-cortex mask (AAL atlas)
TR        = 2.0     # repetition time in seconds

# ── 2. VGG-19 hyperlayer architecture (from Methods) ────────────────────────
# Five max-pooling layers; channels per layer in a standard VGG-19
C_LAYERS = [64, 128, 256, 512, 512]

# Unified spatial dimensions after rescaling all layers (from Methods)
H_BAR, W_BAR = 7, 16       # height x width of rescaled hyperlayer
C_BAR        = sum(C_LAYERS)  # = 1472 total channels

# Native spatial dimensions of each max-pool layer.
# Input image: (H=224, W=527) preserving 2.35:1 cinematic aspect ratio.
# Each pooling stage halves both spatial dimensions.
NATIVE_H = [112, 56, 28, 14,  7]
NATIVE_W = [264, 132, 66, 33, 16]   # ≈ NATIVE_H * (527/224)

# ── 3. Memory unit ──────────────────────────────────────────────────────────
FLOAT32_BYTES = 4   # float32 storage (PyTorch default)

def human_readable(n_bytes):
    """Convert byte count to a human-readable string (KB / MB / GB)."""
    if n_bytes >= 1e9:
        return f"{n_bytes / 1e9:.2f} GB"
    if n_bytes >= 1e6:
        return f"{n_bytes / 1e6:.2f} MB"
    return f"{n_bytes / 1e3:.2f} KB"

# ── 4. Features per TR ───────────────────────────────────────────────────────
# Baseline: full rescaled hyperlayer (H_BAR x W_BAR x C_BAR per TR)
baseline_features_per_tr = H_BAR * W_BAR * C_BAR   # 164,864

# Gaze-aware: single spatial location sampled per TR (1 x 1 x C_BAR)
gaze_features_per_tr = C_BAR                         # 1,472

reduction_factor = baseline_features_per_tr / gaze_features_per_tr  # 112

# ── 5. Feature time-series (design matrix X) ─────────────────────────────────
# Shape: (N_VOLUMES, n_features)  — what ridge regression loads into RAM
X_baseline_bytes = N_VOLUMES * baseline_features_per_tr * FLOAT32_BYTES
X_gaze_bytes     = N_VOLUMES * gaze_features_per_tr     * FLOAT32_BYTES

# ── 6. Weight matrix W ───────────────────────────────────────────────────────
# Shape: (n_features, N_VOXELS)  — learned parameters stored after fitting
W_baseline_bytes = N_VOXELS * baseline_features_per_tr * FLOAT32_BYTES
W_gaze_bytes     = N_VOXELS * gaze_features_per_tr     * FLOAT32_BYTES

# ── 7. Response matrix Y ─────────────────────────────────────────────────────
# Shape: (N_VOLUMES, N_VOXELS)  — shared by both models
Y_bytes = N_VOLUMES * N_VOXELS * FLOAT32_BYTES

# ── 8. Total ridge regression working memory (X + Y + W) ────────────────────
total_baseline_bytes = X_baseline_bytes + Y_bytes + W_baseline_bytes
total_gaze_bytes     = X_gaze_bytes     + Y_bytes + W_gaze_bytes

# ── 9. Native VGG feature maps (before spatial rescaling) ────────────────────
# Stored on disk / in GPU memory during forward pass — not used in ridge fitting
native_per_frame = sum(h * w * c for h, w, c in
                       zip(NATIVE_H, NATIVE_W, C_LAYERS))
native_total_bytes = native_per_frame * N_VOLUMES * FLOAT32_BYTES

# ── Print summary ─────────────────────────────────────────────────────────────
print("=" * 65)
print("  Memory Footprint: Baseline vs. Gaze-Aware Encoding Models")
print("=" * 65)

print(f"\nKey variables")
print(f"  fMRI volumes (N)    : {N_VOLUMES}")
print(f"  Voxels in mask      : {N_VOXELS:,}")
print(f"  Hyperlayer shape    : {H_BAR} x {W_BAR} x {C_BAR}  (H x W x C)")
print(f"  Baseline features/TR: {baseline_features_per_tr:,}")
print(f"  Gaze-aware feat./TR : {gaze_features_per_tr:,}")
print(f"  Parameter reduction : {reduction_factor:.0f}x")

print(f"\nNative VGG feature maps (pre-rescaling)")
for i, (h, w, c) in enumerate(zip(NATIVE_H, NATIVE_W, C_LAYERS)):
    size = h * w * c * FLOAT32_BYTES
    print(f"  pool{i+1}: {h:3d} x {w:3d} x {c:3d}  =  {h*w*c:>9,} floats  "
          f"({human_readable(size)} / frame)")
print(f"  Full movie ({N_VOLUMES} TRs)          :  "
      f"{human_readable(native_total_bytes)}")

print(f"\nDesign matrix X  (N_volumes x n_features)")
print(f"  Baseline    ({N_VOLUMES} x {baseline_features_per_tr:,}): "
      f"{human_readable(X_baseline_bytes)}")
print(f"  Gaze-aware  ({N_VOLUMES} x {gaze_features_per_tr:,}    ): "
      f"{human_readable(X_gaze_bytes)}")

print(f"\nWeight matrix W  (n_features x N_voxels)")
print(f"  Baseline    ({baseline_features_per_tr:,} x {N_VOXELS:,}): "
      f"{human_readable(W_baseline_bytes)}")
print(f"  Gaze-aware  ({gaze_features_per_tr:,}     x {N_VOXELS:,}): "
      f"{human_readable(W_gaze_bytes)}")

print(f"\nResponse matrix Y  (shared, {N_VOLUMES} x {N_VOXELS:,}): "
      f"{human_readable(Y_bytes)}")

print(f"\nTotal ridge regression working memory  (X + Y + W)")
print(f"  Baseline    : {human_readable(total_baseline_bytes)}")
print(f"  Gaze-aware  : {human_readable(total_gaze_bytes)}")
print(f"  Reduction   : {total_baseline_bytes / total_gaze_bytes:.1f}x")
print("=" * 65)

# ── 10. Write LaTeX table ─────────────────────────────────────────────────────
latex = r"""\begin{table}[ht]
\centering
\caption{
    \textbf{Memory footprint comparison: baseline vs.\ gaze-aware encoding model.}
    All quantities are derived analytically from the pipeline parameters
    (see Methods). Storage assumes float32 (4 bytes per value).
    Working memory refers to the peak RAM required during ridge regression
    (design matrix $X$ + response matrix $Y$ + weight matrix $W$).
}
\label{tab:memory}
\begin{tabular}{lrr}
\hline
\textbf{Quantity} & \textbf{Baseline} & \textbf{Gaze-aware} \\
\hline
Features per TR                     & """ + f"{baseline_features_per_tr:,}" + r""" & """ + f"{gaze_features_per_tr:,}" + r""" \\
Model parameters (\#voxels $\times$ features) & """ + f"{N_VOXELS*baseline_features_per_tr:,}" + r""" & """ + f"{N_VOXELS*gaze_features_per_tr:,}" + r""" \\
Design matrix $X$                   & """ + human_readable(X_baseline_bytes) + r""" & """ + human_readable(X_gaze_bytes) + r""" \\
Weight matrix $W$                   & """ + human_readable(W_baseline_bytes) + r""" & """ + human_readable(W_gaze_bytes) + r""" \\
Response matrix $Y$ (shared)        & \multicolumn{2}{c}{""" + human_readable(Y_bytes) + r"""} \\
\hline
\textbf{Total working memory}       & \textbf{""" + human_readable(total_baseline_bytes) + r"""} & \textbf{""" + human_readable(total_gaze_bytes) + r"""} \\
Reduction factor                    & \multicolumn{2}{c}{$""" + f"{total_baseline_bytes/total_gaze_bytes:.0f}" + r"""\times$} \\
\hline
\end{tabular}
\end{table}
"""
print(latex)
# with open("memory_footprint_table.tex", "w") as f:
#    f.write(latex)
#
# print("\nLaTeX table written to: memory_footprint_table.tex")
