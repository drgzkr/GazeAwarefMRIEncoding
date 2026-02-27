#!/usr/bin/env bash
# run_remodnav.sh
#
# Runs Remodnav on the raw StudyForrest eye-tracking physio files to produce
# classified gaze event files (.tsv) for each subject and run.
#
# Prerequisite: pip install remodnav
#
# Input files are expected at:
#   RAW_ET_DIR/{sub}_ses-movie_task-movie_run-{run}_recording-eyegaze_physio.tsv
#
# Output files are written to:
#   OUTPUT_DIR/{sub}-run{run}.tsv
#
# These output files are the input to remodnav_frame_selection.py (Step 1).
# If you have already obtained the pre-classified Remodnav files from the
# StudyForrest eye-tracking data release, you can skip this script and place
# those files directly in the output directory (raw/remodnav/ as set in config.py).
#
# Remodnav parameters (StudyForrest-specific):
#   PIXEL_PITCH   -- derived from the display geometry of the StudyForrest setup
#   SAMPLING_RATE -- EyeLink recording rate in Hz

set -euo pipefail

# ============================================================
# Configuration — update these paths before running
# ============================================================

# Directory containing the raw physio .tsv files from OpenNeuro / StudyForrest
RAW_ET_DIR="/path/to/raw/eyetracking"

# Output directory — should match REMODNAV_DIR in config.py
OUTPUT_DIR="/path/to/raw/remodnav"

# Remodnav parameters (do not change unless you know the display specs differ)
PIXEL_PITCH="0.03299221912188374"
SAMPLING_RATE="1000"

# ============================================================
# Subjects and runs
# ============================================================

SUBJECTS=(
    "sub-01" "sub-02" "sub-03" "sub-04" "sub-05" "sub-06"
    "sub-09" "sub-10" "sub-14" "sub-15" "sub-16" "sub-17"
    "sub-18" "sub-19" "sub-20"
)
# Note: sub-05 and sub-06 are included here for completeness but are excluded
# from the encoding analysis due to low eye-tracking quality (IQR criterion).

RUNS=(1 2 3 4 5 6 7 8)

# ============================================================
# Run Remodnav
# ============================================================

mkdir -p "$OUTPUT_DIR"

for sub in "${SUBJECTS[@]}"; do
    for run in "${RUNS[@]}"; do

        # Zero-pad run number to match StudyForrest filename convention
        run_padded=$(printf "%02d" "$run")

        input="${RAW_ET_DIR}/${sub}_ses-movie_task-movie_run-${run_padded}_recording-eyegaze_physio.tsv"
        output="${OUTPUT_DIR}/${sub}-run${run}.tsv"

        if [ ! -f "$input" ]; then
            echo "WARNING: input not found, skipping: $input"
            continue
        fi

        if [ -f "$output" ]; then
            echo "Skipping (already exists): $output"
            continue
        fi

        echo "Processing: $sub run $run"
        remodnav "$input" "$output" "$PIXEL_PITCH" "$SAMPLING_RATE"

    done
done

echo "Done. Classified event files written to: $OUTPUT_DIR"
