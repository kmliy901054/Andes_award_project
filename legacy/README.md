# Legacy / Experimental Scripts

This folder contains scripts that are **not part of the main runtime path**.

## Files

- `LSTM.py`
  - Early/experimental pose regression training script.
  - Uses a hard-coded data folder path and legacy score list.
  - Kept for reference only.

- `test.py`
  - Offline pose extraction utility.
  - Reads `Training_Data/*.mp4` and writes `pose_output/*_pose.npy`.

- `t.py`
  - Offline merge utility.
  - Merges sensor CSV and pose NPY into `combined_output/*_combined.csv`.

- `score_1dim.py`
  - Small helper script for converting inline score text into a Python list.

## Notes

- These files are retained for reproducibility and historical context.
- For the main app flow, use root-level scripts documented in the main README.
