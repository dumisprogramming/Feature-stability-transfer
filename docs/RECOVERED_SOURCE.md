# Recovered Paper 3 implementation

## What was recovered

`notebooks/Paper3_CompleteCode.ipynb` is the uploaded notebook preserved byte for byte, including saved outputs. It has 20 code cells. `Untitled60.ipynb` contains only the recent file-search program and is intentionally excluded. The notebook/code-cell hashes and extraction mapping are in `reports/notebook_source_map.json`.

Eleven upstream cells are also provided as exact UTF-8 source extracts, without edits or added headers. They are historical Colab/Drive workflow code, not a newly validated pipeline.

| Notebook cell (zero-based) | Script in `Paper3_Code/source_stages/` | Stage |
|---|---|---|
| 0 | `00_dataset_inventory.py` | Mount Drive; inventory dataset ZIPs and CSV schemas |
| 1 | `01_temporal_audit.py` | Streaming label/time counts and eligible six-hour blocks |
| 2 | `02_historical_protocol_freeze.py` | Original timestamped protocol creation |
| 3 | `03_partition_indices.py` | Deterministic source partition indices |
| 4 | `04_feature_materialization.py` | Ordered 49-feature schema, materialization and duplicate signatures |
| 5 | `05_duplicate_group_sanitization.py` | A1, duplicate-group exclusions and balanced partitions |
| 6 | `06_pretraining_gate.py` | Feature-quality, hash and overlap checks |
| 7 | `07_source_training_calibration.py` | Source preprocessing, LR/RF/XGB fitting, threshold calibration and seal |
| 8 | `08_source_feature_stability.py` | Permutation importance, pairwise stability, warning scores and seal |
| 9 | `09_target_plan_freeze.py` | Amendment A2 |
| 10 | `10_target_readonly_preflight.py` | Read-only preflight |

Later notebook cells call the four original target-stage scripts already in `Paper3_Code/`, or inspect model/preprocessor structures. Cells 18 and 19 call the same final evaluator for its two-pass A3 freeze/reveal workflow. Cell 15 preserves a failed prediction invocation (`AssertionError`); cell 17 records a later successful 180-job prediction run. Keeping that history is not a claim that every saved cell succeeded.

## How to use this release now

For numerical verification, follow the portable saved-results commands in the root README. The new `tools/verify_recovered_stability.py` compiles only the recovered pairwise aggregation function, without running notebook top-level code, mounting Drive, loading model files or creating experimental seals. It recomputes summaries from the 60 supplied importance tables and compares them with the supplied completion records.

For source review, read the original notebook or the numbered scripts. The stage order above documents dependencies; it is not an instruction to use Run all on the historical notebook.

## Historical replay constraints

- The paths are hard-coded to `/content/drive/MyDrive/research/researchdata/NIDS_Research/Paper3_DataSet`. The four ZIPs expected there are `NF-BoT-IoT-v3.zip`, `NF-CSE-CIC-IDS2018-v3.zip`, `NF-ToN-IoT-v3.zip`, and `NF-UNSW-NB15-v3.zip`.
- Cell 2 unconditionally writes a timestamped protocol. Re-executing it changes the protocol hash and can overwrite the existing frozen record. Preserve the supplied protocol files for a historical replay; do not execute that cell over original evidence.
- Cells 7 and 8 write timestamped source-model and stability seals, including on resumed runs. Later stages expect the original exact seal hashes. A newly generated seal will generally differ. A fresh experiment needs a separately maintained dependency chain and current timestamps; do not backdate it or disable checks to force historical hash agreement.
- Several stages skip already completed outputs. The saved training-cell output says every dataset/seed checkpoint was reused, so the notebook does not contain a fresh training log for that invocation.
- Cell 8 contains an accidental `drivesyz` ROOT assignment immediately overwritten by the correct Drive path. Both statements are preserved as received.
- Raw ZIPs, partition/materialization/sanitization Parquet files, fitted models, target panels, prediction files and several small original manifests are not included. See `MISSING_FILES.md` for the smallest files to collect next.
- The historical workflow additionally imports PyArrow, joblib, XGBoost and Google Colab. The portable requirements are not a complete historical training lockfile.

A full replay should use a separate workspace and a documented version of the environment. This package has not retrained the 60 models, regenerated importance scores from raw traffic, or re-predicted 180 target jobs. Notebook saved outputs are supporting records, not independent execution or timestamp verification.
