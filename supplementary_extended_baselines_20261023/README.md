# Additional source-only baselines — 23 October 2026

This package contains a NEW POST-HOC analysis of the original saved source predictions and environmental features. It does not train new intrusion detectors or rerun the raw-data pipeline. Original artifacts remain unchanged. See PLAN.md for fixed definitions, categorical coding choices and evaluation scope.

## Inputs verified

The supplied ZIP has 188 files with a valid CRC. Individual historical records agree with all 120 saved calibration/source-test prediction files and all 20 environmental panels. The 40 source prediction panels align across three classifiers by unique row ID and source label. Twenty panels contain 860,000 environmental rows in total across seeds, not necessarily distinct flows across seeds. The predictor schema has 49 columns. Source-test and environmental row IDs do not overlap within each source/seed. All 60 saved source-test balanced accuracies were independently recomputed successfully. Checksums establish consistency with the supplied records, not independent proof of historical execution.

## Results

Strict role-excluded validation, original four datasets, five seeds, three classifiers; lower MAE is better. A unit is source dataset × seed × classifier. Pooled all-seed scores use 60 units.

| Specification | MAE | RMSE |
|---|---:|---:|
| Model identity only | 0.050100 | 0.062685 |
| Identity + original primary warnings | 0.066898 | 0.083791 |
| Identity + source model disagreement | 0.082288 | 0.104253 |
| Identity + source feature heterogeneity | 0.093482 | 0.108922 |
| Identity + both additional warnings | 0.133426 | 0.188600 |
| Identity + all four warnings | 0.113817 | 0.158858 |

Model identity alone has the lowest pooled MAE among these fixed specifications in the all-seed analysis and each of five seed-deletion repetitions. However, results vary across datasets: adding all four warnings improves BoT-IoT MAE from 0.046430 to 0.028554 and CSE-CIC-IDS2018 from 0.085539 to 0.033985, while worsening ToN-IoT from 0.030607 to 0.234379 and UNSW-NB15 from 0.037822 to 0.158350. These descriptive comparisons do not establish population superiority or a universal failure of source-only warnings.

All 144 regression designs are full rank. There are 1,800 OOF predictions across six specifications and six seed selections. Independent NumPy regression agrees within 1e-10. The 600 predictions shared with the previous comparison agree exactly in this runtime. Independent source heterogeneity computation on one complete panel differs by less than 1e-15.

## Run in Colab with the original files on Drive

Place this complete folder at the top level of Feature-stability-transfer on GitHub, alongside the earlier supplementary_controls_20261023 folder. Do not upload the supplied per-flow input panels merely to run this code. Run a fresh clone, or pull the updated repository if it already exists. Then:

```python
from google.colab import drive
from pathlib import Path
from datetime import datetime, timezone
import subprocess
import sys

drive.mount('/content/drive')
repo = Path('/content/Feature-stability-transfer')
data_root = Path('/content/drive/MyDrive/research/researchdata/NIDS_Research/Paper3_DataSet')
script_folder = repo / 'supplementary_extended_baselines_20261023'
assert (script_folder / 'run_extended_baselines.py').is_file(), 'Upload the new folder to GitHub and clone/pull it first.'
subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', str(script_folder / 'requirements.txt')], check=True)
stamp = datetime.now(timezone.utc).strftime('run_%Y%m%d_%H%M%S_%f')
output = data_root / 'supplementary_extended_baselines_20261023' / stamp
subprocess.run([
    sys.executable, str(script_folder / 'run_extended_baselines.py'),
    '--data-root', str(data_root), '--repo', str(repo), '--output', str(output)
], check=True)
print('Saved results:', output)
```

The output directory must be new. This script reads the original Drive inputs and writes only new result files. The supplied results folder is the preparation run; a new Colab run creates its own provenance record. Historical software is not assumed to match the current runtime: the manifest records versions actually used.

## Scope and pending work

This addresses model disagreement and one explicitly defined marginal source feature-distribution heterogeneity baseline. It does not provide genuine cross-validation variance, independent datasets, alternate temporal widths, natural prevalence evaluation, or attack-family transfer. Those remain pending. Class-balanced sampling, binned marginal distributions, high-cardinality categorical codes and four-dataset regression extrapolation limit interpretation. The held-out source is excluded from training target roles, but remaining targets are shared across folds. Training and test loss averages involve two and three target datasets respectively.

No manuscript or supplement was edited by this analysis step. Integration should retain the post-hoc designation, all six specifications, per-dataset heterogeneity, and the original confirmatory results.

## Files

- PLAN.md: choices recorded before this new fit.
- run_extended_baselines.py: verification and analysis code.
- requirements.txt: dependencies used for preparation (Python 3.12; Colab versions recorded on rerun).
- results/: aggregate indicators, feature-level distances, alignment counts, OOF regression predictions, all-seed and seed-deletion scores, diagnostics and input/output hashes.
- independent_checks.json: additional implementation/source-metric checks from preparation.
- SHA256SUMS.txt: package file checksums, excluding this checksum file itself.

The package contains aggregate indicators and regression outputs, not raw input panels, per-flow predictions, labels or trained model binaries. No new reuse license is assigned.
