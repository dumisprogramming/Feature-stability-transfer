# Paper 3 supplementary validation — 18 October 2026

Read `RESULTS_REPORT.md` first. The fixed plan, runnable analysis, all supplementary output tables, integrity checks and independent numerical verification are included. Original experimental files are not duplicated in this package and must be obtained from the author's research repository/archive.

These are supplementary analyses of known saved results, not an independent raw-data replication. No original seal or confirmatory result was changed.

## Reproduction

Use Python 3.11 or later with NumPy, pandas, SciPy and scikit-learn. Exact execution versions are recorded in `results/execution_manifest.json`. This run used Python's installed environment without changing the archived dependency file.

From this package directory, with the original repository extracted elsewhere:

```bash
python run_supplementary_validation.py --repo /path/to/Paper3_GitHub_Repository --output replay_results --check-only
python run_supplementary_validation.py --repo /path/to/Paper3_GitHub_Repository --output replay_results
python independent_reference_check.py --repo /path/to/Paper3_GitHub_Repository --output replay_results
```

Use a fresh output directory; the analysis refuses to overwrite completed predictions. The script runs structural checks, then writes a pre-fit execution manifest containing the plan, code and input hashes before fitting the supplementary regressions. It never retrains intrusion detectors or writes to the input repository.

`results/predictions.csv` contains all 1,800 predictions across the full-seed and five seed-deletion analyses. `results/scores.csv` reports every specification overall, per fold and per model. Improvement columns use baseline MAE minus candidate MAE: positive is better. `results/association_sensitivity.csv` contains all 27 deletion correlations. These correlations and deletion ranges are descriptive, not significance tests or confidence intervals.

The independent checker uses NumPy least squares on unscaled covariates with separately grouped training data. It checks all 144 fits and 1,800 predictions against the primary scikit-learn implementation.

No new software/data reuse license is assigned by this package. See the original repository and dataset providers for applicable terms.
