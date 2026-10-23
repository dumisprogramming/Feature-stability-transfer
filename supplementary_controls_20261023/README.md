# Post hoc source performance controls

This folder reports a saved-result comparison conducted on 23 October 2026 after the original Paper 3 results and earlier supplementary validation were known. Read PLAN.md first. The historical confirmatory analysis is unchanged.

From the repository root:

```bash
python -m pip install -r supplementary_controls_20261023/requirements.txt
python supplementary_controls_20261023/run_baseline_comparison.py --repo . --output outputs/performance_controls_recomputed
```

The output directory must not already exist. Input files are the repository's 60 Paper3_Source_Stability permutation-importance Parquet files, their stability_complete.json records, source_only_warning_scores.csv and Paper3_Final_Evaluation/transfer_analysis_all_180.csv. No raw datasets, model refitting or Google Drive mount is required for this comparison. The script checks input hashes and writes separate outputs without modifying originals.

Seven fixed OLS specifications compare source performance variability and level with the two primary stability warnings, always including model identity. Evaluation removes the held-out dataset from both source and target roles in meta-regression training. Training outcomes average two targets and test outcomes three; folds share other datasets. All seed-deletion results are retained. Scores are descriptive and no new confirmatory p values are calculated.

Results do not establish a general failure of source-only indicators. No independent datasets, per-flow disagreement, source feature-distribution distances, alternative temporal widths, natural-prevalence evaluation or attack-family tests are added here. Those require additional inputs and experiments.

Supplementary Information S6 contains readable tables and qualifications. Response_to_Comments.md tracks addressed and outstanding requests. The repository has not been published or modified automatically.
