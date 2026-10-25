# Completed source family exclusion experiment

Supplementary Information S13 reports this post hoc detector training ablation. All 450 fits and 1,800 evaluations completed on 25 October 2026 in approximately 75 minutes. The original full-training refits reproduced all 180 prior unstratified external confusion-count records exactly. Original source models and sealed files were not overwritten.

## Result-level verification

From this folder, run `python verify_report.py`. Python 3 standard-library modules are sufficient; no model libraries are required for this saved-report route. Outputs go to a new timestamped outputs/ directory. The command checks all 2,261 report hashes, code/plan/context identity, the 450-fit and 1,800-evaluation grids, all eligibility decisions, source support counts and pins against the recovery report, 4,020 family-recall records, all count-derived BA/recall/FPR values, and matched differences. It compares all 180 full-training external reference confusion records with the earlier natural-prevalence report. The verified aggregate findings are preserved in verification/. Do not overwrite those saved files with new outputs.

## Findings

Twenty-six dataset-specific families passed the fixed common five-seed support rule (3/9/7/7 by source dataset). There are 60 full-training reference refits and 390 family-excluded refits. Exclusion removes attack rows of the selected family from training and calibration, then refits source preprocessing and the original classifier configuration. Source-test and target outcomes do not set thresholds.

Across 390 matched excluded-family source-test comparisons, equal-fit mean recall fell from 0.929736 to 0.671535: 241 lower, seven higher and 142 unchanged. Across 1,170 external paired binary evaluations, mean BA change was -0.006875: 607 lower and 563 higher. Dataset/model and family tables preserve the heterogeneous effects. Families have equal fit weight within datasets; datasets with more eligible families contribute more to pooled means. Reference external BA is repeated per eligible family and must not be directly compared with the differently weighted original 180-job mean.

## Scope and limitations

The primary excluded-family recall diagnostic is within each source capture. External evaluations use existing audited unstratified target panels, without cross-dataset semantic family harmonisation. Family removal changes training and calibration size/composition and may change thresholds; there is no size-matched ablation to isolate semantic novelty. Historical stability warnings are not assigned to refitted detectors, so no family-specific warning validation is claimed. No independent fifth dataset or production validation is added.

The archived report excludes model joblibs and per-flow prediction Parquets retained in the author's Drive. Verification therefore checks saved evidence and arithmetic, not independent replay of fitting, raw row exclusions, preprocessing, calibration or probabilities. Hash agreement is not independent historical certification of execution.

## Repeating the actual experiment

The unchanged Colab notebook records the required pinned source partitions/models, recovered metadata checkpoints and audited target panels. It uses the original estimator versions and resumes matching fit/evaluation checkpoints in the separate supplementary_source_family_holdout_20261025/run_v1 stage. A new machine needs these external Drive prerequisites. The notebook is not a one-command raw-data reproduction package. Use the saved-report verification command above to inspect results without new fits.

The inputs/ archives and planned/ source/context are preserved unchanged. Public upload, a permanent archive DOI and reuse licensing remain author actions; this package does not perform them.
