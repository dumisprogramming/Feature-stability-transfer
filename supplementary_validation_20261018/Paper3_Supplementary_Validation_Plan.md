# Paper 3: feasibility assessment and supplementary analysis plan

**Study:** When Source-Side Feature Stability Fails to Warn of Cross-Dataset Network Intrusion Detection Degradation  
**Plan date:** 18 October 2026 (Asia/Dhaka)  
**Status:** Specified after knowledge of the original results, before calculation of the new analyses below. This is a supplementary robustness plan, not a preregistration, an amendment frozen before original label access, or an independent replication.

## 1. Feasibility findings

The archived repository ZIP, version 1, contains the required transfer-level inputs. Structural inspection confirmed:

- Exactly 180 unique source–target–seed–model rows: all 12 directed transfers × five seeds × three models.
- Four datasets, seeds 2027–2031, and LogisticRegression, RandomForest and XGBoost.
- Finite values for loss, source and target balanced accuracy, and the three warning variables.
- Each warning is constant across the three targets within a source–seed–model unit, consistent with its source-only definition.
- All four proposed strict fold partitions have the expected counts.

Relevant inputs are `Paper3_Final_Evaluation/transfer_analysis_all_180.csv`, the original A3 plan, and `tools/analysis_core.py`. The existing 60-row summary averages outcomes over all three targets; it cannot be reused as the strict training outcome. The outcome must be rebuilt from transfer-level rows inside each fold.

| Quantity in each strict fold | Count |
|---|---:|
| Datasets permitted in meta-regression training | 3 |
| Directed training transfers among these datasets | 6 |
| Training transfer rows | 90 |
| Aggregated source–seed–model training units | 45 |
| Targets averaged per training unit | 2 |
| Held-out source transfer rows | 45 |
| Aggregated held-out source–seed–model test units | 15 |
| Targets averaged per test unit | 3 |

No new regression, correlation, prediction error or significance result was calculated during this feasibility assessment. Archived result values were accessed for schema and finite-value checks; the original outcomes were already known.

## 2. Question and interpretation

Does a source-only stability warning improve prediction of mean cross-dataset balanced-accuracy loss for a source dataset absent from every role in meta-regression training?

“Absent from every role” refers to the degradation-prediction meta-regression. The underlying detector and its warning are still trained/computed using the held-out dataset's own source partitions, as intended in a source-only warning task. This is not a detector trained without that source dataset, an unseen-target validation, or a validation of an entirely new dataset collection.

## 3. Strict leave-one-dataset-out procedure

For each held-out dataset h:

1. Select training transfers only when both source ≠ h and target ≠ h. Assert h is absent from both training roles.
2. Within each remaining source–seed–model combination, average balanced-accuracy loss over its two permitted targets. Retain the source-only warning variables. This produces 45 training units.
3. Select test transfers with source = h and target ≠ h. Average loss across the three targets within each seed–model combination, producing 15 test units.
4. Fit the following six fixed specifications, matching the original choices: mean-only; model-only; model plus sampling instability; model plus environmental instability; model plus sampling–environment gap; model plus both primary warnings.
5. Use ordinary least squares with intercept, LogisticRegression as the model-indicator reference, and training-fold-only mean/population-standard-deviation scaling. Replace a zero scaling standard deviation with 1, as in the original code. Do not tune, select specifications, clip predictions, or change thresholds.
6. Save all 15 predictions for every specification, including poor or out-of-range predictions. Retain fold identity.

Use the original warning definitions and loss = source-test BA − target BA, without clipping. Check numerical design rank and condition number; report deficiencies or instability rather than choosing a replacement estimator after seeing results. Abort on missing/duplicate keys, inconsistent warnings, unexpected fold counts, or non-finite required data.

Training outcomes average two targets and test outcomes average three. This difference is a consequence of excluding h and must be reported. The remaining three datasets occur as both training sources/targets and test targets. Thus this stricter exclusion removes h from training, but does not remove every shared-data dependency.

## 4. Fixed comparisons and reporting

Primary supplementary comparison: combined primary warnings plus model identity versus model identity alone.

Primary score: mean absolute error (MAE). Report ΔMAE = MAE(model-only) − MAE(candidate); positive values indicate improvement. Report the combined model against the original model-plus-sampling-instability comparator as a secondary comparison. Retain all six specifications in every output.

Report MAE and RMSE for each of the four folds and overall. Overall MAE is the equal-weight mean of the four fold MAEs (also pooled MAE because folds are equal in size). Overall RMSE is the square root of pooled mean squared error. Report overall Spearman correlation descriptively; undefined correlations remain missing, not zero. Report all four paired fold improvements and the count positive/zero/negative, without treating folds as independent trials. Also report errors by model family.

Do not infer success from a favorable subset. If the combined model has lower overall MAE, describe the improvement for these benchmarks together with fold variation. Mixed fold signs must be explicit. Equal or worse error supports no improvement under this supplementary procedure, not universal absence of useful warnings.

## 5. Dependence sensitivity analyses

Keep the original A3 results and their stated assumptions unchanged. Do not replace them silently or describe 180 jobs as independent environments.

**Dataset-deletion association sensitivity:** For each h, remove every transfer involving h in either role. Rebuild the 45 source–seed–model units from the two remaining targets per source; average over five seeds to obtain nine source–model units. Recompute descriptive Spearman associations for environmental instability, sampling–environment gap and sampling instability against mean loss. Report every deletion result, the range and sign changes. Do not report deletion p-values or confidence intervals.

**Seed-deletion sensitivity:** For each of the five omitted seeds, rebuild the original association units from the other four seeds and recompute the three descriptive correlations. Rerun the strict validation above with four seeds: 36 training units and 12 test units per fold. Report the complete five sets of overall MAE differences and their range. This measures sensitivity to the sampled seeds; it is not additional independent dataset replication.

With only four datasets and overlapping folds, this plan does not claim a reliable population-level dependence correction. No job-level bootstrap, naive standard error, t-test over the four folds, or new significance claim will be used. Deletion ranges describe robustness, not confidence intervals. More independent datasets would be needed for stronger generalisation evidence.

## 6. Execution and provenance rules

- Preserve all historical data, code, seals and A3 files unchanged.
- Save supplementary scripts and outputs separately, for example under `supplementary_validation_20261018/`.
- Before execution, record this plan's SHA-256, input hashes, implementation hash and software versions. These are integrity records, not independent timestamp certificates.
- Verify the supplied artifact hashes before analysis. The structural inspection here is not a replacement for full artifact verification.
- Implement and review fold construction and preprocessing without displaying new performance results; then execute the fixed analysis.
- Record any deviation before the affected calculation; retain this version and publish the deviation and reason.
- Report all results regardless of direction. A full raw-data rerun, additional datasets, novelty literature review and manuscript rewriting are outside steps 1–2 and remain future work.

## 7. Audit record

Input ZIP SHA-256: `5b7009c4c159f5f7fc3e8d7aec8c56e5e3e9a3c38187936be7653095723573a8`

Transfer table SHA-256: `440ea72799ad3bf26ccb8e55d7896b2682ba8ec02deeb39a0190361d95599de6`

Structural checks: complete_unique_transfer_grid = PASS, required_numeric_fields_finite = PASS, warnings_constant_across_targets_within_source_seed_model = PASS, all_four_strict_fold_structures_pass = PASS.
