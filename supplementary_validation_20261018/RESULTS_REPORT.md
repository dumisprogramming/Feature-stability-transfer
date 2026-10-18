# Paper 3: supplementary validation results

**Completed:** 18 October 2026. **Status:** additional analyses planned after the original findings were known, before these new calculations. No preregistration or new independent raw-data experiment is claimed.

## Main finding

Excluding each held-out source dataset from both roles in meta-regression training did not improve the warning models' prediction of degradation. Model-only MAE was **0.050100**; model plus both primary warnings had MAE **0.066898**. Thus the combined warnings increased error by **0.016799**, approximately 1.68 balanced-accuracy percentage points. This is a prediction-error difference, not a reduction in detector accuracy.

The combined model was worse in all four dataset folds, all three model-family summaries, and all five seed-deletion runs. Each individual warning specification also had worse overall MAE than model-only. This supports the original finding of no predictive improvement within the evaluated setup; it does not prove that all source-only warnings are uninformative.

## Procedure

For held-out dataset h, training included only transfers whose source and target were both different from h. Training outcomes were rebuilt by averaging over the two permitted targets, yielding 45 source–seed–model units. Test outcomes used h as source and averaged over the other three targets, yielding 15 units. Four folds produced 60 predictions per specification. All six fixed OLS specifications were retained; scaling used training data only. No predictions were clipped, and no tuning was performed.

The held-out source's detector and warnings still used its own source data, as intended for source-only warning assessment. This is an exclusion from degradation-meta-regression training, not an unseen-target test. Test targets also occur in training, so some shared-data dependence remains.

## All specifications: strict validation

Positive MAE improvement means better than model-only; negative means worse.

| specification | mae | rmse | mae_improvement_vs_model_only |
| --- | --- | --- | --- |
| mean_only | 0.052884 | 0.063218 | -0.002784 |
| model_only | 0.050100 | 0.062685 | 0.000000 |
| model_plus_environmental_instability | 0.054891 | 0.067099 | -0.004791 |
| model_plus_primary_combined | 0.066898 | 0.083791 | -0.016799 |
| model_plus_sampling_environment_gap | 0.058306 | 0.070041 | -0.008206 |
| model_plus_sampling_instability | 0.062848 | 0.080561 | -0.012748 |

Absolute errors here should not be directly ranked against the original LOSO errors as if the tasks were identical: training outcome target sets changed. The intended comparison is between specifications within this stricter validation.

## Combined warnings versus model-only: every dataset fold

| held_out_dataset | mae | mae_improvement_vs_model_only |
| --- | --- | --- |
| NF_BoT_IoT_v3 | 0.049271 | -0.002842 |
| NF_CSE_CIC_IDS2018_v3 | 0.099823 | -0.014284 |
| NF_ToN_IoT_v3 | 0.044230 | -0.013623 |
| NF_UNSW_NB15_v3 | 0.074268 | -0.036446 |

## Combined warnings: every seed deletion

| run | mae | mae_improvement_vs_model_only |
| --- | --- | --- |
| omit_seed:2027 | 0.067891 | -0.020446 |
| omit_seed:2028 | 0.074446 | -0.020471 |
| omit_seed:2029 | 0.069227 | -0.017959 |
| omit_seed:2030 | 0.071766 | -0.020586 |
| omit_seed:2031 | 0.057571 | -0.010220 |

## Combined warnings: model-family summaries

| model | mae | mae_improvement_vs_model_only |
| --- | --- | --- |
| LogisticRegression | 0.074077 | -0.005227 |
| RandomForest | 0.063173 | -0.024457 |
| XGBoost | 0.063445 | -0.020711 |

## Association sensitivity

After deleting each dataset from both roles and rebuilding nine source–model units, environmental-instability correlations ranged from **0.050 to 0.500**. They remained positive, but varied appreciably. Sampling–environment-gap correlations ranged from **−0.367 to 0.383**, changing sign across deletions. Sampling-instability correlations ranged from **0.167 to 0.267**.

After deleting one seed at a time, environmental-instability correlations ranged from **0.238 to 0.378**; gap correlations ranged from **−0.112 to 0.119**. All exact values, including the sampling baseline, are provided in `results/association_sensitivity.csv`.

These are descriptive sensitivity results. No new p-values or confidence intervals were calculated. Positive environmental correlations must not be described as absent associations, or as statistically supported warning evidence on the basis of these deletion analyses.

## Verification and limits

- The archived inputs passed 200 of 200 artifact-integrity checks before execution.
- All 24 full-seed/seed-deletion fold partitions passed role-exclusion and count assertions.
- Outcome and warning perturbation checks confirmed separation of test data from training outcome construction and scaling.
- All 144 fitted designs had full column rank. The maximum reported condition number was approximately 10.78 for the scaled design including its intercept.
- Independent NumPy fits on unscaled covariates reproduced all 1,800 predictions with maximum absolute difference below 1.0 × 10⁻¹⁵. Score aggregation consistency checks also passed. This verifies the supplementary calculations, not the entire original pipeline.
- Only four datasets are represented, and the folds overlap. Four fold comparisons and repeated seeds do not establish independent population-level replication. Deletion ranges are not confidence intervals.
- The data remain binary benchmark transfers with fixed conventional models and no adaptation. No new raw-data detector training, additional dataset, or deployment evaluation was performed.
- The implementation followed the fixed plan; no analysis deviations were needed. Historical files were read, not edited.

## Suggested manuscript addition

A supplementary validation excluded each held-out source dataset from both source and target roles in degradation-meta-regression training. Training outcomes were recomputed over the two remaining target datasets per source, while held-out outcomes averaged the three eligible targets. Across four folds, model-only prediction achieved MAE 0.0501, compared with 0.0669 for model identity plus both primary warnings. The combined specification had higher MAE in every held-out dataset fold and in all five leave-one-seed-out analyses. Dataset-deletion association analyses yielded environmental-instability correlations of 0.050–0.500 and sampling–environment-gap correlations of −0.367–0.383. These additional analyses were specified after the original results were known and are reported as descriptive robustness evidence; they do not remove all shared-dataset dependence or establish generalisation beyond the evaluated benchmarks.

## Next research tasks

The stricter saved-result validation is now complete. A complete raw-data rerun remains unvalidated. A focused novelty comparison and, if feasible, additional independent datasets would address other concerns. The manuscript itself has not been edited in this task.
