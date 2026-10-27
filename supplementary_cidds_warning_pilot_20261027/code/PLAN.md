# CIDDS source-stability and transfer pilot — fixed plan, 27 October 2026

## Status and purpose

This is a supplementary, post-hoc, descriptive extension using existing qualified CIDDS partitions. No original confirmatory protocol, model, result or manuscript is overwritten. It tests whether the source-stability and transfer-loss calculations can be reproduced in an additional collection setting. It cannot establish general warning utility from only two eligible related source periods.

Inputs are exactly the twenty NPZ files referenced in the reviewed `cidds_partition_qualification_report_20261027_100813_428270.zip`. Two CIDDS-001 internal weeks qualified as sources for all seeds 2027–2031. Both CIDDS-002 weeks qualified as targets but failed the fixed source requirement of four eligible environment windows (each had three). Those failures are retained; no threshold or grouping is relaxed.

There are ten source fits (2 sources × 5 seeds), ten source-test evaluations and thirty external evaluations (3 non-self targets per source fit). Six ordered capture transfers each have five seeds. The four periods are from one Coburg collection setting. Seeds are repetitions; they do not increase the independent source count to ten. Same-role feature overlap across periods creates further dependence.

## Important cohort limitations

The preceding stage sampled at most 1,000 distinct model-input groups per class and 30-minute block, then excluded 4,874 selected SHA-256 groups with conflicting binary labels across the four candidate captures. That exclusion uses potential target labels. It is **target-label-informed supplementary cohort construction**, not prospective label-blind validation. Unknown labels were excluded, and frequency-based duplicate multiplicity was removed within each capture. The cohort is not representative of naturally observed traffic frequencies or production attack prevalence.

All four candidate predictors are fixed: supplied numeric duration (`duration_native`, without conversion), TCP indicator, destination port, and unidirectional packet count. This is a four-feature, one-classifier extension. It does not reproduce the original 49-feature, three-classifier NetFlow-v3 experiment or the six-feature Argus CTU pilot. No HIKARI features are harmonised or used. No fine-family semantic equivalence is asserted.

Training and environmental inputs are disjoint by candidate feature signature. Their clock periods can overlap; these are source-environment groups, not a prospective future-time holdout. Naive-clock timestamps and earliest retained representatives do not certify timezone or physical flow identity.

## Input verification before fitting

Verify the embedded report manifest and exact partition/projection code hashes. Rehash each Drive NPZ against its recorded byte count and SHA-256, then copy it to disposable Colab storage and verify again. Load only with allow_pickle=False. Recompute every selected feature signature; verify four-column finite arrays, class counts, role uniqueness, environmental and pseudo group class sizes, and matching environment/pseudo signature pools.

Recompute seed-dependent global role assignments and cross-capture intersections from the actual NPZ arrays. Any different-role shared signature, remaining selected binary-label conflict, changed input, or eligibility discrepancy blocks fitting. All four test panels are checked for integrity, including their label arrays; the process must not claim zero target-label access before fitting. Target labels do not enter preprocessing, model fitting, calibration, feature importance, or warning definitions.

## Fixed model and calibration

For each eligible source and seed:
1. Fit StandardScaler on that source's training predictors only. Retain its exact mean and scale arrays.
2. Fit LogisticRegression with C=1, solver=lbfgs, tol=1e-8, max_iter=5000, fit_intercept=True, class_weight=None and the given seed. Use a one-thread numerical limit. No tuning, target adaptation or extra models are allowed. A convergence warning blocks completion rather than triggering unplanned changes.
3. Compute attack probabilities on the source calibration panel. Select the threshold maximizing balanced accuracy, using predicted attack when probability >= threshold. Evaluate every distinct calibration probability and an all-negative threshold nextafter(max_probability,+infinity). Among scores tied within absolute 1e-12, choose the highest threshold. Do not adjust it on source-test or target outcomes.
4. Save numerical scaler/coefficient/intercept/threshold arrays without pickle. Check that independently reconstructed expit probabilities agree with sklearn on calibration data to absolute 1e-12. Retain calibration labels and probabilities for threshold-optimum verification. The source training panel itself is not included in the compact report, so a saved-results verifier cannot independently refit the model.

The package uses Colab's installed NumPy, SciPy, scikit-learn, pandas and threadpoolctl; it makes no installation or environment changes. Exact package/Python versions are recorded in the initial run identity and must match to resume. There is no claim that the originally prepared historical environment has been recreated.

## Source-only importance and warnings

Compute warnings before target performance evaluations in each source job, using the already materialised source environmental and matched pseudo groups. Do not refit models or thresholds in an importance group.

For each group, calculate baseline balanced accuracy, then permute each of the four predictor columns independently five times with fixed NumPy default_rng permutations derived from SHA-256 of source/seed/group-kind/group-index/repeat/feature tags. Importance is the mean reduction in balanced accuracy at the fixed source-calibrated threshold. Negative importance values remain valid.

Save baseline and all 5×4 permuted confusion-count vectors (TN, FP, FN, TP). To avoid floating rounding breaking mathematical importance ties, rank the exact integer importance numerators:

`numerator_j = sum_r [(TP_base - TP_permuted_rj) * n_benign + (TN_base - TN_permuted_rj) * n_attack]`

The common group denominator is `2 * n_benign * n_attack * 5`. This yields the mean balanced-accuracy importance exactly up to final float conversion. Average-rank Spearman correlations use the four-feature numerator vectors.

Within each group kind, calculate every unordered pair of group importance-rank vectors. Stability is their mean Spearman correlation and instability is one minus that mean. A constant rank vector makes its pair undefined. If any required pair is undefined, the corresponding instability is missing; never substitute zero or silently omit such pairs.

Environmental instability uses clock-based groups. Sampling instability uses the matched pseudo groups from the same selected source pool. The sampling–environment gap is environmental instability minus sampling instability; it is missing if either component is missing. Keep all pair coefficients, undefined counts and group sizes. Four predictors and a linear model make this a narrow sensitivity check, not universal validation of feature-importance warning methods.

## Evaluation and reporting

Evaluate the fixed source model and threshold on its own source-test panel and on each of the three other qualified target-test panels. Do not refit or recalibrate for target data. Save labels and probabilities, confusion counts, balanced accuracy, false-positive rate and attack recall. Transfer loss is that seed's source-test balanced accuracy minus target balanced accuracy; retain negative loss (improvement).

Report all six directions, source-period means across five seeds, and overall unweighted job means. Weighting is at the completed job level, not pooled raw-flow level. Test panels have capped class counts; balanced accuracy does not imply observed operational prevalence. Keep source/target IDs and seed in every result.

**Do not fit warning-utility regressions, calculate correlation p-values, or claim a general association from two source periods.** No model-identity comparison is meaningful with one classifier. No strict LOSO meta-prediction is advertised. The broader requests for independent-site warning utility, measured production prevalence and certified cross-capture fine-family equivalence remain open.

## Checkpoints, report and independent arithmetic verification

Use the new Drive stage `supplementary_cidds_warning_pilot_20261027/run_v1`. The identity binds executed code, the fixed plan, input report, verifier, and package versions. Save a complete marker with exact artifact hashes after each fit, source-importance group and evaluation. Resume only completed, matching markers; refuse changed artifacts. An interrupted unsealed job may be recomputed.

The compact ZIP includes the report chain, executed code/plan, numerical models, calibration/test labels and probabilities, source-importance confusion counts, rank-pair results, input references and aggregate summaries. It excludes raw predictors, per-row IPs/UIDs, candidate cohorts and original partition matrices. All data are public benchmark-derived, but the ZIP is for review before any GitHub upload.

The independent verifier rehashes report/checkpoint chains; recomputes calibration optima, forty metric sets, importance numerators, rank correlations, warning gaps and aggregate/directional arithmetic. It does not independently replay omitted training arrays, model fitting, original feature permutations, or raw dataset recovery. A verification PASS means the selected saved-result checks passed; inspect the separate pilot COMPLETE/PARTIAL status as well.
