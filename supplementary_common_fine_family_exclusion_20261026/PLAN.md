# Post hoc six field representative label family exclusion

Prepared 26 October 2026 after the completed CICIDS2017 pilot and family-support audit were known. This is a new supplementary experiment. It must not overwrite the original 49-feature experiments, the six-field parent panels, or S16 pilot results.

## Question and scope

On one CSE-CIC-IDS2018 source and one CICIDS2017 target, describe changes in candidate-family recall and source-only stability warnings when a named representative-label family is removed from training and calibration. Compare a full reference and a class-size-matched control. This is a bounded ablation of retained representative labels, not proof of semantically equivalent unseen attack families, a calibrated warning, open-set recognition or production performance. No new detection method is proposed.

Three candidates pass the recorded minimum of 50 family representatives in training, calibration, source test and target: GoldenEye (380/86/96/1038), Hulk (1102/208/224/16077), SSH Patator (867/174/160/285). Selection uses the supplied support counts. Slowloris, Slowhttptest and FTP Patator are excluded for insufficient source support. Earlier family-specific target outcomes are already known; no fresh target-blind selection or confirmatory status is claimed.

The provider-supported named-tool correspondence is a candidate mapping, not verification of identical generator settings or labels. Fine-family mixtures among all raw occurrences sharing a six-field pattern have not been recovered. Counts and exclusions therefore concern the first representative label in the frozen panels. Do not describe them as removal of every raw occurrence or absence of all related attack mechanisms.

## Fixed inputs and signatures

Reuse the exact five pinned NPZs from `supplementary_common_representation_partition_20261026`. Ordered features are destination port, duration in microseconds, forward/backward packet counts and forward/backward byte counts. Inputs and preprocessing stay float64; six integer-valued fields form an exact 48-byte little-endian int64 signature. Verify parent identity, manifest, all five file hashes, source row-location separation and all ten panel-pair signature intersections. Every file remains unchanged.

This retains global binary-conflict removal, signature deduplication, first-representative selection and class-balanced target sampling from the parent pilot. These restrictions materially change the flow population. Primary target exclusion covers selected source panels, not all unselected source patterns. Historical exporter settings and byte accounting are not verified equivalent. The original 49-feature results cannot be pooled with these results.

## Nine fits and training controls

Use one logistic-regression classifier and seed 2027. Fit all three conditions separately for each of the three candidates, producing nine fits even though full-reference training rows are identical across candidates.

1. Full reference: all 20,000 training and 4,000 calibration rows.
2. Family excluded: remove the candidate's representative labels from both training and calibration. Keep every other original row.
3. Size matched control: sample without replacement from the full training/calibration roles, separately by binary class, to match condition 2's exact class counts. Use deterministic family/role/class seeds, one draw and original order. This control can retain candidate-family labels, whose counts must be reported. It does not match every finer characteristic or isolate a causal semantic novelty effect.

For GoldenEye, exclusion removes 380 training and 86 calibration representatives. Hulk removes 1,102 and 208; SSH Patator removes 867 and 174. Benign counts remain 10,000 in training and 2,000 in calibration. Require at least 50 retained examples per binary class in both roles. Persist the selected row positions locally and in Drive.

Fit a median imputer and StandardScaler only on each condition's retained training rows. Imputation is a no-op for the complete-case features. LR uses C=1, L2, lbfgs, max_iter=3000, random_state=2027, other installed defaults, one numerical thread. A convergence warning blocks execution without automatic retuning. Source calibration selects the maximum-BA finite threshold in [0,1] from the complete ROC threshold list; ties within absolute 1e-12 prefer closest to 0.5, then ROC order. Do not use test or target outcomes for threshold selection, model choice, feature selection or exclusion size.

The full reference must reproduce the earlier pilot calibration threshold and source/target confusion counts. A mismatch blocks completion; do not adjust the fit to force agreement. The earlier outcomes are explicitly known reference checks.

## Matched source environments and warnings

For each candidate, remove its representative labels from the environmental panel for **all** three training conditions. Retain whole six-hour blocks that still have at least 500 rows in each class. The audit gives eight eligible blocks per selected candidate. Deterministically retain 500 per class per block; in this saved panel, each eligible class already has exactly 500. The resulting 8,000-row panel and eight class-matched pseudo-groups are identical across that candidate's three conditions. These are subsets of the nine original six-field environments. Different candidates omit different windows and therefore must not be interpreted as identical perturbation designs.

Recompute permutation importance separately for every fit. Three repeats over six transformed predictors yield 18 permutation scores per group. Eight actual and eight pseudo-groups per condition give 144 total group calculations and 2,592 permutation scores. Within a family/group/feature/repeat, permutation draws are shared across conditions. For each partition, average pairwise Spearman rank correlations over 28 pairs, using average ranks and zero correlation for a constant-ranked pair (count such pairs). Environmental instability is 1−S_env; sampling instability is 1−S_sampling; gap is S_sampling−S_env.

Store each fit and group with code/input/software identities and hashes. Seal each source warning record, then a global record of all nine source fits/warnings, before the experiment's new target scoring. Prior target results are already known, so this ordering is a within-stage execution control rather than a claim of historical blindness.

## Evaluation and fixed reporting

Keep all 4,000 source-test and 80,000 target representatives unchanged. Apply each fit's source-only imputation, scaling and threshold. Complete 18 panel evaluations. The main descriptive endpoint is target recall of the corresponding mapped representative family; record its numerator, denominator and miss rate. Source-test candidate-family recall is a paired diagnostic. Whole-panel BA, confusion counts, benign and attack recall, attack precision, macro F1 and MCC are secondary. Zero-denominator precision/F1/MCC follow the pilot's zero rule. Do not invert target decisions, clip negative BA losses or retune after seeing results.

For every family, report family-excluded minus full-reference and family-excluded minus size-matched-control changes in source-family recall, target-family recall, target BA and all three warning scores. Retain all nine condition records and six descriptive contrasts, including favourable, adverse and null changes. Do not select a family or warning by target performance. No pooled significance tests, warning meta-regression, confidence intervals, superiority, warning–loss correlation or predictive-utility claim is planned. Nine fits and reused test rows are not nine independent captures; three family cases cannot validate general warning value.

## Resume and outputs

New Drive stage: `Paper3_DataSet/supplementary_common_fine_family_exclusion_20261026/run_v1/`.

Checkpoint after each fit, each importance group, each source warning seal, the global source seal and each evaluation. A resume verifies exact code, plan, configuration, software and artifact hashes, and reuses only verified records. Keep the notebook unchanged. If the session disconnects, reconnect, remount Drive and run the same notebook again. Do not delete checkpoints or bypass a hash/version mismatch. No dependency installation is performed, so no restart is requested by this notebook.

The compact downloadable report includes source code, plan, context, identities, fit details, all group importance arrays, warning seals, aggregate metrics, paired contrasts and checksums. Fitted models, selected-row arrays, group assignments and per-row predictions stay in Drive and are omitted from the report. Hash/arithmetic checks of the ZIP will not independently replay raw fitting or scoring. Only `COMMON_FAMILY_EXCLUSION_COMPLETE` marks this bounded experiment complete; `COMMON_FAMILY_EXCLUSION_BLOCKED_RESUMABLE` is a diagnostic report. Broader independent-capture warning validation, full family semantics and measured production prevalence remain separate unresolved scope.
