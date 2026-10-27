# Own capture single family pattern sensitivity

Prepared 26 October 2026 after S16, S17 and the complete raw family-provenance audit were known. This is a separate post hoc sensitivity analysis. S17 remains unchanged as its original first-representative-label ablation. No original panel, model, prediction, protocol or result is overwritten.

## Why a sensitivity is warranted

The raw provenance scan reported 444 mixed exact-family signatures among 117,000 selected signatures: 240 training, 37 calibration, 52 source-test, 113 environmental and two target patterns. No selected pattern had a within-capture binary-label conflict. One retained training pattern also had a GoldenEye raw label (one matching raw occurrence), and one had SSH raw labels (three occurrences), despite different first-representative labels. Among 1,102 Hulk training representatives, 238 also had other attack-family labels. These are ambiguities under a six-field projection, not proof of label errors or that all matching raw flows were used in training. The first-label formulation was disclosed in S17.

The reviewed report passed 46 recorded file hashes, 19 checkpoint-detail digests and 22 checkpoint-output checks. All 117,000 pattern records, representative hit records and 15 candidate/role summaries were independently reaggregated from the returned per-input records. Source and target raw files were not independently reread in the preparation environment. The author's audit scanned all ten pinned source CSVs and eight target members in 410.86 seconds, under the original eligibility filters.

## Fixed purity restriction

Use the same five original NPZs and the exact reviewed provenance-report ZIP, with pinned hashes. Align every provenance ID with its panel role, original panel-row index, signature SHA-256, representative label, binary label and original file/row location. Verify the provenance index covers every selected row exactly once. The original parent panel hashes and cross-role signature separation are checked before restriction.

Keep a pattern only if its eligible raw occurrences within its OWN capture have exactly one exact family label, matching the representative, and no binary conflict. Drop all mixed patterns, irrespective of whether their removal improves performance. This is an exact raw-label consistency rule, not a semantic classifier or correctness adjudication. Cross-capture label differences are retained as domain differences rather than forced into agreement. The rule does not certify identical generators, equivalent fine-family definitions or absence of related mechanisms.

Apply the rule to source training, calibration, source test, environment and target. This uses previously known raw target labels to restrict the target cohort; it is explicitly label-informed and post hoc. Do not describe it as a new target-blind confirmatory test. Keep row order; do not replace dropped examples or rebalance classes.

Fixed retained panel counts are training 19,760 (10,000 benign / 9,760 attack), calibration 3,963 (2,000 / 1,963), source test 3,948 (2,000 / 1,948), environment 8,887 (4,500 / 4,387), and target 79,998 (40,000 / 39,998). Removed original panel-row indices are fixed in context.json and saved in purity_selection.json. Fitting/environment selection indices refer to the resulting purified panel order. No new raw scan is needed.

## Candidate selection and nine fits

Retain the same three candidates, classifier and seed as S17. GoldenEye support is 380/86/96/1038 in training/calibration/source test/target; Hulk support is 864/172/172/16076; SSH Patator support is 867/174/160/285. All remain above 50 in each required role. Do not add, remove or tune candidates after scoring.

For each candidate separately fit full_reference on the entire purified training/calibration roles; family_excluded removes that candidate from both purified fitting roles; size_matched_control samples from the purified full fitting roles separately by binary class to match excluded counts, using one deterministic family/role/class draw. The control can retain the candidate family. It does not match every finer distribution or isolate a causal semantic effect. There are nine new LR fits, not independent captures.

A median imputer and StandardScaler are fitted only on each retained training set. LR uses C=1, L2, lbfgs, max_iter=3000, random_state=2027 and installed defaults otherwise, with one numerical thread. Stop on a convergence warning rather than retune. Thresholds maximize source-calibration BA over complete finite ROC thresholds, with absolute 1e-12 ties closest to 0.5 and then ROC order. No source-test/target outcome tunes parameters or thresholds. Cohort construction uses known labels as disclosed above.

The three new full references must agree with each other in calibration threshold and source/target confusion counts, because they have identical purified fitting/evaluation rows. They are NOT required to reproduce S16/S17's original threshold or scores: purification changes those inputs. Original results are retained for context, not overwritten or forced to match.

## Recomputed warnings

For each candidate remove its representative labels from the already purified environmental panel for ALL three fitting conditions. Retain whole six-hour groups having at least 500 rows in each class. Keep the existing quota; do not lower it to restore old window counts. GoldenEye has seven eligible groups, Hulk eight and SSH seven. All conditions within a candidate use the same environmental rows, class-matched pseudo-groups and permutation draws. Different candidates omit different windows; this design does not isolate changes in time-group membership.

Recompute three-repeat permutation importance across six predictors for every fit. Total work is 132 group calculations and 2,376 permutation scores. Average pairwise Spearman correlation uses 21 pairs for seven-group partitions or 28 for eight-group partitions, average ranks and zero correlation for constant-ranked pairs, whose counts are retained. Environmental instability, sampling instability and gap retain the earlier definitions. All nine source-warning records and a global source seal precede new target scoring, after all earlier outcomes were already known.

## Outcomes and reporting

Score each fit on the fixed purified source-test and target panels: 18 evaluations. The primary descriptive endpoint is target candidate-family recall, with counts and miss rate. Source candidate recall is diagnostic. Whole-panel BA, confusion counts, class recalls, attack precision, macro F1 and MCC are secondary. Retain negative BA losses and apply the prior zero-denominator rules. Do not invert predictions or tune thresholds after scoring.

Report all nine family/condition records and six exclusion-minus-comparator contrasts in source candidate recall, target candidate recall, target BA and each warning. Every condition within a candidate shares the same newly purified test rows, supporting the paired descriptive contrast. Raw comparison of aggregate scores against S17 changes source/test samples and environmental windows and is not a pure model-effect estimate. Do not infer general warning utility, correlation, equivalence, superiority or significance from three correlated cases. No p values, confidence intervals or warning meta-regression are planned. Classifier, seed, quota, warning formulas and candidates are fixed.

After exact single-label purification and representative-family exclusion, retained fitting signatures have no matching candidate-labelled occurrence in the audited eligible source files. This statement is restricted to those signatures/files/labels; it does not establish absence of related attacks, label ground truth or verified cross-capture semantic equivalence.

## Checkpoints and scope

Write only under `Paper3_DataSet/supplementary_pure_family_sensitivity_20261026/run_v1/`. Store code/plan/context/software identity, purity selection, fits, importance groups, source seals and evaluations. Resume only verified, identical checkpoints. Keep the notebook unchanged; reconnect and Run all after a disconnect. No dependency installation is performed.

The compact report contains the purity-removal map and hashes, code, plan, context, fit details, all importance arrays, warning seals, count metrics and descriptive contrasts. Model binaries, row-selection arrays and per-row predictions remain in Drive and are omitted from the report. Verification of returned hashes/arithmetic is not an independent raw-fit/probability replay. Only PURE_FAMILY_SENSITIVITY_COMPLETE marks this planned run complete. A blocked-resumable ZIP is diagnostic. The real sensitivity has not run in preparation; synthetic tests do not provide a Drive runtime estimate.

This sensitivity addresses the discovered projection-level family ambiguity within the bounded experiment. It does not complete broader independent-capture warning validation, semantic certification or measured production prevalence. No new public upload, persistent DOI or coauthor approval is asserted.
