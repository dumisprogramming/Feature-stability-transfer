# CTU-13 bounded Logistic Regression warning pilot

Prepared 27 October 2026, after the completed CTU inventory and partition qualification were reviewed. This is a separate post hoc supplementary analysis. Earlier study outcomes and all qualification labels are already known. It is not a confirmatory or newly target-blind experiment.

## Fixed scope and inputs

Use only the pinned qualification stage supplementary_ctu13_partition_qualification_20261027/run_v1. Its reviewed compact report passed 196 hashes and 65 scenario–seed records. Five sources (1, 2, 6, 9, 13), eight targets (1, 2, 3, 6, 8, 9, 12, 13), seeds 2027–2031 and 35 ordered non-self transfers are fixed by support gates, not model outcomes. This gives 25 source fits, 25 source tests and 175 external target evaluations. No new datasets, raw recovery or original-study rerun is required.

The engine verifies all pinned qualification text records and all 260 selected panel NPZs. Recompute selected-array role assignments, local and all-capture cross-role intersections, numeric and class counts, actual/pseudo environmental quotas, and per-panel arrays against the pinned qualification summaries. Unknowns are absent; do not recode Background as benign. Source-independent signature roles and the target-label-informed exclusion of 1,008 conflicting signatures are inherited. The selected cohort is class-balanced, protocol-conditioned and deduplicated; it is not natural production traffic. The original raw-recovery proof is not independently replayed by this stage.

Six float64 fields: protocol_is_tcp, destination_port, duration_seconds, total_packets, total_bytes, source_bytes. These differ from both original 49-feature and earlier S16–S18 directional representations. No identity, timestamp or label becomes a predictor.

## Classifier and threshold

One model family only: StandardScaler fitted on the source training panel, then LogisticRegression(solver=lbfgs, C=1, tol=1e-8, max_iter=5000, fit_intercept=True), default L2 regularization, no class weighting or feature selection. All six columns retained, including source-constant columns. CPU, one BLAS/OpenMP thread, float64, no hyperparameter search and no target adaptation. Any ConvergenceWarning or nonfinite fitted output stops that unit; no automatic solver or cap change.

Choose a single source-calibration threshold maximizing balanced accuracy under prediction p >= threshold; ties within absolute 1e-12 use the largest threshold. Include the all-negative threshold nextafter(max calibration probability, +infinity). Save candidate thresholds and calibration confusion counts. Freeze the same threshold for source test, every permutation score and every external target. Calibration labels are used for this source-only choice; calibration BA is selection-optimistic and not a held-out accuracy claim.

Save source-trained scaler/coefficient arrays, calibration probabilities and source/target test probabilities. No pickled estimators are required. Independent forward prediction from the saved coefficients must agree within absolute 1e-12 with the fitted pipeline; no model is retrained on resume.

## Warning definitions

For every actual and matched pseudo environment group (100 benign + 100 attack), compute baseline BA. For each of six features, shuffle that column across the whole group, without stratifying by class, five times; importance is baseline BA minus permuted BA. Other columns and labels stay fixed. Keep signed drops; do not clip negative values. Use a deterministic seed from source, seed, group ordinal, feature and repeat, shared between actual/pseudo types as a common-random-number design. These are five permutation repeats within one fixed grouping, not five new independent environments.

Within each type, average the five drops for each group/feature. Spearman correlation uses midranks of the complete six-feature mean-importance vectors for every unordered pair of groups, including ties. Environmental instability I_env = 1 - mean(actual pairwise rho); sampling instability I_sampling = 1 - mean(pseudo pairwise rho); gap = I_env - I_sampling. Instability ranges from 0 to 2; the gap ranges from -2 to 2. Equal importance for all six features makes the rank vector constant and rho undefined. Save undefined pairs and mark the corresponding warning missing unless EVERY pair is finite; do not turn undefined rho into zero or silently change the estimand. No source, group, or feature is dropped to improve a warning result. Matched pseudo groups reuse exactly the same warning rows with the qualified class quota. Unconditional single-feature shuffles can break dependence and physical constraints between counters (for example source_bytes <= total_bytes); these importances measure this specified perturbation, not a causal intervention or an operational traffic change.

Checkpoint each source fit, each actual/pseudo importance group and each test panel, then save pairwise and warning summaries. Display progress and preserve all completed units after interruption.

## Outcomes and association summaries

BA = (TPR + TNR)/2 and loss = source-test BA - target BA. Save confusion counts, BA, precision, recall and F1; precision/F1 are cohort-specific. Original six-hour NF-v3 results remain unchanged. Report every directed case and seed. Average each source/seed's seven losses to form 25 descriptive source units. Also average over seeds to five source-capture units before reporting exploratory Spearman associations for sampling instability, environmental instability and the gap. Per-seed five-source associations are descriptive only. A constant predictor/outcome or fewer than three finite captures gives an explicitly missing rho. Do not report naive flow-level, 175-case or 25-seed-unit p-values, confidence intervals or power claims.

The all-direction results include related-capture and same-specimen cases. A subgroup table explicitly lists provider-recorded same-binary links (1/2/9; 3/4/10/11; 5/13); distinct hashes/groups do not certify different families, independent institutions or equivalent traffic.

## Strict predictive warning check (exploratory)

Two separate outer split schemes: five source-capture folds, and three documented-specimen source groups {1,2,9}, {6}, {13}. The latter grouping uses provider evidence, not independently authenticated executables, and has only three effective folds.

For each held-out set H, remove all source units in H from fitting AND remove every target in H when computing training-source outcomes. Training outcome for source s/seed is its mean transfer loss over the remaining eligible targets excluding self. Held-out outcome averages that source's seven fixed external targets. Every seed from a held-out source stays in the same fold. No test-source label, warning or loss is used to select training scaler, regression coefficients, ridge penalty or specifications.

Fixed specifications: intercept-only training-mean baseline; source-test error alone; environmental instability alone; gap alone; both environmental instability and gap. Use StandardScaler fitted on training units then Ridge(alpha=1) with intercept, no tuning. A model-identity predictor is constant in this single-classifier pilot and equals the intercept baseline. This is an exploratory warning prediction check, not a new detector or an exact replication of the original model-only comparison.

For fair comparisons, use the same complete-case source/seed units with finite environment and gap warnings in ALL five specifications. If a warning is missing, retain it in the descriptive tables but exclude that unit from all prediction specifications and list the reason. A fold needs at least two distinct training source captures and one held-out unit; otherwise record unavailable without relaxing the rule. Report per-held-out-source MAE and RMSE, macro-averaged across available source captures, predictions, fitting provenance and excluded target IDs. Do not infer general utility from five sources/three specimen folds. Overlapping outcomes, shared backgrounds, same-role signatures and repeated captures remain dependent.

## Persistence, environment and reports

New folder: Paper3_DataSet/supplementary_ctu13_warning_pilot_20261027/run_v1. No writes to qualification or inventory folders. A standalone isolated venv avoids importing mixed NumPy versions in the notebook kernel. Fixed tested package versions: numpy 2.3.5, scipy 1.17.0, scikit-learn 1.8.0, joblib 1.5.3, threadpoolctl 3.6.0. Installation requires normal package-download access but does not download datasets. Exact Python/package/runtime versions are recorded and locked for resume. If a Colab Python upgrade changes identity, do not edit the lock; inspect the report before choosing a fresh run.

Source fit, importance group, test-panel and final-analysis checkpoints use exact code/plan/context/runtime identities and SHA-256 outputs. Same unchanged notebook verifies and restores completed checkpoints. A partially written unit can rerun; committed fits are not repeated. Hashes ensure consistency with the records, not independent timestamping or proof of execution.

Compact report includes code/plan/context, input audit, numerical fit records, calibration and evaluation probabilities/labels, all importance drops, pairwise tables, warnings, outcomes, exploratory prediction checks, markers and checksums. It omits input feature panels. These model-result arrays support arithmetic and saved-coefficient prediction checks during execution; an external compact-report review cannot independently rerun raw recovery or feature-based predictions without the omitted inputs. No raw IPs or endpoints are included. Treat model/prediction outputs as research artifacts and review before public release.

Completion CTU13_WARNING_PILOT_COMPLETE means computations and integrity checks completed, not that warnings worked or all broader reviewer requests are closed. Record scientific warning/prediction availability separately. Errors produce CTU13_WARNING_PILOT_PARTIAL_RESUMABLE and a diagnostic ZIP where possible. One institution, five support-qualified sources, a restricted representation and label-informed cohort cannot establish broader independent-site validation, measured production prevalence, universal fine-family equivalence, adequate power or general warning utility.

Runtime for the full CTU pilot has not been measured. Local synthetic tests verify plumbing, numerical metrics, threshold selection, ranking, strict fold exclusions, resume and corruption handling; they are not scientific CTU results. Allow approximately 1–2 GB extra Drive space and 1–2 GB temporary runtime space for the venv, cache and artifacts; actual output use is expected smaller but not measured yet.

## Attribution and implementation references

CTU-13: https://www.stratosphereips.org/datasets-ctu13 ; Garcia et al. Computers & Security 45 (2014), 100–123, doi:10.1016/j.cose.2014.05.011. Data terms remain the provider's; no new dataset or software license is assigned by this package.
Scikit-learn logistic regression: https://scikit-learn.org/1.8/modules/generated/sklearn.linear_model.LogisticRegression.html
Permutation-importance definition: https://scikit-learn.org/1.8/modules/permutation_importance.html
