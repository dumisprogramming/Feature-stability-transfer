# Temporal grouping sensitivity materials

Post-hoc 12-/24-hour regrouping of original six-hour sampled source rows. Historical six-hour scores are retained as baseline. No detector retraining or new raw-flow sampling was performed. Read PLAN.md before interpreting the results.

The author completed the Colab run; its archive is retained unchanged as run_v1.zip. The results directory provides convenient copies of the same aggregate warnings, group inventory, prediction scores and diagnostics. verification_report.json records independent checks performed on 24 October 2026. Checkpoint hashes establish consistency, not independent historical timestamps or proof of execution.

Model-only pooled MAE is 0.05010. With both primary warnings it is 0.06690, 0.05885 and 0.05656 at 6, 12 and 24 hours. The combined warnings remain worse in every dataset fold and seed-deletion run. All specifications are retained. No new p values, equivalence or general impossibility claim is made.

CSE-CIC-IDS2018 has identical 12-/24-hour partitions and exactly equal warning scores. UNSW-NB15 has only two 24-hour groups. New grouping sizes, counts and historical-versus-new permutation states constrain interpretation. These are not independent new datasets or new raw-data temporal windows.

To inspect saved source importance calculations, extract run_v1.zip. Group checkpoint payloads are self-checksummed; group_inventory.csv maps each model/width/group to its checkpoint. Each importance array has 49 predictors and three repeats in the preserved schema order. The source-warning columns use one minus mean pairwise rank stability, with a constant-vector pair assigned zero correlation; gap is sampling stability minus environmental stability. See the full plan and code for exact definitions.

To rerun source importance, open the included notebook in Colab with the original Paper3 Drive files. Saved models, preprocessors, source-test predictions, environmental rows and original importance records are prerequisites. The notebook saves to a separate run folder and checks compatibility before use. Merely downloading this public package is insufficient to reconstruct the full raw-data pipeline. No raw flows, model binaries or per-flow target predictions are included here. No new reuse license is assigned.

For GitHub upload, keep run_v1.zip as ONE file inside this folder; do not extract its 1,212 members into the upload batch. Upload the whole supplementary_temporal_grouping_20261023 folder alongside the previous supplementary folders. This preparation step did not change GitHub.
