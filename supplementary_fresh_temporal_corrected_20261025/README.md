# Corrected fresh temporal source sampling sensitivity

Use the final report for the findings in Supplementary S12. The original_superseded_temporal_report.zip contains a known precision-exclusion error and is retained ONLY for provenance, never as valid temporal evidence. previous_corrected_report.zip contains the validated corrected sample checkpoints and a partial importance run stopped at a baseline-reuse gate. final_report.zip contains all 4,800 references, 240 warning units, 80 corrected sample summaries, 60 source replay records and eight strict regression analyses.

## Result-level replay

Run `python -m pip install -r requirements.txt`, followed by `python reproduce.py` here. The command verifies all 4,930 final hashes, original checkpoint provenance, warning aggregation and 1,152 regression fits /14,400 predictions. It writes to a new timestamped recomputed directory and does not overwrite archived reports. Exit code0 means all selected checks passed; an exception requires investigation. This code was run during preparation. It does not fit detectors, resample raw flows or repeat permutation predictions. The historical EXPERIMENT_CODE.py inside each report is provenance and uses Colab/Drive paths; it is not the portable entry point.

## Interpretation

Using ORIGINAL BALANCED target outcomes, model-only MAE is0.050100, and model plus both warnings gives0.074246,0.069710,0.048550,0.066369 at3/6/12/24h. The small12h gain changes sign across seed deletions and improves only two held-out dataset folds. No new significance, equivalence, optimal-width or deployment-readiness claim follows. All analyses are post hoc on the same four captures. Do not describe this as universal warning failure.

The separate precision-matched audit completed on 25 October 2026 checked all 60 saved unstratified target panels against all four corresponding source partitions. It found zero float32-signature overlap and identical panel selection under the corrected rule. No target predictions or dependent regressions required recalculation for this precision issue. The unstratified branches are therefore no longer provisional for this concern. See ../supplementary_unstratified_precision_audit_20261025/ for the unchanged audit report, result-level checks and limits. This does not establish independent new datasets or production validity.

## Limits

Corrected sample summaries report zero source raw-row and float32-signature overlap. Sample Parquets, raw datasets, trained models and per-flow permutation predictions are excluded and remain external prerequisites, so this package cannot independently certify those raw checks. Reuse provenance was checked against the preserved earlier records, including3,728 legacy reuses and1,072 non-legacy calculations. The1,293 imported corrected checkpoints overlap those categories. One baseline mismatch of0.00025 was recorded and the group recalculated; its cause was not established.

## Publication and rights

This preparation does not upload to GitHub, create a DOI, grant a new license or imply acceptance. Preserve originals and choose reuse rights consistently with the parent repository. Upload this folder before claiming public availability. No journal metadata is invented.
