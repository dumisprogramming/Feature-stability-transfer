# Raw family provenance and single family signature sensitivity

Verified on 27 October 2026. This folder supports Supplement S18 and preserves S16–S17 unchanged. Both original result ZIPs are included byte for byte.

## Findings and scope

The raw-occurrence audit identified 444 mixed exact-family signatures among 117,000 selected patterns, without within-capture binary conflicts among these patterns. The separate sensitivity removed all 444 from all roles, then completed nine LR fits, 132 importance groups, 2,376 scores and 18 evaluations. See `sensitivity_results.csv` and the unchanged reports for all outcomes.

Target labels informed the post hoc cohort restriction after prior outcomes were known. Retained signatures have one exact recorded family in their own audited capture, not certified label truth or cross-capture semantic equivalence. One pair, model and seed cannot validate general warning prediction. Production prevalence was not measured.

## Verify saved results locally

Use Python with NumPy and SciPy. From this folder:

```bash
python -m zipfile -e family_provenance_report_20261026_155911_486150.zip provenance
python -m zipfile -e pure_family_sensitivity_report_20261026_235323_741929.zip sensitivity
python verify_provenance.py provenance
python verify_sensitivity.py sensitivity . provenance RECHECK.json
```

The provenance check reaggregates 117,000 saved pattern records from per-file/member summaries. The sensitivity check verifies recorded hashes, prepared code/context, checkpoint details, reconstructed purity masks/counts, all importance means and Spearman pairs, count-derived metrics, family recalls, contrasts, class-size matching and agreement of new references. Omitted models, actual row arrays and per-row predictions are not independently replayed. The original raw files are not rescanned by these checks.

## Historical execution

The included notebook and engine preserve the completed experiment's configuration. A historical rerun requires pinned parent NPZ panels and the original provenance ZIP in Drive; these prerequisites are not supplied here. Do not overwrite frozen original folders or describe a new run as having occurred before prior outcomes were known. No new Colab run is needed to report this completed sensitivity.

## Upload and archive

Extract the outer support ZIP and add this entire folder to the repository, retaining both embedded report ZIPs. GitHub stores ZIPs without unpacking them. This folder contains fewer than 100 files and is separate from existing S1–S17 materials. Uploading the outer ZIP alone also preserves an archive but does not expose this README or CSV as browsable repository files. Verify the public upload before claiming these materials are available. No upload, DOI or reuse license is created by this package.
