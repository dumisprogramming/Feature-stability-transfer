# Remaining files for a full experimental release

The original source-stage implementation is now present in `notebooks/Paper3_CompleteCode.ipynb`; cells 0–10 are also extracted unchanged in `Paper3_Code/source_stages/`. The earlier code gap is resolved at the source-inspection level. Full raw-data reproduction is not yet validated. See `RECOVERED_SOURCE.md` for the stage map and replay constraints.

## Small existing files to collect first

Most filenames below are referenced by the supplied target scripts. The recovered notebook establishes the exact schema location: `Paper3_Materialized/feature_schema_v1.json` and its `.sha256` sidecar. Collect these existing files, rather than generating a new schema and claiming original provenance:

| Folder | Requested files |
|---|---|
| `Paper3_Materialized` | `feature_schema_v1.json` and `feature_schema_v1.sha256` |
| `Paper3_Sanitized` | Per-seed `sanitization_manifest.json`, `sanitization_complete.json`, `sanitization_summary.csv` |
| `Paper3_Source_Models` | `source_model_seal_manifest.json`, `source_model_seal_manifest.sha256`, `source_test_metrics_all_models.csv`, and per-seed `source_seed_complete.json` files |
| `Paper3_Target_Panels` | `target_candidate_panel_seal.json`, `target_candidate_panel_index.json`, `direction_index_seal.json`, `direction_index_summary.csv`, candidate/direction manifests |
| `Paper3_Target_Predictions` | `pre_label_target_prediction_seal.json`, `target_prediction_inventory.csv`, per-job `*_prediction_manifest.json` files |

The exact ordered 49-feature schema is needed for a full rerun. The received protocol records the count/exclusions and the target scripts derive the feature order from a sanitised training Parquet file. The formal schema artifact with the recorded schema hash was not included; do not replace it with an inferred list and claim the same provenance.

Raw datasets and large models/Parquet files need not be uploaded to GitHub. A later full rerun can obtain datasets from their providers and regenerate intermediate files once the required prerequisites and a documented replay environment are available. Exact byte-for-byte recreation of timestamped manifests may differ from numerical reproduction; document both separately.
