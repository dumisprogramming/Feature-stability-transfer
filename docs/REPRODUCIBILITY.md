# Reproducibility scope

## What this package can verify

- Byte agreement with available protocol/hash sidecars and result/stability manifests.
- The supplied 180-job table, 60-unit and 12-unit aggregations.
- Prespecified positive one-sided blocked Spearman tests, Holm adjustment and original bootstrap intervals, using the included original statistical function definitions.
- The original six ordinary-least-squares LOSO specifications from saved high-precision units.
- Descriptive means, standard deviations, ranges, plot assets and table CSVs from supplied result tables.

`tools/analysis_core.py` contains seven function definitions copied verbatim from the supplied final-evaluation script. It adds only the imports/constants needed to call those functions independently. This avoids executing Colab mounting, the original top-level workflow, raw label access or historical writes. The original script remains unchanged.

## What it cannot independently verify

- Rebuilding source partitions, sanitisation, fitted models or stability scores from raw CSVs: the source-stage code has now been recovered, but those raw-data stages have not been rerun here and their prerequisite artifacts are incomplete.
- Regenerating target probabilities and per-flow metrics: fitted models, preprocessors, candidate/direction panels and raw predictions/labels are not included.
- External proof of when a file was created or when a researcher first saw labels: hashes and self-reported chronology are not a trusted timestamping service.
- Causal mechanisms, generalisation to other networks, or the validity of every modelling assumption.

The provided reanalysis checks numerical agreement with the originally reported procedure. It is not a new independent experiment, a new bootstrap design, or a remedy for the limited four-dataset scope.

## Statistical interpretation

The 180 jobs share source/target datasets, models and seeds; they are not 180 independent environments. The original association procedure averages them to 12 source-dataset-by-model units and permutes source assignments independently within model. The original bootstrap resamples four sources separately within each model. Those choices are retained for faithful replay, not claimed to eliminate all shared-dataset dependence.

LOSO holds out the source role only. The held-out dataset can still occur as a target in the training units' transfer-loss averages. Therefore this is not a completely unseen-dataset-in-every-role test. No target adaptation or threshold change is introduced by this package.

## Data safety and provenance

All files in the six supplied archive folders are kept unchanged, including historical absolute Drive paths and timestamps. No credentials were found by the targeted source/text scan. The word `password` in a temporal audit is an attack-category count, not a stored credential. The supplied XLSX has no external-link parts. This scan is limited to the supplied file contents and is not a formal privacy certification.

No estimator pickle/joblib file is needed or loaded by the portable verification route. Generated figures are scientific Matplotlib plots. The package does not include raw network traffic, the original private Drive contents or browser sessions.

## Recovered notebook validation

The original complete notebook and eleven exact code-cell extracts are now included. The search-only `Untitled60.ipynb` is excluded. `tools/verify_recovered_stability.py` isolates only the recovered pairwise-stability function and checks its summaries against all 60 supplied permutation-importance tables and completion records. This checks aggregation of saved importance values; it does not regenerate permutation importances or fit models. See `reports/recovered_stability_verification.json` and `docs/RECOVERED_SOURCE.md`.
