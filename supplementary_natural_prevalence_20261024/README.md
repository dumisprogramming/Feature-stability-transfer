# Post hoc unstratified target evaluation

Completed in Colab on 24 October 2026: 180 fixed-model jobs on 60 panels of 100,000 target rows, sampled without class quotas from the original four raw captures after exact source-signature exclusions. Original classifiers, preprocessors and thresholds were retained. No detector was retrained. All source-model replays passed.

These are benchmark capture class mixtures, not production prevalence. This evaluates detector performance beyond the balanced original panels. It does not refit or retest warning regressions on the new outcomes, add independent datasets, resample alternative temporal windows or implement family-held-out training.

The exact received report ZIP retains all job records and checksums. The loose CSVs and code are convenience copies. The notebook requires the author's original Drive artifacts and raw archives; this support folder is not a standalone raw-data reproduction package. The report audit recomputed count-derived metrics but did not independently replay per-flow target probabilities or average precision. See Supplementary Information S10 for definitions and limitations.

Upload this complete folder to the repository root. Preserve its internal structure and original result ZIP. No publication DOI or new license is assigned here.
