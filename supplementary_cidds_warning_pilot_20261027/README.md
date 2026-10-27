# CIDDS source stability and transfer pilot

Support for Supplementary Information S20 of **Source-Only Feature-Importance Stability: Evaluating Its Value for Warning of Cross-Dataset Performance Loss in Network Intrusion Detection Systems**.

The completed pilot has 10 logistic-regression source fits, 10 source tests and 30 external evaluations over five seeds. Mean source BA is 0.8461466, mean target BA 0.7488101 and mean loss 0.0973365. Of 30 transfers, 28 degrade and 2 improve. All 10 warning records are defined. Two related source periods do not establish general warning utility.

## Verify saved results

From this folder, in an environment with NumPy installed:

```bash
python tools/verify_support.py
```

The optional `requirements-verification.txt` records the NumPy version used by the reported Colab pilot. The verifier does not need scikit-learn, pandas or raw datasets. It checks support-file hashes and the original returned report, then independently recomputes calibration optima, metric counts, importance summaries, rank pairs, warning scores and aggregate/directional arithmetic. PASS does not independently reproduce omitted training matrices, model fitting, raw-data recovery or feature permutations.

## Contents

- `archives/`: original returned pilot ZIP, byte for byte. Its embedded partition, qualification, correction and acquisition reports preserve the earlier audit chain. Keep this ZIP compressed for GitHub upload.
- `results/`: exact JSON copies and derived CSVs. Numerical precision is retained in the JSON records; CSV rows retain the supplied values.
- `code/`: exact executed pilot, partition and projection code, fixed pilot plan, and independent saved-result verifier.
- `audit/`: verification and document-preservation records.
- `tools/`: package verification entry point.

Original dataset providers retain their data rights. Raw CSVs, raw addresses, UIDs, candidate cohort arrays and partition matrices are not included. The original pilot archive contains numerical models and benchmark-derived calibration/test labels and probabilities for saved-result verification. The code is archived for provenance; it does not advertise a validated one-command rerun from raw traffic. No new licence or DOI is assigned by this preparation step.

## Scientific scope

The four candidate periods are related CIDDS capture periods from Coburg's collection setting. Only CIDDS-001 internal weeks 1 and 2 meet the fixed source gates in all seeds. CIDDS-002 weeks 1 and 2 qualify as targets but have only three eligible environment windows each, below the source minimum of four. No gate was relaxed.

The selected cohort uses four predictors and bounded distinct-feature-group sampling within class/time window. Binary conflicts were excluded across potential source and target captures using already known labels. This is post-hoc, target-label-informed cohort construction. No absence-of-duplicates claim covers complete raw datasets. Same-role signatures remain shared across periods. Seeds, jobs and periods must not be treated as independent institutions. HIKARI files were inventoried but not fitted because supplied flow CSVs lack timestamps and cross-exporter semantics remain uncertified.

No correlation test, warning-utility regression, operational-prevalence estimate or fine-family-equivalence certification was performed. These broader requirements remain unresolved. This pilot is separate from the original 49-feature, three-model NetFlow-v3 study and the six-feature CTU pilot.

## Dataset citation

Ring M, Wunderlich S, Grüdl D, Landes D, Hotho A. Flow-based benchmark data sets for intrusion detection. Proceedings of the 16th European Conference on Cyber Warfare and Security. ACPI; 2017. pp. 361–369.

Official access: https://www.hs-coburg.de/forschen/cidds-coburg-intrusion-detection-data-sets/

The study manuscript is not represented as accepted or published. Public upload of this support folder remains an author action.
