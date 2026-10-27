# CTU 13 bounded warning pilot

Supplementary S19 for Source-Only Feature-Importance Stability: Evaluating Its Value for Warning of Cross-Dataset Performance Loss in Network Intrusion Detection Systems.

The completed post hoc pilot supplies 25 source LR fits, 175 external evaluations and 25 source tests using five source captures, eight test captures and five seeds. No tested warning specification improves its strict split's intercept baseline. Its six-predictor Argus representation and label-informed, balanced-known-label cohort are separate from the original NetFlow-v3 experiment.

## Verify saved results

Use Python 3.12 or 3.13, open a terminal in this folder, and run:

```bash
python -m pip install -r requirements.txt
python tools/verify_ctu_s19.py --root .
```

No Google Drive or raw dataset download is required for this check. It verifies the three archived report chains, hashes and markers, support gates, calibration thresholds, metrics, importance means, rank pairs and strict ridge predictions. Success prints status PASS. It does not replay model fitting, raw recovery, omitted input-array intersections or predictions from input features.

For a Colab rerun of the model stage, open pilot/Paper3_CTU13_Warning_Pilot_20261027.ipynb and Run all on CPU. It needs the original completed inventory and qualification Drive folders, including panel arrays; the verifier does not reconstruct them. The corrected setup bypasses unavailable ensurepip. Checkpoints lock code, plan, input context and runtime. Do not change historical input seals to force resume.

## Results

| Outer holdout | Intercept MAE | Environment MAE | Gap MAE | Both MAE |
|---|---:|---:|---:|---:|
| Source capture | 0.122148 | 0.136194 | 0.127072 | 0.136541 |
| Documented specimen group | 0.122988 | 0.335895 | 0.250657 | 0.273251 |

Errors are averaged over source captures. Sources are 1, 2, 6, 9 and 13. Specimen folds are {1,2,9}, {6}, {13}; these are not independent institutions. Strict folds remove held-out captures from training-source units and training-target outcomes. Mean source BA is 0.768237, target BA 0.646522 and loss 0.121714. Of 175 external jobs, 116 degrade and 59 improve.

## Contents and scope

- archives/: original inventory, qualification and pilot ZIP reports preserved byte for byte.
- results/: exact CSV copies of outcomes, warnings and predictive checks.
- tools/: portable saved-result verification code.
- pilot/: scientific engine, exact context/plan and corrected Colab notebook.
- audit/: reviewed verification results and run summaries.
- docs/: interpretation and limitations.

The reports include numerical model parameters, labels and probabilities. Raw traffic, endpoints, executables and input feature panels are excluded. Retain the full Drive folders for input-level replay. Checksums establish agreement with records, not independent timestamps, label truth or proof of execution.

This completes the bounded S19 pilot. It does not establish production prevalence, general independent-site warning utility or equivalent fine-family semantics. Conflict exclusion uses target labels after earlier outcomes were known. Unknown labels are not counted as benign. Same-role patterns, reused tests, shared backgrounds and repeated specimens remain dependent. These are post hoc supplementary results, without new significance claims.

CTU-13 provider: https://www.stratosphereips.org/datasets-ctu13. Cite Garcia S, Grill M, Stiborek J, Zunino A. An empirical comparison of botnet detection methods. Computers & Security. 2014;45:100–123. https://doi.org/10.1016/j.cose.2014.05.011.

Dataset terms remain the provider's. This preparation assigns no new reuse license; use applicable repository licensing and author permissions. No paper acceptance or archive DOI is claimed.
