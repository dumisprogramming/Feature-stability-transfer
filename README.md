# Source side feature stability and cross dataset NIDS degradation

Research materials for **When Source-Side Feature Stability Fails to Warn of Cross-Dataset Network Intrusion Detection Degradation**.

Author: S. M. Bahauddin Bakhtiar, Department of Management Information Systems, University of Dhaka, Bangladesh. ORCID: https://orcid.org/0009-0000-5180-367X

This package provides the supplied frozen protocols, source-stability artifacts, final evaluation tables, audit reports, four original target-stage scripts, and the recovered original `Paper3_CompleteCode.ipynb` notebook with eleven source-stage cells extracted unchanged into separate scripts. It supports **verification and reanalysis of saved results**, plus inspection of the recovered upstream implementation. A complete rerun from raw datasets has **not been validated**: original prerequisite files are still missing and historical paths, timestamped seals and hard-coded expected hashes require a separately documented replay. See [recovered source guide](docs/RECOVERED_SOURCE.md).

## Study and main finding

The study evaluates four harmonised NetFlow-v3 datasets, 49 predictors, three classifiers and five seeds (2027–2031). Twelve directed transfers generate 180 target jobs, with 40,000 benign and 40,000 attack flows per job. The confirmatory association analysis uses 12 source-dataset-by-model units; LOSO predictive validation uses 60 source-dataset-by-seed-by-model units.

Mean source balanced accuracy is approximately 0.972, target balanced accuracy 0.590 and loss 0.382. Neither primary stability warning has a statistically supported positive association with loss under the prespecified tests. Adding the warnings does not outperform the model-identity-only LOSO baseline. These findings concern the specified benchmarks, conventional models, binary task and zero-adaptation protocol. They do not establish universal absence of useful source-side warnings or operational deployment readiness.

## Quick start with saved results

Extract the ZIP, open a terminal in this repository directory, and use Python 3.12 in a virtual environment:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell alternative: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python tools/verify_artifacts.py --report outputs/artifact_verification.json
python tools/reproduce_analysis.py
python tools/verify_recovered_stability.py
python tools/paper3_build_publication_assets.py --input-dir Paper3_Final_Evaluation --output-dir outputs/publication_assets
```

The reanalysis command runs the original 13,824 blocked permutations and 10,000 bootstrap replicates per predictor, then recomputes the six LOSO specifications. It can take several minutes on CPU. For a shorter run without recomputing the bootstrap intervals:

```bash
python tools/reproduce_analysis.py --skip-bootstrap --output-dir outputs/quick_reanalysis
```

The shorter run still computes the exact permutation tests. It does not validate the saved bootstrap intervals. Exit code 0 indicates that every check selected for that run passed; a nonzero exit code requires inspection of the JSON report. All new outputs go to separate directories. Original sealed files must remain unchanged.

For Colab, upload and unzip this package, change the working directory to the extracted repository, install `requirements.txt` and run the same commands with `!python`. Google Drive is not required for the saved-results route. The historical scripts in `Paper3_Code/` use a different Colab/Drive workflow and must not be substituted for these commands.

## Contents

| Path | Purpose |
|---|---|
| `Paper3_Code/` | Four original target-stage scripts and eleven exact source-stage cell extracts |
| `notebooks/` | Original complete notebook, retained byte for byte with saved outputs |
| `Paper3_Protocol/` | Original protocol, amendments A1–A3 and available hash sidecars |
| `Paper3_Final_Evaluation/` | Ten supplied CSV/JSON result and seal files |
| `Paper3_Source_Stability/` | Warning scores, 180 per-model Parquet/JSON files and seal records |
| `Paper3_Audit/` | Dataset inventory and temporal-count audits |
| `Paper3_Pretraining_Audit/` | Twenty dataset-seed gate records and detailed audit |
| `Paper3_Manuscript_Tables.xlsx` | Author-supplied workbook, preserved unchanged |
| `tools/` | Portable result-level verification/reanalysis tools and original publication builder |
| `reports/` | Verification results from preparation of this package |
| `outputs/` | Recomputed analyses, plots and table CSVs |
| `docs/` | Reproducibility scope, missing files, dataset access, column inventory and figure/table mapping |

The workbook is a convenience copy; the sealed CSV/JSON files are the authoritative inputs for the provided commands. The publication builder writes CSV tables and a workbook JSON payload, not a new XLSX workbook.

## Integrity and chronology

`verify_artifacts.py` verifies supplied protocol sidecars, A3 and final-results self-seals, nine final result hashes, all 181 source-stability artifact hashes, and the original evaluator/predictor script hashes. Paths in historical JSON records refer to the original Drive workspace; verification resolves known artifacts by their preserved relative path or filename.

Checksums establish agreement with supplied records. They are not independent timestamping, proof of absence of label access, or proof that an absent stage was run correctly. This package does not recreate the original prediction seal because raw prediction/label files and upstream models are not included. See [reproducibility scope](docs/REPRODUCIBILITY.md).

## Original target-stage scripts

The original files begin with `google.colab` Drive mounting and use the original project root. They are archived for provenance, not advertised as portable one-command programs. The dependency sequence, **after the required source-stage prerequisites exist**, is:

1. `paper3_target_candidate_panels.py`
2. `paper3_direction_indices.py`
3. `paper3_label_blind_target_predictions.py`
4. `paper3_final_target_evaluation.py`

The original evaluator freezes/verifies A3 and opens label vaults as part of its workflow. Do not run it against the archived result folders merely to inspect results. A historical replay and a new experiment have different provenance; never overwrite frozen originals or imply that newly generated files were sealed in the past.

## Dependencies

`requirements.txt` records the versions used for the portable checks during repository preparation. `docs/ORIGINAL_ENVIRONMENT.md` separately lists software versions recorded in the supplied historical artifacts. The recovered stability verification additionally uses PyArrow to read supplied importance tables. The original model/prediction workflow requires joblib, XGBoost, Google Colab and missing data/model prerequisites; that full workflow has not been tested here.

## Data access and rights

Raw datasets, trained estimators, per-flow labels and per-flow target predictions are excluded. Obtain raw datasets from their original providers; dataset rights are governed by those providers. See [dataset access](docs/DATASETS.md). Repository: https://github.com/dumisprogramming/Feature-stability-transfer No archival DOI has been assigned. Update the manuscript code-availability statement only after the repository is actually published.

No software/data reuse license is assigned by this preparation step. The author should choose an appropriate license before a public release. A public GitHub repository without a license is not equivalent to an open-source license grant.

## Citation

The Paper 3 manuscript is not represented as accepted or published. Use its title and author when referring to this work, and replace publication metadata only when verified. `CITATION.cff` describes this research software package without inventing a DOI, journal acceptance or GitHub URL.
