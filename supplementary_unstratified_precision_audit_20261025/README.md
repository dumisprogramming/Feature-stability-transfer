# Unstratified target panel precision audit

The read-only Colab audit completed on 25 October 2026. It checked 60 source–target–seed panels of 100,000 rows each, shared by three classifiers. Canonical float32 signatures were compared against source training, calibration, source-test and environmental partitions.

## Findings

- Zero selected target rows matched corresponding source features at model precision.
- All original panel selections reproduced from saved 200,000-row pools.
- Corrected float32 exclusions returned identical ordered panel row IDs and attack fractions in all 60 cases.
- All pools retained enough eligible candidates; no panel repair, model fit or new target prediction was needed.
- These are 6,000,000 panel-row occurrences, not independent flows.

This closes the precision-exclusion concern for the S10 unstratified outcomes and dependent S11 and S12 regressions. It does not change their numerical results or post hoc status.

## Contents and use

`audit_report.zip` is the unchanged report returned from Colab. `report/` exposes its exact files. Run `python verify_report.py` with Python 3 to check report hashes, the full panel grid, partition counts and pinned input consistency. It uses only the standard library. This command verifies saved records; it does not repeat feature-signature comparisons on original Parquets. `report/AUDIT_CODE.py` records the original Drive audit implementation, not a portable raw-data entry point.

## Verification scope

During preparation, the expected context was also checked against the original supplied natural-prevalence report: its whole-file hash, run identity, 81 schema/source-partition pins and 180 job-marker hashes agreed. Feature/label input hashes agreed with the pinned panel context. All ten audit report hashes matched.

Pool hashes were checked against matching Drive completion markers and newly recorded. Those markers were absent from the earlier downloaded experiment report, so they are not independently historically pinned. No complete raw-capture scan independently replayed global lowest-priority pool membership. Target probability predictions and average precision were not independently replayed. This is a saved-panel exclusion audit, not complete raw-data replication.

## Preservation

Keep original experiment directories and sealed inputs unchanged. This folder adds audit evidence. Publication does not create new dataset rights, a software license or a DOI; follow the parent repository's applicable rights.
