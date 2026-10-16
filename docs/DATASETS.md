# Dataset access

Download the four **NetFlow v3** benchmark datasets from the original University of Queensland catalog:

https://staff.itee.uq.edu.au/marius/NIDS_datasets/

| Dataset | Provider download page |
|---|---|
| NF-UNSW-NB15-v3 | https://rdm.uq.edu.au/files/abd2f5d8-e268-4ff0-84fb-f2f7b3ca3e8f |
| NF-ToN-IoT-v3 | https://rdm.uq.edu.au/files/343e2e8c-6e6e-4a0c-813d-a46acea1b7f4 |
| NF-BoT-IoT-v3 | https://rdm.uq.edu.au/files/db0187cc-1f78-4215-8e34-303364498866 |
| NF-CSE-CIC-IDS2018-v3 | https://rdm.uq.edu.au/files/4ac221b1-6bd6-42b1-bdf7-03f4fc7efb22 |

Links were checked against the provider catalog on 16 October 2026. The archives were not downloaded again during preparation of this repository. Follow the provider's citation and reuse terms. Dataset files are not relicensed or redistributed by this package.

The v3 catalog describes 53 NetFlow features; the study excludes IP addresses and absolute timestamps from modelling, leaving 49 predictors. Class-label fields are not predictors. The exact archive member names/column inventory observed during the study are recorded in `Paper3_Audit/dataset_inventory_audit.json`. Public description counts can differ from the audited archives; use the actual archived input and audit when reproducing the study rather than silently substituting another dataset version.

The current package needs no raw dataset for result-level verification. A full rerun additionally needs the recovered source-stage code, its missing prerequisite artifacts and a documented input/archive version. The dataset DOI, dataset article DOI and this repository's future release identifier are distinct objects; do not substitute one for another.
