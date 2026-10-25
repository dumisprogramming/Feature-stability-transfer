# Independent dataset compatibility assessment

Checked on 25 October 2026.

The primary University of Queensland NetFlow dataset catalogue lists four V3 datasets, matching the four already used here. It describes V3 as adding temporal features to V2. Source: https://staff.itee.uq.edu.au/marius/NIDS_datasets/ (NetFlow V3 Datasets section).

BigFlow-NIDS was considered because its release has 55 flow attributes. Its own Mendeley Data description explicitly says it merges NF-UNSW-NB15-v3, NF-ToN-IoT-v3, NF-BoT-IoT-v3 and NF-CSE-CIC-IDS2018-v3. It is therefore not an independent fifth capture for this study. Source: https://data.mendeley.com/datasets/nv729tbdgz/2 ; DOI: 10.17632/nv729tbdgz.2 . The associated paper is https://doi.org/10.1016/j.dib.2026.112530 .

No directly compatible independent fifth dataset was verified in this bounded search. This is not proof that none exists. A new dataset must establish independent capture provenance, matching definitions and units for all 49 ordered predictors, valid binary labels, source-feature overlap exclusions, fixed preprocessing and threshold reuse, and adequate sample support. Similar column names or a merged/repackaged version of existing captures are insufficient.

If only raw packet captures or a different feature representation are available, a validated feature-extraction/harmonisation study is needed before applying the frozen models. Filling missing temporal predictors with zeros or silently reducing the schema would change the experiment. This extension remains pending; it is not represented as completed external validation.

Suggested factual response to the supervisor: We evaluated compatibility with additional public releases. The current UQ V3 collection contains the four captures already used, while BigFlow-NIDS merges those same captures. We therefore retain independent capture validation as a limitation rather than count a merged release as a new dataset. Additional independently collected data with the fixed feature semantics are needed.
