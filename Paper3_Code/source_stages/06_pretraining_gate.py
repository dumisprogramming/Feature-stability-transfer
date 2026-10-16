from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = Path(
    "/content/drive/MyDrive/research/researchdata/"
    "NIDS_Research/Paper3_DataSet"
)

PROTOCOL_DIR = ROOT / "Paper3_Protocol"
MATERIALIZED_DIR = ROOT / "Paper3_Materialized"
SANITIZED_DIR = ROOT / "Paper3_Sanitized"
AUDIT_DIR = ROOT / "Paper3_Pretraining_Audit"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_PROTOCOL_HASH = (
    "e39a5007a7f2cbdc348d5179ce593af"
    "5766bc6688f5baa3a46a56399f683a488"
)

EXPECTED_SCHEMA_HASH = (
    "67d61eebb3377e3053f943f1e4c03123"
    "7b3ef35dd06296d8538fe552b16be0c6"
)

EXPECTED_AMENDMENT_HASH = (
    "19c30a37d62d36f5b3c9bb795f64e105"
    "99dcef78a9c51bf96ca80b58d7db4b47"
)

DATASETS = [
    "NF_BoT_IoT_v3",
    "NF_CSE_CIC_IDS2018_v3",
    "NF_ToN_IoT_v3",
    "NF_UNSW_NB15_v3",
]

SEEDS = [2027, 2028, 2029, 2030, 2031]

PARTITIONS = [
    "train",
    "calibration",
    "source_test",
    "environment",
]


def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for part in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(part)

    return digest.hexdigest()


# ---------------------------------------------------------------
# Verify protocol chain
# ---------------------------------------------------------------

protocol_file = (
    PROTOCOL_DIR / "paper3_protocol_frozen_v1.json"
)

schema_file = (
    MATERIALIZED_DIR / "feature_schema_v1.json"
)

amendment_file = (
    PROTOCOL_DIR
    / "amendment_A1_duplicate_group_disjoint.json"
)

if sha256_file(protocol_file) != EXPECTED_PROTOCOL_HASH:
    raise RuntimeError("Original protocol hash mismatch.")

if sha256_file(schema_file) != EXPECTED_SCHEMA_HASH:
    raise RuntimeError("Feature-schema hash mismatch.")

if sha256_file(amendment_file) != EXPECTED_AMENDMENT_HASH:
    raise RuntimeError("Amendment A1 hash mismatch.")

with open(schema_file, "r", encoding="utf-8") as file:
    FEATURES = json.load(file)["predictors_in_order"]

print("=" * 78)
print("PAPER 3 — FINAL PRE-TRAINING DATA GATE")
print("=" * 78)
print("Original protocol: PASS")
print("Feature schema: PASS")
print("Amendment A1: PASS")
print(f"Predictors: {len(FEATURES)}")


audit_rows = []
detailed_audit = {}
problems = []

for dataset in DATASETS:
    detailed_audit[dataset] = {}

    print(f"\n{'-' * 78}")
    print(f"DATASET: {dataset}")
    print(f"{'-' * 78}")

    for seed in SEEDS:
        seed_dir = (
            SANITIZED_DIR
            / dataset
            / f"seed_{seed}"
        )

        manifest_file = (
            seed_dir / "sanitization_manifest.json"
        )

        with open(
            manifest_file,
            "r",
            encoding="utf-8",
        ) as file:
            manifest = json.load(file)

        seed_details = {
            "partition_details": {},
            "problems": [],
        }

        hashes_valid = True
        total_infinite = 0

        for partition in PARTITIONS:
            parquet_file = (
                seed_dir / f"{partition}.parquet"
            )

            expected_hash = manifest[
                "output_files"
            ][partition]["sha256"]

            observed_hash = sha256_file(parquet_file)

            if observed_hash != expected_hash:
                hashes_valid = False
                seed_details["problems"].append(
                    f"{partition}_hash_mismatch"
                )

            frame = pd.read_parquet(parquet_file)

            missing_features = [
                feature for feature in FEATURES
                if feature not in frame.columns
            ]

            if missing_features:
                seed_details["problems"].append(
                    f"{partition}_missing_features"
                )
                problems.append(
                    f"{dataset} seed {seed} "
                    f"{partition}: missing predictors"
                )
                continue

            feature_array = frame[
                FEATURES
            ].to_numpy(dtype=np.float64)

            infinite_count = int(
                np.isinf(feature_array).sum()
            )

            total_infinite += infinite_count

            labels_valid = (
                frame["Label"].isin([0, 1]).all()
            )

            label_counts = (
                frame["Label"]
                .value_counts()
                .sort_index()
                .to_dict()
            )

            balanced = (
                int(label_counts.get(0, 0))
                == int(label_counts.get(1, 0))
            )

            missing_counts = (
                frame[FEATURES].isna().sum()
            )

            missing_fractions = (
                missing_counts / len(frame)
            )

            partition_details = {
                "rows": int(len(frame)),
                "label_0": int(
                    label_counts.get(0, 0)
                ),
                "label_1": int(
                    label_counts.get(1, 0)
                ),
                "balanced": bool(balanced),
                "labels_valid": bool(labels_valid),
                "infinite_values": infinite_count,
                "maximum_missing_fraction": float(
                    missing_fractions.max()
                ),
            }

            if partition == "train":
                all_missing_features = (
                    missing_counts[
                        missing_counts == len(frame)
                    ].index.tolist()
                )

                constant_features = []

                for feature in FEATURES:
                    if (
                        frame[feature]
                        .dropna()
                        .nunique()
                        <= 1
                    ):
                        constant_features.append(feature)

                partition_details[
                    "all_missing_features"
                ] = all_missing_features

                partition_details[
                    "constant_features"
                ] = constant_features

                if all_missing_features:
                    seed_details["problems"].append(
                        "all_missing_training_features"
                    )

            if infinite_count != 0:
                seed_details["problems"].append(
                    f"{partition}_contains_infinity"
                )

            if not labels_valid:
                seed_details["problems"].append(
                    f"{partition}_invalid_labels"
                )

            if not balanced:
                seed_details["problems"].append(
                    f"{partition}_not_balanced"
                )

            seed_details["partition_details"][
                partition
            ] = partition_details

        maximum_final_overlap = int(
            manifest["maximum_final_overlap"]
        )

        if maximum_final_overlap != 0:
            seed_details["problems"].append(
                "nonzero_cross_partition_overlap"
            )

        train_details = seed_details[
            "partition_details"
        ]["train"]

        gate_pass = (
            hashes_valid
            and total_infinite == 0
            and maximum_final_overlap == 0
            and not train_details[
                "all_missing_features"
            ]
            and not seed_details["problems"]
        )

        if not gate_pass:
            problems.append(
                f"{dataset} seed {seed}: "
                f"{seed_details['problems']}"
            )

        audit_rows.append({
            "dataset": dataset,
            "seed": seed,
            "train_rows": train_details["rows"],
            "calibration_rows": seed_details[
                "partition_details"
            ]["calibration"]["rows"],
            "source_test_rows": seed_details[
                "partition_details"
            ]["source_test"]["rows"],
            "environment_rows": seed_details[
                "partition_details"
            ]["environment"]["rows"],
            "constant_train_features": len(
                train_details["constant_features"]
            ),
            "all_missing_train_features": len(
                train_details[
                    "all_missing_features"
                ]
            ),
            "max_train_missing_pct": round(
                100
                * train_details[
                    "maximum_missing_fraction"
                ],
                4,
            ),
            "total_infinite": total_infinite,
            "file_hashes_valid": hashes_valid,
            "cross_partition_overlap": (
                maximum_final_overlap
            ),
            "gate_pass": gate_pass,
        })

        seed_details["gate_pass"] = gate_pass
        detailed_audit[dataset][str(seed)] = (
            seed_details
        )

        print(
            f"  Seed {seed}: "
            f"constant={len(train_details['constant_features'])}, "
            f"all-missing="
            f"{len(train_details['all_missing_features'])}, "
            f"max-missing="
            f"{100 * train_details['maximum_missing_fraction']:.4f}%, "
            f"infinite={total_infinite}, "
            f"overlap={maximum_final_overlap}, "
            f"hashes={'PASS' if hashes_valid else 'FAIL'}, "
            f"gate={'PASS' if gate_pass else 'FAIL'}"
        )


audit_table = pd.DataFrame(audit_rows)

audit_csv = AUDIT_DIR / "pretraining_gate_summary.csv"
audit_json = AUDIT_DIR / "pretraining_gate_details.json"

audit_table.to_csv(audit_csv, index=False)

audit_json.write_text(
    json.dumps(
        {
            "protocol_sha256": EXPECTED_PROTOCOL_HASH,
            "schema_sha256": EXPECTED_SCHEMA_HASH,
            "amendment_A1_sha256": (
                EXPECTED_AMENDMENT_HASH
            ),
            "details": detailed_audit,
            "problems": problems,
        },
        indent=2,
        sort_keys=True,
    ),
    encoding="utf-8",
)

overall_pass = (
    audit_table["gate_pass"].all()
    and len(problems) == 0
)

print("\n" + "=" * 78)
print(
    "FINAL PRE-TRAINING GATE: "
    f"{'PASS' if overall_pass else 'FAIL'}"
)
print(f"Problems: {problems}")
print(f"Summary: {audit_csv}")
print("=" * 78)

if not overall_pass:
    raise RuntimeError(
        "Pre-training gate failed. Do not train models."
    )