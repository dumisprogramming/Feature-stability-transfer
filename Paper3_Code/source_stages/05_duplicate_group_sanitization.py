from pathlib import Path
from datetime import datetime, timezone
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
SANITIZED_DIR.mkdir(parents=True, exist_ok=True)

PROTOCOL_FILE = (
    PROTOCOL_DIR / "paper3_protocol_frozen_v1.json"
)

SCHEMA_FILE = (
    MATERIALIZED_DIR / "feature_schema_v1.json"
)

EXPECTED_PROTOCOL_HASH = (
    "e39a5007a7f2cbdc348d5179ce593af"
    "5766bc6688f5baa3a46a56399f683a488"
)

EXPECTED_SCHEMA_HASH = (
    "67d61eebb3377e3053f943f1e4c03123"
    "7b3ef35dd06296d8538fe552b16be0c6"
)

DATASETS = [
    "NF_BoT_IoT_v3",
    "NF_CSE_CIC_IDS2018_v3",
    "NF_ToN_IoT_v3",
    "NF_UNSW_NB15_v3",
]

SEEDS = [2027, 2028, 2029, 2030, 2031]

ROLE_SALTS = {
    "source_test": np.uint64(0xA24BAED4963EE407),
    "calibration": np.uint64(0x9FB21C651E98DF25),
    "train": np.uint64(0xC13FA9A902A6328F),
}


def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for part in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(part)

    return digest.hexdigest()


def atomic_parquet(frame, path):
    temporary = path.with_suffix(".temporary.parquet")

    frame.to_parquet(
        temporary,
        index=False,
        engine="pyarrow",
        compression="zstd",
    )

    temporary.replace(path)


def feature_hashes(frame, feature_columns):
    return pd.util.hash_pandas_object(
        frame[feature_columns],
        index=False,
        categorize=True,
    ).to_numpy(dtype=np.uint64)


def splitmix64(values):
    values = values.astype(np.uint64, copy=False)

    with np.errstate(over="ignore"):
        values = values + np.uint64(
            0x9E3779B97F4A7C15
        )
        values = (
            values ^ np.right_shift(values, 30)
        ) * np.uint64(0xBF58476D1CE4E5B9)
        values = (
            values ^ np.right_shift(values, 27)
        ) * np.uint64(0x94D049BB133111EB)
        values = values ^ np.right_shift(values, 31)

    return values


def deterministic_balance(frame, seed, salt):
    class_counts = (
        frame["Label"].value_counts().to_dict()
    )

    if 0 not in class_counts or 1 not in class_counts:
        raise RuntimeError(
            "Both binary classes are required."
        )

    keep_per_class = min(
        int(class_counts[0]),
        int(class_counts[1]),
    )

    selected_parts = []

    for label in [0, 1]:
        class_frame = frame[
            frame["Label"] == label
        ].copy()

        row_ids = class_frame[
            "row_id"
        ].to_numpy(dtype=np.uint64)

        priorities = splitmix64(
            row_ids
            ^ np.uint64(seed)
            ^ salt
        )

        if len(class_frame) > keep_per_class:
            selected = np.argpartition(
                priorities,
                keep_per_class - 1,
            )[:keep_per_class]

            class_frame = class_frame.iloc[
                selected
            ].copy()

        selected_parts.append(class_frame)

    balanced = pd.concat(
        selected_parts,
        ignore_index=True,
    )

    return (
        balanced
        .sort_values(["Label", "row_id"])
        .reset_index(drop=True)
    )


def exclude_forbidden_hashes(
    frame,
    feature_columns,
    forbidden_hashes,
):
    if not forbidden_hashes:
        return frame.copy(), 0

    hashes = feature_hashes(
        frame,
        feature_columns,
    )

    forbidden_array = np.fromiter(
        forbidden_hashes,
        dtype=np.uint64,
    )

    keep_mask = ~np.isin(
        hashes,
        forbidden_array,
    )

    removed = int((~keep_mask).sum())

    return (
        frame.loc[keep_mask].reset_index(drop=True),
        removed,
    )


# ---------------------------------------------------------------
# 1. Verify frozen artifacts
# ---------------------------------------------------------------

if sha256_file(PROTOCOL_FILE) != EXPECTED_PROTOCOL_HASH:
    raise RuntimeError("Protocol hash mismatch.")

if sha256_file(SCHEMA_FILE) != EXPECTED_SCHEMA_HASH:
    raise RuntimeError("Feature-schema hash mismatch.")

with open(SCHEMA_FILE, "r", encoding="utf-8") as file:
    schema = json.load(file)

FEATURES = schema["predictors_in_order"]

if len(FEATURES) != 49:
    raise RuntimeError("Expected exactly 49 predictors.")

print("=" * 78)
print("PAPER 3 — DUPLICATE-GROUP SANITIZATION")
print("=" * 78)
print("Protocol verification: PASS")
print("Feature-schema verification: PASS")


# ---------------------------------------------------------------
# 2. Collect the complete observed overlap audit
# ---------------------------------------------------------------

overlap_rows = []

for dataset in DATASETS:
    for seed in SEEDS:
        manifest_file = (
            MATERIALIZED_DIR
            / dataset
            / f"seed_{seed}"
            / "materialization_manifest.json"
        )

        with open(
            manifest_file,
            "r",
            encoding="utf-8",
        ) as file:
            manifest = json.load(file)

        for pair, count in manifest[
            "cross_partition_feature_hash_overlap"
        ].items():
            overlap_rows.append({
                "dataset": dataset,
                "seed": seed,
                "partition_pair": pair,
                "overlapping_feature_hashes": int(count),
            })

overlap_table = pd.DataFrame(overlap_rows)

overlap_audit_file = (
    PROTOCOL_DIR
    / "pre_modelling_feature_overlap_audit.csv"
)

overlap_table.to_csv(
    overlap_audit_file,
    index=False,
)

observed_maximum = int(
    overlap_table[
        "overlapping_feature_hashes"
    ].max()
)

observed_total = int(
    overlap_table[
        "overlapping_feature_hashes"
    ].sum()
)

print(f"Maximum observed pair overlap: {observed_maximum:,}")
print(f"Sum across audited pairs: {observed_total:,}")


# ---------------------------------------------------------------
# 3. Create amendment before sanitization or modelling
# ---------------------------------------------------------------

amendment_file = (
    PROTOCOL_DIR
    / "amendment_A1_duplicate_group_disjoint.json"
)

amendment_hash_file = (
    PROTOCOL_DIR
    / "amendment_A1_duplicate_group_disjoint.sha256"
)

if amendment_file.exists():
    amendment_hash = sha256_file(amendment_file)

    if not amendment_hash_file.exists():
        raise RuntimeError(
            "Existing amendment has no hash file."
        )

    recorded_hash = (
        amendment_hash_file
        .read_text(encoding="utf-8")
        .strip()
    )

    if amendment_hash != recorded_hash:
        raise RuntimeError(
            "Existing amendment hash mismatch."
        )

else:
    amendment = {
        "amendment_id": (
            "P3_A1_DUPLICATE_GROUP_DISJOINT"
        ),
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "timing": (
            "Created after data-quality auditing but "
            "before model training, feature-importance "
            "estimation, or cross-dataset evaluation."
        ),
        "parent_protocol_sha256": (
            EXPECTED_PROTOCOL_HASH
        ),
        "feature_schema_sha256": (
            EXPECTED_SCHEMA_HASH
        ),
        "reason": (
            "The pre-modelling audit detected identical "
            "49-feature signatures across source partitions."
        ),
        "observed_maximum_pair_overlap": (
            observed_maximum
        ),
        "observed_total_across_pairs": (
            observed_total
        ),
        "primary_rule": {
            "group_definition": (
                "Identical cleaned values across all "
                "49 model predictors."
            ),
            "signature_method": (
                "Pandas deterministic 64-bit row hash; "
                "collision probability is negligible at "
                "the audited sample sizes."
            ),
            "partition_precedence": [
                "environment",
                "source_test",
                "calibration",
                "train",
            ],
            "action": (
                "When a feature signature occurs in more "
                "than one partition, retain it only in the "
                "highest-priority partition."
            ),
            "environment_samples": (
                "Preserved to retain the previously frozen "
                "2,000 observations per class per block."
            ),
            "within_partition_multiplicity": (
                "Retained in the primary analysis to preserve "
                "observed traffic frequency."
            ),
            "post_purge_balancing": (
                "Training, calibration and source-test "
                "partitions are independently downsampled "
                "to equal benign and attack counts."
            ),
            "replacement": False,
        },
        "unchanged_elements": [
            "research question",
            "four datasets",
            "six-hour environmental unit",
            "eligible blocks",
            "models",
            "five seeds",
            "49 predictors",
            "source-only warning scores",
            "target-blind evaluation",
            "primary outcome",
        ],
        "model_results_seen_before_amendment": False,
        "target_results_seen_before_amendment": False,
    }

    amendment_text = json.dumps(
        amendment,
        indent=2,
        sort_keys=True,
    )

    amendment_file.write_text(
        amendment_text,
        encoding="utf-8",
    )

    amendment_hash = sha256_file(amendment_file)

    amendment_hash_file.write_text(
        amendment_hash,
        encoding="utf-8",
    )

print(f"Amendment A1 SHA-256: {amendment_hash}")
print("Amendment timing verification: PRE-MODELLING")


# ---------------------------------------------------------------
# 4. Sanitize each dataset and seed
# ---------------------------------------------------------------

overall_rows = []

for dataset in DATASETS:

    dataset_output = SANITIZED_DIR / dataset
    dataset_output.mkdir(parents=True, exist_ok=True)

    completion_file = (
        dataset_output / "sanitization_complete.json"
    )

    if completion_file.exists():
        print(f"\nSKIPPED—already sanitized: {dataset}")
        continue

    print(f"\n{'-' * 78}")
    print(f"DATASET: {dataset}")
    print(f"{'-' * 78}")

    dataset_manifests = {}

    for seed in SEEDS:

        source_dir = (
            MATERIALIZED_DIR
            / dataset
            / f"seed_{seed}"
        )

        output_dir = (
            dataset_output / f"seed_{seed}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        frames = {
            name: pd.read_parquet(
                source_dir / f"{name}.parquet"
            )
            for name in [
                "environment",
                "source_test",
                "calibration",
                "train",
            ]
        }

        original_counts = {
            name: int(len(frame))
            for name, frame in frames.items()
        }

        # Highest-priority partition remains unchanged.
        environment = frames["environment"].copy()

        environment_hashes = set(
            feature_hashes(
                environment,
                FEATURES,
            ).tolist()
        )

        # Source test loses environment-overlapping groups.
        source_test, test_overlap_removed = (
            exclude_forbidden_hashes(
                frames["source_test"],
                FEATURES,
                environment_hashes,
            )
        )

        before_balance = len(source_test)

        source_test = deterministic_balance(
            source_test,
            seed,
            ROLE_SALTS["source_test"],
        )

        test_balance_removed = (
            before_balance - len(source_test)
        )

        source_test_hashes = set(
            feature_hashes(
                source_test,
                FEATURES,
            ).tolist()
        )

        # Calibration loses environment and final test groups.
        calibration_forbidden = (
            environment_hashes
            | source_test_hashes
        )

        calibration, cal_overlap_removed = (
            exclude_forbidden_hashes(
                frames["calibration"],
                FEATURES,
                calibration_forbidden,
            )
        )

        before_balance = len(calibration)

        calibration = deterministic_balance(
            calibration,
            seed,
            ROLE_SALTS["calibration"],
        )

        cal_balance_removed = (
            before_balance - len(calibration)
        )

        calibration_hashes = set(
            feature_hashes(
                calibration,
                FEATURES,
            ).tolist()
        )

        # Training loses every evaluation/calibration group.
        train_forbidden = (
            environment_hashes
            | source_test_hashes
            | calibration_hashes
        )

        train, train_overlap_removed = (
            exclude_forbidden_hashes(
                frames["train"],
                FEATURES,
                train_forbidden,
            )
        )

        before_balance = len(train)

        train = deterministic_balance(
            train,
            seed,
            ROLE_SALTS["train"],
        )

        train_balance_removed = (
            before_balance - len(train)
        )

        sanitized_frames = {
            "environment": environment,
            "source_test": source_test,
            "calibration": calibration,
            "train": train,
        }

        final_hash_sets = {
            name: set(
                feature_hashes(
                    frame,
                    FEATURES,
                ).tolist()
            )
            for name, frame in sanitized_frames.items()
        }

        pairs = [
            ("train", "calibration"),
            ("train", "source_test"),
            ("train", "environment"),
            ("calibration", "source_test"),
            ("calibration", "environment"),
            ("source_test", "environment"),
        ]

        final_overlaps = {
            f"{first}|{second}": len(
                final_hash_sets[first]
                & final_hash_sets[second]
            )
            for first, second in pairs
        }

        if max(final_overlaps.values()) != 0:
            raise RuntimeError(
                f"{dataset}, seed {seed}: "
                "sanitization did not remove all overlap."
            )

        environment_counts = (
            environment.groupby([
                "actual_environment_id",
                "Label",
            ]).size()
        )

        if not (environment_counts == 2000).all():
            raise RuntimeError(
                f"{dataset}, seed {seed}: "
                "environment quota changed."
            )

        for role in [
            "train",
            "calibration",
            "source_test",
        ]:
            counts = (
                sanitized_frames[role]["Label"]
                .value_counts()
            )

            if counts.get(0, 0) != counts.get(1, 0):
                raise RuntimeError(
                    f"{dataset}, seed {seed}: "
                    f"{role} is not balanced."
                )

        output_files = {}

        for name, frame in sanitized_frames.items():
            output_file = output_dir / f"{name}.parquet"

            atomic_parquet(frame, output_file)

            output_files[name] = {
                "rows": int(len(frame)),
                "sha256": sha256_file(output_file),
                "path": str(output_file),
            }

        removal_counts = {
            "source_test_overlap_removed": int(
                test_overlap_removed
            ),
            "source_test_balance_removed": int(
                test_balance_removed
            ),
            "calibration_overlap_removed": int(
                cal_overlap_removed
            ),
            "calibration_balance_removed": int(
                cal_balance_removed
            ),
            "train_overlap_removed": int(
                train_overlap_removed
            ),
            "train_balance_removed": int(
                train_balance_removed
            ),
        }

        manifest = {
            "dataset": dataset,
            "seed": seed,
            "parent_protocol_sha256": (
                EXPECTED_PROTOCOL_HASH
            ),
            "amendment_A1_sha256": amendment_hash,
            "feature_schema_sha256": (
                EXPECTED_SCHEMA_HASH
            ),
            "original_rows": original_counts,
            "sanitized_rows": {
                name: int(len(frame))
                for name, frame
                in sanitized_frames.items()
            },
            "removal_counts": removal_counts,
            "final_cross_partition_overlap": (
                final_overlaps
            ),
            "maximum_final_overlap": 0,
            "environment_quota_preserved": True,
            "output_files": output_files,
            "status": "PASS",
        }

        manifest_file = (
            output_dir / "sanitization_manifest.json"
        )

        manifest_file.write_text(
            json.dumps(
                manifest,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        dataset_manifests[str(seed)] = manifest

        summary_row = {
            "dataset": dataset,
            "seed": seed,
            "train_rows": len(train),
            "calibration_rows": len(calibration),
            "source_test_rows": len(source_test),
            "environment_rows": len(environment),
            "overlap_removed": (
                test_overlap_removed
                + cal_overlap_removed
                + train_overlap_removed
            ),
            "final_overlap": 0,
        }

        overall_rows.append(summary_row)

        print(
            f"  Seed {seed}: "
            f"train={len(train):,}, "
            f"cal={len(calibration):,}, "
            f"test={len(source_test):,}, "
            f"environment={len(environment):,}, "
            f"overlap-removed="
            f"{summary_row['overlap_removed']:,}, "
            f"final-overlap=0 — PASS"
        )

    completion = {
        "dataset": dataset,
        "parent_protocol_sha256": (
            EXPECTED_PROTOCOL_HASH
        ),
        "amendment_A1_sha256": amendment_hash,
        "feature_schema_sha256": (
            EXPECTED_SCHEMA_HASH
        ),
        "seeds": dataset_manifests,
        "status": "PASS",
    }

    completion_file.write_text(
        json.dumps(
            completion,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print(f"Sanitization completed: {dataset}")


summary_table = pd.DataFrame(overall_rows)

summary_file = (
    SANITIZED_DIR / "sanitization_summary.csv"
)

summary_table.to_csv(
    summary_file,
    index=False,
)

print("\n" + "=" * 78)
print("ALL PAPER 3 PARTITIONS FEATURE-GROUP-DISJOINT: PASS")
print(f"Amendment A1 SHA-256: {amendment_hash}")
print(f"Summary: {summary_file}")
print("=" * 78)