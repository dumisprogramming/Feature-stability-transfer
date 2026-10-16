from pathlib import Path
import hashlib
import json
import time
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(
    "/content/drive/MyDrive/research/researchdata/"
    "NIDS_Research/Paper3_DataSet"
)

PROTOCOL_FILE = (
    ROOT / "Paper3_Protocol/paper3_protocol_frozen_v1.json"
)

PARTITION_DIR = ROOT / "Paper3_Partitions"
OUTPUT_DIR = ROOT / "Paper3_Materialized"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_PROTOCOL_HASH = (
    "e39a5007a7f2cbdc348d5179ce593af"
    "5766bc6688f5baa3a46a56399f683a488"
)

DATASETS = {
    "NF_BoT_IoT_v3": "NF-BoT-IoT-v3.zip",
    "NF_CSE_CIC_IDS2018_v3": "NF-CSE-CIC-IDS2018-v3.zip",
    "NF_ToN_IoT_v3": "NF-ToN-IoT-v3.zip",
    "NF_UNSW_NB15_v3": "NF-UNSW-NB15-v3.zip",
}

SEEDS = [2027, 2028, 2029, 2030, 2031]
CHUNK_SIZE = 250_000

EXCLUDED_COLUMNS = {
    "FLOW_START_MILLISECONDS",
    "FLOW_END_MILLISECONDS",
    "IPV4_SRC_ADDR",
    "IPV4_DST_ADDR",
    "Label",
    "Attack",
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


# ------------------------------------------------------------------
# 1. Verify frozen protocol
# ------------------------------------------------------------------

observed_protocol_hash = sha256_file(PROTOCOL_FILE)

if observed_protocol_hash != EXPECTED_PROTOCOL_HASH:
    raise RuntimeError(
        "Frozen protocol hash mismatch. Stop immediately."
    )

print("=" * 78)
print("PAPER 3 — FEATURE MATERIALIZATION AND OVERLAP AUDIT")
print("=" * 78)
print("Protocol verification: PASS")


# ------------------------------------------------------------------
# 2. Verify identical schema and freeze exact feature names
# ------------------------------------------------------------------

dataset_members = {}
dataset_features = {}

for dataset_name, zip_filename in DATASETS.items():
    zip_path = ROOT / zip_filename

    with zipfile.ZipFile(zip_path, "r") as archive:
        members = [
            member for member in archive.namelist()
            if member.lower().endswith(".csv")
            and "netflow_v3_features" not in member.lower()
            and not member.startswith("__MACOSX")
        ]

        csv_member = max(
            members,
            key=lambda member: archive.getinfo(member).file_size,
        )

        dataset_members[dataset_name] = csv_member

        with archive.open(csv_member) as csv_file:
            header = pd.read_csv(
                csv_file,
                nrows=0,
            ).columns.tolist()

        model_features = [
            column for column in header
            if column not in EXCLUDED_COLUMNS
        ]

        if len(model_features) != 49:
            raise RuntimeError(
                f"{dataset_name}: expected 49 predictors, "
                f"found {len(model_features)}."
            )

        dataset_features[dataset_name] = model_features

reference_features = dataset_features[
    "NF_BoT_IoT_v3"
]

for dataset_name, features in dataset_features.items():
    if features != reference_features:
        raise RuntimeError(
            f"Feature schema mismatch: {dataset_name}"
        )

feature_schema = {
    "schema_id": "P3_NF_V3_49_FEATURES_V1",
    "protocol_sha256": EXPECTED_PROTOCOL_HASH,
    "predictor_count": len(reference_features),
    "predictors_in_order": reference_features,
    "excluded_columns": sorted(EXCLUDED_COLUMNS),
}

schema_text = json.dumps(
    feature_schema,
    indent=2,
    sort_keys=True,
)

schema_file = OUTPUT_DIR / "feature_schema_v1.json"
schema_file.write_text(schema_text, encoding="utf-8")

schema_hash = sha256_file(schema_file)

(OUTPUT_DIR / "feature_schema_v1.sha256").write_text(
    schema_hash,
    encoding="utf-8",
)

print(f"Feature count: {len(reference_features)}")
print(f"Feature-schema SHA-256: {schema_hash}")
print("Cross-dataset feature-schema verification: PASS")


# ------------------------------------------------------------------
# 3. Materialize selected rows
# ------------------------------------------------------------------

for dataset_name, zip_filename in DATASETS.items():

    dataset_start = time.time()

    source_partition_dir = (
        PARTITION_DIR / dataset_name
    )

    dataset_output_dir = OUTPUT_DIR / dataset_name
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    completion_file = (
        dataset_output_dir / "materialization_complete.json"
    )

    if completion_file.exists():
        print(f"\nSKIPPED—already completed: {dataset_name}")
        continue

    print(f"\n{'-' * 78}")
    print(f"DATASET: {dataset_name}")
    print(f"{'-' * 78}")

    seed_indices = {}
    all_selected_ids = []

    for seed in SEEDS:
        seed_index_dir = (
            source_partition_dir / f"seed_{seed}"
        )

        environment_index = pd.read_csv(
            seed_index_dir / "environment_index.csv"
        )

        global_index = pd.read_csv(
            seed_index_dir / "global_index.csv"
        )

        environment_index["row_id"] = (
            environment_index["row_id"].astype("int64")
        )

        global_index["row_id"] = (
            global_index["row_id"].astype("int64")
        )

        seed_indices[seed] = {
            "environment": environment_index,
            "global": global_index,
        }

        all_selected_ids.append(
            environment_index["row_id"].to_numpy()
        )

        all_selected_ids.append(
            global_index["row_id"].to_numpy()
        )

    selected_union = np.unique(
        np.concatenate(all_selected_ids)
        .astype("int64")
    )

    print(
        f"Unique rows required across five seeds: "
        f"{len(selected_union):,}"
    )

    zip_path = ROOT / zip_filename
    csv_member = dataset_members[dataset_name]

    selected_parts = []
    processed_rows = 0

    with zipfile.ZipFile(zip_path, "r") as archive:
        with archive.open(csv_member) as csv_file:

            reader = pd.read_csv(
                csv_file,
                usecols=reference_features + ["Label"],
                chunksize=CHUNK_SIZE,
                low_memory=False,
            )

            for chunk_number, chunk in enumerate(
                reader,
                start=1,
            ):
                chunk_start = processed_rows
                chunk_end = processed_rows + len(chunk)

                left = np.searchsorted(
                    selected_union,
                    chunk_start,
                    side="left",
                )

                right = np.searchsorted(
                    selected_union,
                    chunk_end,
                    side="left",
                )

                required_ids = selected_union[left:right]

                if len(required_ids) > 0:
                    positions = (
                        required_ids - chunk_start
                    ).astype("int64")

                    selected_chunk = (
                        chunk.iloc[positions].copy()
                    )

                    selected_chunk.insert(
                        0,
                        "row_id",
                        required_ids,
                    )

                    selected_parts.append(selected_chunk)

                processed_rows = chunk_end

                if chunk_number % 10 == 0:
                    elapsed = (
                        time.time() - dataset_start
                    ) / 60

                    print(
                        f"  Chunks: {chunk_number:,} | "
                        f"Rows scanned: {processed_rows:,} | "
                        f"Elapsed: {elapsed:.1f} min"
                    )

    materialized_union = pd.concat(
        selected_parts,
        ignore_index=True,
    )

    materialized_union = (
        materialized_union
        .sort_values("row_id")
        .reset_index(drop=True)
    )

    if materialized_union["row_id"].duplicated().any():
        raise RuntimeError(
            f"{dataset_name}: duplicate materialized row IDs."
        )

    observed_ids = materialized_union[
        "row_id"
    ].to_numpy(dtype="int64")

    if not np.array_equal(
        observed_ids,
        selected_union,
    ):
        raise RuntimeError(
            f"{dataset_name}: selected row-ID mismatch."
        )

    # --------------------------------------------------------------
    # 4. Numeric conversion and quality audit
    # --------------------------------------------------------------

    feature_quality = {}
    total_infinite = 0
    total_coercion_missing = 0

    for feature in reference_features:
        original_missing = int(
            materialized_union[feature].isna().sum()
        )

        numeric_values = pd.to_numeric(
            materialized_union[feature],
            errors="coerce",
        )

        numeric_array = numeric_values.to_numpy(
            dtype=np.float64,
            na_value=np.nan,
        )

        infinite_mask = np.isinf(numeric_array)
        infinite_count = int(infinite_mask.sum())

        numeric_array[infinite_mask] = np.nan

        final_missing = int(
            np.isnan(numeric_array).sum()
        )

        coercion_missing = max(
            0,
            final_missing
            - original_missing
            - infinite_count,
        )

        materialized_union[feature] = (
            numeric_array.astype(np.float32)
        )

        feature_quality[feature] = {
            "original_missing": original_missing,
            "infinite_converted_to_missing": (
                infinite_count
            ),
            "non_numeric_converted_to_missing": (
                coercion_missing
            ),
            "final_missing": final_missing,
        }

        total_infinite += infinite_count
        total_coercion_missing += coercion_missing

    labels = pd.to_numeric(
        materialized_union["Label"],
        errors="coerce",
    )

    if labels.isna().any():
        raise RuntimeError(
            f"{dataset_name}: missing labels after conversion."
        )

    if not labels.isin([0, 1]).all():
        raise RuntimeError(
            f"{dataset_name}: invalid binary labels."
        )

    materialized_union["Label"] = (
        labels.astype(np.int8)
    )

    raw_label_lookup = (
        materialized_union
        .set_index("row_id")["Label"]
    )

    seed_manifests = {}

    # --------------------------------------------------------------
    # 5. Produce seed-specific Parquet files
    # --------------------------------------------------------------

    for seed in SEEDS:
        seed_output_dir = (
            dataset_output_dir / f"seed_{seed}"
        )
        seed_output_dir.mkdir(parents=True, exist_ok=True)

        environment_index = seed_indices[
            seed
        ]["environment"].copy()

        global_index = seed_indices[
            seed
        ]["global"].copy()

        expected_environment_labels = (
            environment_index["Label"]
            .astype(np.int8)
            .to_numpy()
        )

        observed_environment_labels = (
            raw_label_lookup.loc[
                environment_index["row_id"]
            ].to_numpy()
        )

        if not np.array_equal(
            expected_environment_labels,
            observed_environment_labels,
        ):
            raise RuntimeError(
                f"{dataset_name}, seed {seed}: "
                "environment label mismatch."
            )

        expected_global_labels = (
            global_index["Label"]
            .astype(np.int8)
            .to_numpy()
        )

        observed_global_labels = (
            raw_label_lookup.loc[
                global_index["row_id"]
            ].to_numpy()
        )

        if not np.array_equal(
            expected_global_labels,
            observed_global_labels,
        ):
            raise RuntimeError(
                f"{dataset_name}, seed {seed}: "
                "global label mismatch."
            )

        environment_data = (
            environment_index.drop(columns=["Label"])
            .merge(
                materialized_union,
                on="row_id",
                how="left",
                validate="one_to_one",
            )
        )

        global_data = (
            global_index.drop(columns=["Label"])
            .merge(
                materialized_union,
                on="row_id",
                how="left",
                validate="one_to_one",
            )
        )

        train_data = (
            global_data[
                global_data["role"] == "train"
            ]
            .drop(columns=["role"])
            .reset_index(drop=True)
        )

        calibration_data = (
            global_data[
                global_data["role"] == "calibration"
            ]
            .drop(columns=["role"])
            .reset_index(drop=True)
        )

        source_test_data = (
            global_data[
                global_data["role"] == "source_test"
            ]
            .drop(columns=["role"])
            .reset_index(drop=True)
        )

        environment_data = (
            environment_data.reset_index(drop=True)
        )

        expected_counts = {
            "train": 50_000,
            "calibration": 10_000,
            "source_test": 10_000,
        }

        observed_counts = {
            "train": len(train_data),
            "calibration": len(calibration_data),
            "source_test": len(source_test_data),
        }

        if observed_counts != expected_counts:
            raise RuntimeError(
                f"{dataset_name}, seed {seed}: "
                f"unexpected role counts {observed_counts}"
            )

        partition_frames = {
            "train": train_data,
            "calibration": calibration_data,
            "source_test": source_test_data,
            "environment": environment_data,
        }

        hash_sets = {}
        duplicate_rows = {}

        for partition_name, frame in partition_frames.items():
            hashes = feature_hashes(
                frame,
                reference_features,
            )

            hash_sets[partition_name] = set(
                hashes.tolist()
            )

            duplicate_rows[partition_name] = int(
                len(hashes) - len(np.unique(hashes))
            )

        partition_pairs = [
            ("train", "calibration"),
            ("train", "source_test"),
            ("train", "environment"),
            ("calibration", "source_test"),
            ("calibration", "environment"),
            ("source_test", "environment"),
        ]

        overlap_counts = {}

        for first, second in partition_pairs:
            key = f"{first}|{second}"

            overlap_counts[key] = len(
                hash_sets[first].intersection(
                    hash_sets[second]
                )
            )

        output_files = {}

        for partition_name, frame in partition_frames.items():
            output_file = (
                seed_output_dir
                / f"{partition_name}.parquet"
            )

            atomic_parquet(frame, output_file)

            output_files[partition_name] = {
                "path": str(output_file),
                "rows": int(len(frame)),
                "sha256": sha256_file(output_file),
            }

        seed_manifest = {
            "dataset": dataset_name,
            "seed": seed,
            "protocol_sha256": EXPECTED_PROTOCOL_HASH,
            "feature_schema_sha256": schema_hash,
            "output_files": output_files,
            "within_partition_duplicate_rows": (
                duplicate_rows
            ),
            "cross_partition_feature_hash_overlap": (
                overlap_counts
            ),
            "maximum_cross_partition_overlap": int(
                max(overlap_counts.values())
            ),
            "status": "PASS",
        }

        manifest_file = (
            seed_output_dir
            / "materialization_manifest.json"
        )

        manifest_file.write_text(
            json.dumps(
                seed_manifest,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        seed_manifests[str(seed)] = seed_manifest

        print(
            f"  Seed {seed}: "
            f"train={len(train_data):,}, "
            f"cal={len(calibration_data):,}, "
            f"test={len(source_test_data):,}, "
            f"environment={len(environment_data):,}, "
            f"max-overlap="
            f"{max(overlap_counts.values()):,}"
        )

    quality_file = (
        dataset_output_dir
        / "feature_quality_audit.json"
    )

    quality_summary = {
        "dataset": dataset_name,
        "protocol_sha256": EXPECTED_PROTOCOL_HASH,
        "feature_schema_sha256": schema_hash,
        "scanned_source_rows": int(processed_rows),
        "unique_materialized_rows": int(
            len(materialized_union)
        ),
        "total_infinite_values_converted": int(
            total_infinite
        ),
        "total_non_numeric_values_converted": int(
            total_coercion_missing
        ),
        "per_feature": feature_quality,
    }

    quality_file.write_text(
        json.dumps(
            quality_summary,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    completion_summary = {
        "dataset": dataset_name,
        "protocol_sha256": EXPECTED_PROTOCOL_HASH,
        "feature_schema_sha256": schema_hash,
        "unique_materialized_rows": int(
            len(materialized_union)
        ),
        "quality_audit_sha256": sha256_file(
            quality_file
        ),
        "seeds": seed_manifests,
        "status": "PASS",
    }

    completion_file.write_text(
        json.dumps(
            completion_summary,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    elapsed = (time.time() - dataset_start) / 60

    print(f"COMPLETED: {dataset_name}")
    print(
        f"Unique materialized rows: "
        f"{len(materialized_union):,}"
    )
    print(
        f"Infinite values converted: "
        f"{total_infinite:,}"
    )
    print(
        f"Non-numeric values converted: "
        f"{total_coercion_missing:,}"
    )
    print(f"Elapsed: {elapsed:.1f} minutes")
    print("Materialization status: PASS")


print("\n" + "=" * 78)
print("ALL PAPER 3 FEATURE MATERIALIZATION: PASS")
print(f"Output directory: {OUTPUT_DIR}")
print("=" * 78)