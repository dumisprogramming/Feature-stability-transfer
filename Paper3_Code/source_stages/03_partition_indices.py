from pathlib import Path
from collections import defaultdict
import hashlib
import json
import zipfile
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(
    "/content/drive/MyDrive/research/researchdata/"
    "NIDS_Research/Paper3_DataSet"
)

AUDIT_DIR = ROOT / "Paper3_Audit" / "temporal_audit"
PROTOCOL_DIR = ROOT / "Paper3_Protocol"
PARTITION_DIR = ROOT / "Paper3_Partitions"
PARTITION_DIR.mkdir(parents=True, exist_ok=True)

PROTOCOL_FILE = (
    PROTOCOL_DIR / "paper3_protocol_frozen_v1.json"
)

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

CHUNK_SIZE = 500_000
BLOCK_MS = 6 * 60 * 60 * 1000

ENVIRONMENT_PER_CLASS = 2_000
TRAIN_PER_CLASS = 25_000
CALIBRATION_PER_CLASS = 5_000
SOURCE_TEST_PER_CLASS = 5_000

GLOBAL_REQUIRED = (
    TRAIN_PER_CLASS
    + CALIBRATION_PER_CLASS
    + SOURCE_TEST_PER_CLASS
)

SALT_ENV = np.uint64(0xA24BAED4963EE407)
SALT_GLOBAL = np.uint64(0x9FB21C651E98DF25)
SALT_PSEUDO = np.uint64(0xC13FA9A902A6328F)


def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for part in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(part)

    return digest.hexdigest()


def splitmix64(values):
    values = values.astype(np.uint64, copy=False)

    with np.errstate(over="ignore"):
        values = values + np.uint64(0x9E3779B97F4A7C15)
        values = (
            values ^ np.right_shift(values, 30)
        ) * np.uint64(0xBF58476D1CE4E5B9)
        values = (
            values ^ np.right_shift(values, 27)
        ) * np.uint64(0x94D049BB133111EB)
        values = values ^ np.right_shift(values, 31)

    return values


def priorities(row_ids, seed, salt):
    values = (
        row_ids.astype(np.uint64, copy=False)
        ^ np.uint64(seed)
        ^ salt
    )
    return splitmix64(values)


def merge_smallest(existing, new_priorities, new_ids, keep):
    if len(new_ids) == 0:
        return existing

    new_priorities = np.asarray(
        new_priorities,
        dtype=np.uint64,
    )
    new_ids = np.asarray(new_ids, dtype=np.int64)

    if existing is None:
        combined_priorities = new_priorities
        combined_ids = new_ids
    else:
        combined_priorities = np.concatenate([
            existing[0],
            new_priorities,
        ])
        combined_ids = np.concatenate([
            existing[1],
            new_ids,
        ])

    if len(combined_ids) > keep:
        selected = np.argpartition(
            combined_priorities,
            keep - 1,
        )[:keep]

        combined_priorities = combined_priorities[selected]
        combined_ids = combined_ids[selected]

    return combined_priorities, combined_ids


def sorted_smallest(item, keep):
    priority_values, row_ids = item

    order = np.lexsort((
        row_ids,
        priority_values,
    ))

    order = order[:keep]

    return (
        priority_values[order],
        row_ids[order],
    )


# Verify that the frozen protocol has not changed
actual_protocol_hash = sha256_file(PROTOCOL_FILE)

print("=" * 78)
print("PAPER 3 — DETERMINISTIC PARTITION-INDEX CONSTRUCTION")
print("=" * 78)
print(f"Expected protocol hash: {EXPECTED_PROTOCOL_HASH}")
print(f"Observed protocol hash: {actual_protocol_hash}")

if actual_protocol_hash != EXPECTED_PROTOCOL_HASH:
    raise RuntimeError(
        "Protocol hash mismatch. Stop: the frozen protocol "
        "may have been modified."
    )

print("Protocol verification: PASS")


for dataset_name, zip_filename in DATASETS.items():

    dataset_output = PARTITION_DIR / dataset_name
    dataset_output.mkdir(parents=True, exist_ok=True)

    completion_file = (
        dataset_output / "partition_index_complete.json"
    )

    if completion_file.exists():
        print(f"\nSKIPPED—already completed: {dataset_name}")
        continue

    block_count_file = (
        AUDIT_DIR / f"{dataset_name}_6hour_counts.csv"
    )

    block_table = pd.read_csv(block_count_file)
    eligible_table = block_table[
        block_table["eligible"].astype(bool)
    ].copy()

    eligible_table["block_datetime"] = pd.to_datetime(
        eligible_table["period"],
        utc=True,
    )

    eligible_table["block_id_ms"] = (
        eligible_table["block_datetime"].astype("int64")
        // 1_000_000
    ).astype("int64")

    eligible_blocks = sorted(
        eligible_table["block_id_ms"].unique().tolist()
    )

    eligible_keys = {
        (int(block_id), int(label))
        for block_id in eligible_blocks
        for label in [0, 1]
    }

    number_of_blocks = len(eligible_blocks)

    # Extra candidates guarantee that enough global rows remain
    # after all selected environment rows are excluded.
    global_candidate_limit = (
        GLOBAL_REQUIRED
        + number_of_blocks * ENVIRONMENT_PER_CLASS
    )

    env_top = {
        seed: {} for seed in SEEDS
    }

    global_top = {
        seed: {
            0: None,
            1: None,
        }
        for seed in SEEDS
    }

    zip_path = ROOT / zip_filename
    processed_rows = 0
    start_time = time.time()

    print(f"\n{'-' * 78}")
    print(f"DATASET: {dataset_name}")
    print(f"Eligible blocks: {number_of_blocks}")
    print(
        "Global candidate limit per class: "
        f"{global_candidate_limit:,}"
    )
    print(f"{'-' * 78}")

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

        print(f"CSV member: {csv_member}")

        with archive.open(csv_member) as csv_file:

            reader = pd.read_csv(
                csv_file,
                usecols=[
                    "FLOW_START_MILLISECONDS",
                    "Label",
                ],
                chunksize=CHUNK_SIZE,
                low_memory=False,
            )

            for chunk_number, chunk in enumerate(
                reader,
                start=1,
            ):
                chunk_length = len(chunk)

                row_ids = np.arange(
                    processed_rows,
                    processed_rows + chunk_length,
                    dtype=np.int64,
                )

                start_values = pd.to_numeric(
                    chunk["FLOW_START_MILLISECONDS"],
                    errors="coerce",
                ).fillna(-1).astype("int64").to_numpy()

                labels = pd.to_numeric(
                    chunk["Label"],
                    errors="coerce",
                ).fillna(-1).astype("int8").to_numpy()

                block_ids = (
                    start_values // BLOCK_MS
                ) * BLOCK_MS

                eligible_mask = (
                    np.isin(block_ids, eligible_blocks)
                    & np.isin(labels, [0, 1])
                )

                eligible_positions = np.flatnonzero(
                    eligible_mask
                )

                grouped_positions = {}

                if len(eligible_positions) > 0:
                    grouping_frame = pd.DataFrame({
                        "position": eligible_positions,
                        "block": block_ids[
                            eligible_positions
                        ],
                        "label": labels[
                            eligible_positions
                        ],
                    })

                    for key, group in grouping_frame.groupby(
                        ["block", "label"],
                        sort=False,
                    ):
                        normalized_key = (
                            int(key[0]),
                            int(key[1]),
                        )

                        if normalized_key in eligible_keys:
                            grouped_positions[
                                normalized_key
                            ] = group[
                                "position"
                            ].to_numpy()

                for seed in SEEDS:
                    env_priorities = priorities(
                        row_ids,
                        seed,
                        SALT_ENV,
                    )

                    global_priorities = priorities(
                        row_ids,
                        seed,
                        SALT_GLOBAL,
                    )

                    for key, positions in grouped_positions.items():
                        existing = env_top[seed].get(key)

                        env_top[seed][key] = merge_smallest(
                            existing=existing,
                            new_priorities=env_priorities[
                                positions
                            ],
                            new_ids=row_ids[positions],
                            keep=ENVIRONMENT_PER_CLASS,
                        )

                    for label in [0, 1]:
                        positions = np.flatnonzero(
                            labels == label
                        )

                        global_top[seed][label] = merge_smallest(
                            existing=global_top[seed][label],
                            new_priorities=global_priorities[
                                positions
                            ],
                            new_ids=row_ids[positions],
                            keep=global_candidate_limit,
                        )

                processed_rows += chunk_length

                if chunk_number % 10 == 0:
                    elapsed = (
                        time.time() - start_time
                    ) / 60

                    print(
                        f"  Chunks: {chunk_number:,} | "
                        f"Rows: {processed_rows:,} | "
                        f"Elapsed: {elapsed:.1f} min"
                    )

    seed_manifests = {}

    for seed in SEEDS:
        seed_directory = dataset_output / f"seed_{seed}"
        seed_directory.mkdir(parents=True, exist_ok=True)

        environment_frames = []

        for block_id in eligible_blocks:
            for label in [0, 1]:
                key = (int(block_id), label)

                if key not in env_top[seed]:
                    raise RuntimeError(
                        f"Missing environment group: "
                        f"{dataset_name}, seed={seed}, key={key}"
                    )

                _, selected_ids = sorted_smallest(
                    env_top[seed][key],
                    ENVIRONMENT_PER_CLASS,
                )

                if len(selected_ids) != ENVIRONMENT_PER_CLASS:
                    raise RuntimeError(
                        f"Insufficient environment rows for {key}"
                    )

                environment_frames.append(
                    pd.DataFrame({
                        "row_id": selected_ids,
                        "Label": label,
                        "block_id_ms": int(block_id),
                    })
                )

        environment_index = pd.concat(
            environment_frames,
            ignore_index=True,
        )

        environment_row_ids = (
            environment_index["row_id"]
            .astype("int64")
            .to_numpy()
        )

        # Create matched random pseudo-environments using exactly
        # the same observations as the temporal environments.
        environment_index[
            "pseudo_environment_id"
        ] = -1

        for label in [0, 1]:
            label_positions = np.flatnonzero(
                environment_index["Label"].to_numpy()
                == label
            )

            label_row_ids = (
                environment_index.iloc[
                    label_positions
                ]["row_id"]
                .astype("int64")
                .to_numpy()
            )

            pseudo_priorities = priorities(
                label_row_ids,
                seed,
                SALT_PSEUDO,
            )

            pseudo_order = np.lexsort((
                label_row_ids,
                pseudo_priorities,
            ))

            ordered_positions = label_positions[
                pseudo_order
            ]

            pseudo_values = np.repeat(
                np.arange(number_of_blocks),
                ENVIRONMENT_PER_CLASS,
            )

            environment_index.loc[
                ordered_positions,
                "pseudo_environment_id",
            ] = pseudo_values

        environment_index[
            "actual_environment_id"
        ] = pd.to_datetime(
            environment_index["block_id_ms"],
            unit="ms",
            utc=True,
        ).dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Select disjoint train/calibration/source-test rows.
        global_frames = []

        for label in [0, 1]:
            priority_values, candidate_ids = (
                global_top[seed][label]
            )

            not_environment = ~np.isin(
                candidate_ids,
                environment_row_ids,
            )

            priority_values = priority_values[
                not_environment
            ]
            candidate_ids = candidate_ids[
                not_environment
            ]

            priority_values, candidate_ids = sorted_smallest(
                (priority_values, candidate_ids),
                GLOBAL_REQUIRED,
            )

            if len(candidate_ids) < GLOBAL_REQUIRED:
                raise RuntimeError(
                    f"Insufficient global rows: "
                    f"{dataset_name}, seed={seed}, label={label}"
                )

            roles = (
                ["train"] * TRAIN_PER_CLASS
                + ["calibration"] * CALIBRATION_PER_CLASS
                + ["source_test"] * SOURCE_TEST_PER_CLASS
            )

            global_frames.append(
                pd.DataFrame({
                    "row_id": candidate_ids,
                    "Label": label,
                    "role": roles,
                })
            )

        global_index = pd.concat(
            global_frames,
            ignore_index=True,
        )

        # Integrity checks
        env_set = set(
            environment_index["row_id"].astype(int)
        )

        global_set = set(
            global_index["row_id"].astype(int)
        )

        if env_set.intersection(global_set):
            raise RuntimeError(
                f"Environment/global overlap for seed {seed}"
            )

        if global_index["row_id"].duplicated().any():
            raise RuntimeError(
                f"Duplicate global row IDs for seed {seed}"
            )

        expected_environment_rows = (
            number_of_blocks
            * 2
            * ENVIRONMENT_PER_CLASS
        )

        if len(environment_index) != expected_environment_rows:
            raise RuntimeError(
                "Unexpected environment sample size."
            )

        environment_index = environment_index.sort_values([
            "actual_environment_id",
            "Label",
            "row_id",
        ]).reset_index(drop=True)

        global_index = global_index.sort_values([
            "role",
            "Label",
            "row_id",
        ]).reset_index(drop=True)

        environment_file = (
            seed_directory / "environment_index.csv"
        )

        global_file = (
            seed_directory / "global_index.csv"
        )

        environment_index.to_csv(
            environment_file,
            index=False,
        )

        global_index.to_csv(
            global_file,
            index=False,
        )

        role_counts = (
            global_index.groupby(
                ["role", "Label"]
            ).size().to_dict()
        )

        seed_manifest = {
            "dataset": dataset_name,
            "seed": seed,
            "source_csv_rows": int(processed_rows),
            "eligible_environment_blocks": number_of_blocks,
            "environment_rows": int(
                len(environment_index)
            ),
            "train_rows": int(
                (global_index["role"] == "train").sum()
            ),
            "calibration_rows": int(
                (
                    global_index["role"]
                    == "calibration"
                ).sum()
            ),
            "source_test_rows": int(
                (
                    global_index["role"]
                    == "source_test"
                ).sum()
            ),
            "environment_global_overlap": 0,
            "environment_index_sha256": sha256_file(
                environment_file
            ),
            "global_index_sha256": sha256_file(
                global_file
            ),
            "protocol_sha256": EXPECTED_PROTOCOL_HASH,
            "role_label_counts": {
                f"{role}|{label}": int(count)
                for (role, label), count
                in role_counts.items()
            },
        }

        manifest_file = (
            seed_directory / "partition_manifest.json"
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
            f"environment={len(environment_index):,}, "
            f"train=50,000, calibration=10,000, "
            f"source_test=10,000 — PASS"
        )

    completion = {
        "dataset": dataset_name,
        "protocol_sha256": EXPECTED_PROTOCOL_HASH,
        "source_rows": int(processed_rows),
        "eligible_blocks": number_of_blocks,
        "seeds": seed_manifests,
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

    elapsed = (time.time() - start_time) / 60

    print(f"COMPLETED: {dataset_name}")
    print(f"Elapsed: {elapsed:.1f} minutes")
    print("Partition-index status: PASS")


print("\n" + "=" * 78)
print("ALL PAPER 3 PARTITION INDICES: PASS")
print(f"Output directory: {PARTITION_DIR}")
print("=" * 78)