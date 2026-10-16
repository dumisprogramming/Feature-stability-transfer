from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
import zipfile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(
    "/content/drive/MyDrive/research/researchdata/"
    "NIDS_Research/Paper3_DataSet"
)

EXPECTED_A2 = (
    "2217078acfa10700f4fae29c6f32f171"
    "ce7a4e2ecb22d22ce5eabb6087e5aa5e"
)

A2_PATH = (
    ROOT / "Paper3_Protocol" /
    "amendment_A2_target_evaluation_plan.json"
)
SANITIZED_ROOT = ROOT / "Paper3_Sanitized"
PANEL_ROOT = ROOT / "Paper3_Target_Panels"
CANDIDATE_ROOT = PANEL_ROOT / "candidates"
LABEL_VAULT_ROOT = ROOT / "Paper3_Target_Label_Vault"

SELECTION_SEED = 42027
ROWS_PER_CLASS = 50_000
CHUNK_SIZE = 250_000

DATASETS = {
    "NF_BoT_IoT_v3": {
        "zip": "NF-BoT-IoT-v3.zip",
        "csv_suffix": "/data/NF-BoT-IoT-v3.csv",
    },
    "NF_CSE_CIC_IDS2018_v3": {
        "zip": "NF-CSE-CIC-IDS2018-v3.zip",
        "csv_suffix": "/data/NF-CICIDS2018-v3.csv",
    },
    "NF_ToN_IoT_v3": {
        "zip": "NF-ToN-IoT-v3.zip",
        "csv_suffix": "/data/NF-ToN-IoT-v3.csv",
    },
    "NF_UNSW_NB15_v3": {
        "zip": "NF-UNSW-NB15-v3.zip",
        "csv_suffix": "/data/NF-UNSW-NB15-v3.csv",
    },
}


def sha256_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def canonical_json_bytes(obj):
    return json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def atomic_write_bytes(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_json(path, obj):
    payload = json.dumps(
        obj,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    atomic_write_bytes(path, payload)


def atomic_to_parquet(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def splitmix64(values):
    """Deterministic uint64 mixing; overflow is intentional."""
    x = np.asarray(values, dtype=np.uint64).copy()
    with np.errstate(over="ignore"):
        x += np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> np.uint64(31))
    return x


def dataset_salt(dataset):
    raw = hashlib.sha256(
        f"paper3|{dataset}|{SELECTION_SEED}".encode("utf-8")
    ).digest()[:8]
    return np.uint64(int.from_bytes(raw, "little", signed=False))


def selection_keys(dataset, row_ids):
    return splitmix64(
        np.asarray(row_ids, dtype=np.uint64) ^ dataset_salt(dataset)
    )


def get_feature_schema():
    reference = SANITIZED_ROOT / "NF_BoT_IoT_v3" / "seed_2027" / "train.parquet"
    assert reference.exists(), f"Missing schema reference: {reference}"
    columns = list(pq.ParquetFile(reference).schema_arrow.names)
    features = [c for c in columns if c not in {"row_id", "Label"}]
    assert len(features) == 49, f"Expected 49 predictors, observed {len(features)}"

    for dataset in DATASETS:
        for seed in range(2027, 2032):
            for partition in ("train", "calibration", "source_test", "environment"):
                path = SANITIZED_ROOT / dataset / f"seed_{seed}" / f"{partition}.parquet"
                assert path.exists(), f"Missing sanitized partition: {path}"
                observed = list(pq.ParquetFile(path).schema_arrow.names)
                observed_features = [
                    c for c in observed
                    if c not in {
                        "row_id", "Label", "block_id_ms",
                        "pseudo_environment_id", "actual_environment_id",
                    }
                ]
                assert observed_features == features, (
                    f"Feature order mismatch: {dataset}, seed {seed}, {partition}"
                )
    return features


def locate_csv_member(zip_path, suffix):
    with zipfile.ZipFile(zip_path) as zf:
        matches = [
            name for name in zf.namelist()
            if name.endswith(suffix.lstrip("/")) or name.endswith(suffix)
        ]
    assert len(matches) == 1, (
        f"Expected one CSV ending in {suffix}; observed {matches}"
    )
    return matches[0]


def retain_smallest(existing, incoming, limit):
    if existing is None:
        combined = incoming
    else:
        combined = pd.concat([existing, incoming], ignore_index=True)
    combined = combined.sort_values(
        ["selection_key", "row_id"], kind="mergesort"
    ).head(limit)
    return combined.reset_index(drop=True)


def build_candidate(dataset, config, features):
    zip_path = ROOT / config["zip"]
    assert zip_path.exists(), f"Missing ZIP: {zip_path}"
    member = locate_csv_member(zip_path, config["csv_suffix"])

    feature_path = CANDIDATE_ROOT / dataset / "target_candidate_features.parquet"
    label_path = LABEL_VAULT_ROOT / dataset / "target_candidate_labels.parquet"
    manifest_path = CANDIDATE_ROOT / dataset / "candidate_manifest.json"

    existing = [feature_path.exists(), label_path.exists(), manifest_path.exists()]
    if any(existing):
        assert all(existing), (
            f"Partial candidate artifacts exist for {dataset}. Stop and audit them."
        )
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        assert sha256_file(feature_path) == manifest["feature_file_sha256"]
        assert sha256_file(label_path) == manifest["label_vault_sha256"]
        fdf = pd.read_parquet(feature_path)
        ldf = pd.read_parquet(label_path)
        assert list(fdf.columns) == ["row_id"] + features
        assert list(ldf.columns) == ["row_id", "Label"]
        assert len(fdf) == 2 * ROWS_PER_CLASS == len(ldf)
        assert fdf["row_id"].equals(ldf["row_id"])
        counts = ldf["Label"].value_counts().to_dict()
        assert counts == {0: ROWS_PER_CLASS, 1: ROWS_PER_CLASS}
        print(f"  {dataset}: REUSED—sealed candidate artifacts verified")
        return manifest

    reservoirs = {0: None, 1: None}
    offset = 0
    chunks = 0
    invalid_labels = 0
    inf_converted = 0

    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member) as stream:
            reader = pd.read_csv(
                stream,
                usecols=features + ["Label"],
                chunksize=CHUNK_SIZE,
                low_memory=False,
            )

            for chunk in reader:
                n = len(chunk)
                row_ids = np.arange(offset, offset + n, dtype=np.int64)
                keys = selection_keys(dataset, row_ids)
                labels = pd.to_numeric(chunk["Label"], errors="coerce").to_numpy()
                invalid_labels += int((~np.isin(labels, [0, 1])).sum())

                for label in (0, 1):
                    positions = np.flatnonzero(labels == label)
                    if positions.size == 0:
                        continue

                    if positions.size > ROWS_PER_CLASS:
                        local_keys = keys[positions]
                        keep = np.argpartition(
                            local_keys, ROWS_PER_CLASS - 1
                        )[:ROWS_PER_CLASS]
                        positions = positions[keep]

                    selected = chunk.iloc[positions][features].copy()
                    for column in features:
                        selected[column] = pd.to_numeric(
                            selected[column], errors="coerce"
                        ).astype("float64")

                    values = selected.to_numpy(dtype=np.float64, copy=False)
                    inf_mask = np.isinf(values)
                    inf_converted += int(inf_mask.sum())
                    if inf_mask.any():
                        values[inf_mask] = np.nan
                        selected.loc[:, features] = values

                    selected.insert(0, "row_id", row_ids[positions])
                    selected.insert(1, "selection_key", keys[positions])
                    selected.insert(2, "Label", np.int8(label))

                    reservoirs[label] = retain_smallest(
                        reservoirs[label], selected, ROWS_PER_CLASS
                    )

                offset += n
                chunks += 1
                if chunks % 10 == 0:
                    print(
                        f"    {dataset}: chunks={chunks}, rows={offset:,}, "
                        f"kept0={len(reservoirs[0]) if reservoirs[0] is not None else 0:,}, "
                        f"kept1={len(reservoirs[1]) if reservoirs[1] is not None else 0:,}"
                    )

    assert invalid_labels == 0, f"Invalid labels found in {dataset}: {invalid_labels}"
    assert reservoirs[0] is not None and len(reservoirs[0]) == ROWS_PER_CLASS
    assert reservoirs[1] is not None and len(reservoirs[1]) == ROWS_PER_CLASS

    panel = pd.concat([reservoirs[0], reservoirs[1]], ignore_index=True)
    panel = panel.sort_values("row_id", kind="mergesort").reset_index(drop=True)
    assert panel["row_id"].is_unique
    assert panel["Label"].value_counts().to_dict() == {
        0: ROWS_PER_CLASS,
        1: ROWS_PER_CLASS,
    }

    feature_frame = panel[["row_id"] + features].copy()
    label_frame = panel[["row_id", "Label"]].copy()
    label_frame["Label"] = label_frame["Label"].astype("int8")

    # The prediction stage receives only feature_path. Labels stay in the vault.
    atomic_to_parquet(feature_frame, feature_path)
    atomic_to_parquet(label_frame, label_path)

    feature_sha = sha256_file(feature_path)
    label_sha = sha256_file(label_path)

    manifest = {
        "artifact": "paper3_target_candidate_panel",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "parent_amendment_A2_sha256": EXPECTED_A2,
        "selection_seed": SELECTION_SEED,
        "selection_algorithm": (
            "lowest SplitMix64(dataset-specific SHA256 salt XOR zero-based raw row_id); "
            "ties resolved by row_id"
        ),
        "rows_scanned": int(offset),
        "chunks_scanned": int(chunks),
        "rows_per_class": ROWS_PER_CLASS,
        "total_candidate_rows": int(len(panel)),
        "invalid_labels": int(invalid_labels),
        "infinite_values_converted_to_missing": int(inf_converted),
        "predictor_count": len(features),
        "predictor_order": features,
        "raw_zip_name": zip_path.name,
        "raw_zip_size_bytes": int(zip_path.stat().st_size),
        "raw_csv_member": member,
        "row_id_definition": "zero-based physical row position in the raw data CSV",
        "feature_file": str(feature_path),
        "feature_file_sha256": feature_sha,
        "feature_file_contains_label": False,
        "label_vault_file": str(label_path),
        "label_vault_sha256": label_sha,
        "label_vault_access_rule": (
            "May be used for balanced panel construction and final evaluation; "
            "must not be read by target-prediction code"
        ),
        "target_performance_calculated": False,
        "status": "PASS",
    }
    atomic_write_json(manifest_path, manifest)

    print(
        f"  {dataset}: rows={len(panel):,}, benign={ROWS_PER_CLASS:,}, "
        f"attack={ROWS_PER_CLASS:,}, inf→missing={inf_converted:,} — PASS"
    )

    del panel, feature_frame, label_frame, reservoirs
    gc.collect()
    return manifest


print("=" * 78)
print("PAPER 3 — TARGET CANDIDATE PANELS AND LABEL-VAULT SEAL")
print("=" * 78)

assert A2_PATH.exists(), f"Missing A2: {A2_PATH}"
observed_a2 = sha256_file(A2_PATH)
print("Expected A2 SHA-256:", EXPECTED_A2)
print("Observed A2 SHA-256:", observed_a2)
print("A2 verification:", "PASS" if observed_a2 == EXPECTED_A2 else "FAIL")
assert observed_a2 == EXPECTED_A2, "A2 hash mismatch"

features = get_feature_schema()
print("Feature schema verification: PASS")
print("Predictors:", len(features))
print("Candidate size per target:", f"{2 * ROWS_PER_CLASS:,}")
print("Target labels are written only to the separate label vault.")

manifests = []
for dataset, config in DATASETS.items():
    print("\n" + "-" * 78)
    print("TARGET DATASET:", dataset)
    print("-" * 78)
    manifests.append(build_candidate(dataset, config, features))

candidate_index = {
    "artifact": "paper3_target_candidate_panel_index",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "parent_amendment_A2_sha256": EXPECTED_A2,
    "datasets": {
        m["dataset"]: {
            "feature_file": m["feature_file"],
            "feature_file_sha256": m["feature_file_sha256"],
            "label_vault_file": m["label_vault_file"],
            "label_vault_sha256": m["label_vault_sha256"],
            "rows": m["total_candidate_rows"],
        }
        for m in manifests
    },
    "target_performance_calculated": False,
    "status": "PASS",
}

index_path = PANEL_ROOT / "target_candidate_panel_index.json"
if index_path.exists():
    with open(index_path, "r", encoding="utf-8") as f:
        old_index = json.load(f)
    old_cmp = dict(old_index)
    new_cmp = dict(candidate_index)
    old_cmp.pop("created_utc", None)
    new_cmp.pop("created_utc", None)
    assert old_cmp == new_cmp, "Existing candidate index differs from current artifacts"
else:
    atomic_write_json(index_path, candidate_index)

seal_payload = {
    "parent_amendment_A2_sha256": EXPECTED_A2,
    "candidate_index_sha256": sha256_file(index_path),
    "candidate_feature_hashes": {
        m["dataset"]: m["feature_file_sha256"] for m in manifests
    },
    "label_vault_hashes": {
        m["dataset"]: m["label_vault_sha256"] for m in manifests
    },
}
seal_sha = hashlib.sha256(canonical_json_bytes(seal_payload)).hexdigest()
seal_record = dict(seal_payload)
seal_record["seal_sha256"] = seal_sha
seal_record["status"] = "SEALED"

seal_path = PANEL_ROOT / "target_candidate_panel_seal.json"
if seal_path.exists():
    with open(seal_path, "r", encoding="utf-8") as f:
        assert json.load(f) == seal_record, "Existing candidate seal mismatch"
else:
    atomic_write_json(seal_path, seal_record)

print("\n" + "=" * 78)
print("TARGET CANDIDATE PANEL CONSTRUCTION: PASS")
print("Candidate panels:", len(manifests))
print("Rows per candidate panel:", f"{2 * ROWS_PER_CLASS:,}")
print("Target candidate-panel seal SHA-256:", seal_sha)
print("Prediction-visible feature panels:", CANDIDATE_ROOT)
print("Separated target label vault:", LABEL_VAULT_ROOT)
print("Target predictions calculated: NO")
print("Target performance calculated: NO")
print("=" * 78)
