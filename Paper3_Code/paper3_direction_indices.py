from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path
from datetime import datetime, timezone
import gc
import hashlib
import json
import os

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
EXPECTED_CANDIDATE_SEAL = (
    "084427a57c884ccb4ecdc4298975e5ff5"
    "9f4ca632ecdedf24c62edb82c4b2df1"
)

A2_PATH = (
    ROOT / "Paper3_Protocol" /
    "amendment_A2_target_evaluation_plan.json"
)
SANITIZED_ROOT = ROOT / "Paper3_Sanitized"
PANEL_ROOT = ROOT / "Paper3_Target_Panels"
CANDIDATE_ROOT = PANEL_ROOT / "candidates"
LABEL_VAULT_ROOT = ROOT / "Paper3_Target_Label_Vault"
DIRECTION_ROOT = PANEL_ROOT / "direction_indices"

CANDIDATE_INDEX_PATH = PANEL_ROOT / "target_candidate_panel_index.json"
CANDIDATE_SEAL_PATH = PANEL_ROOT / "target_candidate_panel_seal.json"

DATASETS = [
    "NF_BoT_IoT_v3",
    "NF_CSE_CIC_IDS2018_v3",
    "NF_ToN_IoT_v3",
    "NF_UNSW_NB15_v3",
]
SEEDS = [2027, 2028, 2029, 2030, 2031]
MODELS = ["LogisticRegression", "RandomForest", "XGBoost"]
PARTITIONS = ["train", "calibration", "source_test", "environment"]

SELECTION_SEED = 42027
CANDIDATE_ROWS_PER_CLASS = 50_000
FINAL_ROWS_PER_CLASS = 40_000


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


def atomic_to_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def splitmix64(values):
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


def feature_columns():
    reference = SANITIZED_ROOT / DATASETS[0] / "seed_2027" / "train.parquet"
    assert reference.exists(), f"Missing feature reference: {reference}"
    columns = list(pq.ParquetFile(reference).schema_arrow.names)
    features = [c for c in columns if c not in {"row_id", "Label"}]
    assert len(features) == 49, f"Expected 49 predictors, observed {len(features)}"
    return features


def canonical_row_keys(frame, features):
    """
    Collision-free byte keys for exact equality after common numeric cleaning.
    Values are float64; infinities become missing; -0.0 becomes +0.0; all NaNs
    are assigned from np.nan before the byte view is created.
    """
    matrix = frame[features].to_numpy(dtype="float64", copy=True)
    matrix[~np.isfinite(matrix)] = np.nan
    matrix[matrix == 0.0] = 0.0
    matrix[np.isnan(matrix)] = np.nan
    matrix = np.ascontiguousarray(matrix.astype("<f8", copy=False))
    width = matrix.dtype.itemsize * matrix.shape[1]
    return matrix.view(np.dtype((np.void, width))).ravel().copy()


def verify_candidate_seal():
    assert CANDIDATE_INDEX_PATH.exists(), f"Missing: {CANDIDATE_INDEX_PATH}"
    assert CANDIDATE_SEAL_PATH.exists(), f"Missing: {CANDIDATE_SEAL_PATH}"
    with open(CANDIDATE_SEAL_PATH, "r", encoding="utf-8") as f:
        record = json.load(f)

    payload = {
        "parent_amendment_A2_sha256": record["parent_amendment_A2_sha256"],
        "candidate_index_sha256": record["candidate_index_sha256"],
        "candidate_feature_hashes": record["candidate_feature_hashes"],
        "label_vault_hashes": record["label_vault_hashes"],
    }
    observed = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    assert observed == record["seal_sha256"]
    assert observed == EXPECTED_CANDIDATE_SEAL
    assert record["parent_amendment_A2_sha256"] == EXPECTED_A2
    assert sha256_file(CANDIDATE_INDEX_PATH) == record["candidate_index_sha256"]
    return record


def load_target_candidate(dataset, features, candidate_seal):
    feature_path = CANDIDATE_ROOT / dataset / "target_candidate_features.parquet"
    label_path = LABEL_VAULT_ROOT / dataset / "target_candidate_labels.parquet"
    manifest_path = CANDIDATE_ROOT / dataset / "candidate_manifest.json"

    assert feature_path.exists() and label_path.exists() and manifest_path.exists()
    assert sha256_file(feature_path) == candidate_seal["candidate_feature_hashes"][dataset]
    assert sha256_file(label_path) == candidate_seal["label_vault_hashes"][dataset]

    feature_schema = list(pq.ParquetFile(feature_path).schema_arrow.names)
    label_schema = list(pq.ParquetFile(label_path).schema_arrow.names)
    assert feature_schema == ["row_id"] + features
    assert label_schema == ["row_id", "Label"]

    feature_frame = pd.read_parquet(feature_path)
    label_frame = pd.read_parquet(label_path)
    assert len(feature_frame) == 2 * CANDIDATE_ROWS_PER_CLASS
    assert len(label_frame) == len(feature_frame)
    assert feature_frame["row_id"].equals(label_frame["row_id"])
    assert feature_frame["row_id"].is_unique

    labels = label_frame["Label"].to_numpy(dtype=np.int8)
    counts = pd.Series(labels).value_counts().to_dict()
    assert counts == {0: CANDIDATE_ROWS_PER_CLASS, 1: CANDIDATE_ROWS_PER_CLASS}

    row_ids = feature_frame["row_id"].to_numpy(dtype=np.int64)
    exact_keys = canonical_row_keys(feature_frame, features)
    rank_keys = selection_keys(dataset, row_ids)

    del feature_frame, label_frame
    gc.collect()

    return {
        "row_ids": row_ids,
        "labels": labels,
        "exact_keys": exact_keys,
        "rank_keys": rank_keys,
        "feature_path": feature_path,
        "label_path": label_path,
        "manifest_path": manifest_path,
    }


def load_source_exact_keys(dataset, seed, features):
    parts = []
    total_rows = 0
    for partition in PARTITIONS:
        path = SANITIZED_ROOT / dataset / f"seed_{seed}" / f"{partition}.parquet"
        assert path.exists(), f"Missing source partition: {path}"
        columns = list(pq.ParquetFile(path).schema_arrow.names)
        observed_features = [
            c for c in columns
            if c not in {
                "row_id", "Label", "block_id_ms",
                "pseudo_environment_id", "actual_environment_id",
            }
        ]
        assert observed_features == features, (
            f"Feature mismatch: {dataset}, seed {seed}, {partition}"
        )
        frame = pd.read_parquet(path, columns=features)
        parts.append(canonical_row_keys(frame, features))
        total_rows += len(frame)
        del frame

    keys = np.concatenate(parts)
    assert len(keys) == total_rows
    return keys, total_rows


def existing_direction_result(index_path, manifest_path):
    present = [index_path.exists(), manifest_path.exists()]
    if not any(present):
        return None
    assert all(present), (
        f"Partial direction artifacts exist at {index_path.parent}. Stop and audit."
    )
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert sha256_file(index_path) == manifest["target_index_sha256"]
    index = pd.read_parquet(index_path)
    assert list(index.columns) == ["row_id"]
    assert len(index) == 2 * FINAL_ROWS_PER_CLASS
    assert index["row_id"].is_unique
    assert manifest["target_index_contains_label"] is False
    assert manifest["final_source_target_overlap"] == 0
    assert manifest["target_performance_calculated"] is False
    return manifest


def build_direction(source, target, seed, source_keys, source_rows, target_data):
    direction_dir = DIRECTION_ROOT / f"{source}__to__{target}" / f"seed_{seed}"
    index_path = direction_dir / "target_index.parquet"
    manifest_path = direction_dir / "direction_manifest.json"

    old_manifest = existing_direction_result(index_path, manifest_path)
    if old_manifest is not None:
        print(
            f"    {source} → {target}, seed {seed}: "
            "REUSED—sealed index verified"
        )
        return old_manifest

    row_ids = target_data["row_ids"]
    labels = target_data["labels"]
    target_keys = target_data["exact_keys"]
    rank_keys = target_data["rank_keys"]

    overlap = np.isin(target_keys, source_keys, assume_unique=False)
    overlap_total = int(overlap.sum())
    overlap_by_class = {
        str(label): int((overlap & (labels == label)).sum())
        for label in (0, 1)
    }

    selected_positions = []
    available_by_class = {}
    for label in (0, 1):
        available = np.flatnonzero((labels == label) & (~overlap))
        available_by_class[str(label)] = int(len(available))
        assert len(available) >= FINAL_ROWS_PER_CLASS, (
            f"HARD STOP: {source} → {target}, seed {seed}, class {label} has "
            f"only {len(available):,} overlap-free candidates; "
            f"{FINAL_ROWS_PER_CLASS:,} are required."
        )
        order = np.lexsort((row_ids[available], rank_keys[available]))
        selected_positions.append(available[order[:FINAL_ROWS_PER_CLASS]])

    selected_positions = np.concatenate(selected_positions)
    final_overlap = int(
        np.isin(target_keys[selected_positions], source_keys, assume_unique=False).sum()
    )
    assert final_overlap == 0

    selected_labels = labels[selected_positions]
    assert pd.Series(selected_labels).value_counts().to_dict() == {
        0: FINAL_ROWS_PER_CLASS,
        1: FINAL_ROWS_PER_CLASS,
    }

    selected_ids = np.sort(row_ids[selected_positions])
    assert len(selected_ids) == 2 * FINAL_ROWS_PER_CLASS
    assert len(np.unique(selected_ids)) == len(selected_ids)

    index_frame = pd.DataFrame({"row_id": selected_ids.astype(np.int64)})
    atomic_to_parquet(index_frame, index_path)
    index_sha = sha256_file(index_path)

    source_sanitization_manifest = (
        SANITIZED_ROOT / source / f"seed_{seed}" / "sanitization_manifest.json"
    )
    assert source_sanitization_manifest.exists()

    manifest = {
        "artifact": "paper3_direction_specific_target_index",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "parent_amendment_A2_sha256": EXPECTED_A2,
        "target_candidate_panel_seal_sha256": EXPECTED_CANDIDATE_SEAL,
        "source_dataset": source,
        "target_dataset": target,
        "seed": int(seed),
        "shared_across_models": MODELS,
        "source_partition_rows": int(source_rows),
        "source_signature_definition": (
            "exact 49-feature canonical little-endian float64 byte equality; "
            "infinities→NaN, signed zero normalized, NaN representation normalized"
        ),
        "source_sanitization_manifest": str(source_sanitization_manifest),
        "source_sanitization_manifest_sha256": sha256_file(
            source_sanitization_manifest
        ),
        "target_candidate_feature_file": str(target_data["feature_path"]),
        "target_candidate_feature_sha256": sha256_file(target_data["feature_path"]),
        "target_label_vault_sha256": sha256_file(target_data["label_path"]),
        "candidate_rows": int(len(row_ids)),
        "source_target_overlap_removed": overlap_total,
        "source_target_overlap_removed_by_class": overlap_by_class,
        "overlap_free_candidates_by_class": available_by_class,
        "final_rows_per_class": FINAL_ROWS_PER_CLASS,
        "final_target_rows": int(len(index_frame)),
        "final_source_target_overlap": final_overlap,
        "target_index_file": str(index_path),
        "target_index_sha256": index_sha,
        "target_index_columns": ["row_id"],
        "target_index_contains_label": False,
        "target_index_contains_features": False,
        "target_prediction_calculated": False,
        "target_performance_calculated": False,
        "status": "PASS",
    }
    atomic_write_json(manifest_path, manifest)

    print(
        f"    {source} → {target}, seed {seed}: "
        f"overlap-removed={overlap_total:,}, final={len(index_frame):,}, "
        f"final-overlap={final_overlap} — PASS"
    )
    return manifest


print("=" * 82)
print("PAPER 3 — DIRECTION-SPECIFIC TARGET-INDEX CONSTRUCTION")
print("=" * 82)

assert A2_PATH.exists(), f"Missing A2: {A2_PATH}"
observed_a2 = sha256_file(A2_PATH)
print("A2 verification:", "PASS" if observed_a2 == EXPECTED_A2 else "FAIL")
assert observed_a2 == EXPECTED_A2, "A2 hash mismatch"

candidate_seal = verify_candidate_seal()
print("Target candidate-panel seal verification: PASS")
print("Candidate-panel seal SHA-256:", EXPECTED_CANDIDATE_SEAL)

features = feature_columns()
print("Predictors:", len(features))
print("Directed transfers:", len(DATASETS) * (len(DATASETS) - 1))
print("Direction-specific indices:", len(DATASETS) * (len(DATASETS) - 1) * len(SEEDS))
print("Rows per final index:", f"{2 * FINAL_ROWS_PER_CLASS:,}")
print("Target performance calculated before construction: NO")

print("\nLoading four candidate feature panels and the separated construction labels...")
targets = {}
for dataset in DATASETS:
    targets[dataset] = load_target_candidate(dataset, features, candidate_seal)
    print(f"  {dataset}: candidate verification and exact signatures — PASS")

all_manifests = []
for source in DATASETS:
    print("\n" + "-" * 82)
    print("SOURCE DATASET:", source)
    print("-" * 82)
    for seed in SEEDS:
        target_names = [d for d in DATASETS if d != source]
        reusable = True
        reused_manifests = []
        for target in target_names:
            direction_dir = DIRECTION_ROOT / f"{source}__to__{target}" / f"seed_{seed}"
            index_path = direction_dir / "target_index.parquet"
            manifest_path = direction_dir / "direction_manifest.json"
            old = existing_direction_result(index_path, manifest_path)
            if old is None:
                reusable = False
                break
            reused_manifests.append(old)

        if reusable:
            all_manifests.extend(reused_manifests)
            print(f"  Seed {seed}: all three direction indices REUSED—verified")
            continue

        source_keys, source_rows = load_source_exact_keys(source, seed, features)
        print(
            f"  Seed {seed}: source rows/signatures={source_rows:,}; "
            "constructing three targets"
        )
        for target in target_names:
            manifest = build_direction(
                source,
                target,
                seed,
                source_keys,
                source_rows,
                targets[target],
            )
            all_manifests.append(manifest)
        del source_keys
        gc.collect()

assert len(all_manifests) == 60, f"Expected 60 manifests, observed {len(all_manifests)}"

summary_rows = []
for m in all_manifests:
    summary_rows.append({
        "source_dataset": m["source_dataset"],
        "target_dataset": m["target_dataset"],
        "seed": m["seed"],
        "source_partition_rows": m["source_partition_rows"],
        "candidate_rows": m["candidate_rows"],
        "overlap_removed": m["source_target_overlap_removed"],
        "overlap_removed_benign": m["source_target_overlap_removed_by_class"]["0"],
        "overlap_removed_attack": m["source_target_overlap_removed_by_class"]["1"],
        "available_benign": m["overlap_free_candidates_by_class"]["0"],
        "available_attack": m["overlap_free_candidates_by_class"]["1"],
        "final_rows": m["final_target_rows"],
        "final_overlap": m["final_source_target_overlap"],
        "target_index_sha256": m["target_index_sha256"],
    })

summary = pd.DataFrame(summary_rows).sort_values(
    ["source_dataset", "target_dataset", "seed"]
).reset_index(drop=True)
assert len(summary) == 60
assert (summary["final_rows"] == 80_000).all()
assert (summary["final_overlap"] == 0).all()

summary_path = PANEL_ROOT / "direction_index_summary.csv"
if summary_path.exists():
    old_summary = pd.read_csv(summary_path)
    pd.testing.assert_frame_equal(old_summary, summary, check_dtype=False)
else:
    atomic_to_csv(summary, summary_path)

direction_records = {}
for m in all_manifests:
    key = f"{m['source_dataset']}__to__{m['target_dataset']}__seed_{m['seed']}"
    manifest_path = (
        DIRECTION_ROOT /
        f"{m['source_dataset']}__to__{m['target_dataset']}" /
        f"seed_{m['seed']}" /
        "direction_manifest.json"
    )
    direction_records[key] = {
        "manifest_sha256": sha256_file(manifest_path),
        "target_index_sha256": m["target_index_sha256"],
    }

seal_payload = {
    "parent_amendment_A2_sha256": EXPECTED_A2,
    "target_candidate_panel_seal_sha256": EXPECTED_CANDIDATE_SEAL,
    "direction_index_summary_sha256": sha256_file(summary_path),
    "direction_records": direction_records,
    "directed_transfers": 12,
    "direction_specific_indices": 60,
    "rows_per_index": 80_000,
    "final_overlap_total": int(summary["final_overlap"].sum()),
    "target_predictions_calculated": False,
    "target_performance_calculated": False,
}
seal_sha = hashlib.sha256(canonical_json_bytes(seal_payload)).hexdigest()
seal_record = dict(seal_payload)
seal_record["seal_sha256"] = seal_sha
seal_record["status"] = "SEALED"

seal_path = PANEL_ROOT / "direction_index_seal.json"
if seal_path.exists():
    with open(seal_path, "r", encoding="utf-8") as f:
        assert json.load(f) == seal_record, "Existing direction-index seal mismatch"
else:
    atomic_write_json(seal_path, seal_record)

print("\n" + "=" * 82)
print("DIRECTION-SPECIFIC TARGET-INDEX CONSTRUCTION: PASS")
print("Directed transfers: 12")
print("Direction-specific indices: 60")
print("Rows per final index: 80,000 (40,000 benign + 40,000 attack)")
print("Total overlap removed:", f"{int(summary['overlap_removed'].sum()):,}")
print("Final source-target overlap total:", int(summary["final_overlap"].sum()))
print("Direction-index seal SHA-256:", seal_sha)
print("Summary:", summary_path)
print("Target predictions calculated: NO")
print("Target performance calculated: NO")
print("=" * 82)
