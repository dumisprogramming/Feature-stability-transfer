from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
import platform
import time

import joblib
import numpy as np
import pandas as pd
import sklearn

try:
    import xgboost
    XGBOOST_VERSION = xgboost.__version__
except Exception:
    XGBOOST_VERSION = "unavailable"


ROOT = Path(
    "/content/drive/MyDrive/research/researchdata/"
    "NIDS_Research/Paper3_DataSet"
)

EXPECTED_PROTOCOL = (
    "e39a5007a7f2cbdc348d5179ce593af5"
    "766bc6688f5baa3a46a56399f683a488"
)
EXPECTED_A1 = (
    "19c30a37d62d36f5b3c9bb795f64e10"
    "599dcef78a9c51bf96ca80b58d7db4b47"
)
EXPECTED_SCHEMA = (
    "67d61eebb3377e3053f943f1e4c03123"
    "7b3ef35dd06296d8538fe552b16be0c6"
)
EXPECTED_SOURCE_MODEL_SEAL = (
    "81eebf7b013fec0ca1802ebb54894fe73"
    "d7023b2c6450da09c72af3b3f5a3182"
)
EXPECTED_A2 = (
    "2217078acfa10700f4fae29c6f32f171"
    "ce7a4e2ecb22d22ce5eabb6087e5aa5e"
)
EXPECTED_CANDIDATE_SEAL = (
    "084427a57c884ccb4ecdc4298975e5ff5"
    "9f4ca632ecdedf24c62edb82c4b2df1"
)
EXPECTED_DIRECTION_SEAL = (
    "ffd64bd8a98b1a6d8bd23eecb2f96113"
    "307f11e27837a1c467487df652cc7d7c"
)

A2_PATH = (
    ROOT / "Paper3_Protocol" /
    "amendment_A2_target_evaluation_plan.json"
)
MODEL_ROOT = ROOT / "Paper3_Source_Models"
SOURCE_MODEL_SEAL_MANIFEST = MODEL_ROOT / "source_model_seal_manifest.json"
SOURCE_MODEL_SEAL_TEXT = MODEL_ROOT / "source_model_seal_manifest.sha256"

PANEL_ROOT = ROOT / "Paper3_Target_Panels"
CANDIDATE_ROOT = PANEL_ROOT / "candidates"
CANDIDATE_SEAL_PATH = PANEL_ROOT / "target_candidate_panel_seal.json"
DIRECTION_ROOT = PANEL_ROOT / "direction_indices"
DIRECTION_SEAL_PATH = PANEL_ROOT / "direction_index_seal.json"

PREDICTION_ROOT = ROOT / "Paper3_Target_Predictions"

DATASETS = [
    "NF_BoT_IoT_v3",
    "NF_CSE_CIC_IDS2018_v3",
    "NF_ToN_IoT_v3",
    "NF_UNSW_NB15_v3",
]
SEEDS = [2027, 2028, 2029, 2030, 2031]
MODELS = ["LogisticRegression", "RandomForest", "XGBoost"]
EXPECTED_ROWS_PER_JOB = 80_000


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


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def verify_payload_seal(path, expected_sha, payload_keys):
    assert path.exists(), f"Missing seal: {path}"
    record = load_json(path)
    payload = {key: record[key] for key in payload_keys}
    observed = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    assert observed == record["seal_sha256"], f"Internal seal mismatch: {path}"
    assert observed == expected_sha, f"Unexpected seal at {path}: {observed}"
    return record


def find_artifact_record(node, filename):
    if isinstance(node, dict):
        path_value = node.get("path")
        sha_value = node.get("sha256")
        if (
            isinstance(path_value, str)
            and Path(path_value).name == filename
            and isinstance(sha_value, str)
        ):
            return {"path": path_value, "sha256": sha_value}
        for value in node.values():
            found = find_artifact_record(value, filename)
            if found is not None:
                return found
    elif isinstance(node, list):
        for value in node:
            found = find_artifact_record(value, filename)
            if found is not None:
                return found
    return None


def find_artifact_record_by_path(node, target_path):
    target_path = Path(target_path).resolve()
    if isinstance(node, dict):
        path_value = node.get("path")
        sha_value = node.get("sha256")
        if isinstance(path_value, str) and isinstance(sha_value, str):
            if Path(path_value).resolve() == target_path:
                return {"path": path_value, "sha256": sha_value}
        for value in node.values():
            found = find_artifact_record_by_path(value, target_path)
            if found is not None:
                return found
    elif isinstance(node, list):
        for value in node:
            found = find_artifact_record_by_path(value, target_path)
            if found is not None:
                return found
    return None


def unpack_source_model(
    model_bundle,
    model_name,
    source,
    seed,
    features,
):
    """Validate a sealed source-model bundle and return its estimator."""
    assert isinstance(model_bundle, dict), (
        f"Unexpected {model_name} artifact type: {type(model_bundle)}"
    )
    required_keys = {
        "dataset",
        "seed",
        "model_name",
        "model",
        "uses_standard_scaler",
        "threshold",
        "features",
        "protocol_sha256",
        "amendment_A1_sha256",
        "feature_schema_sha256",
    }
    missing_keys = required_keys.difference(model_bundle)
    assert not missing_keys, (
        f"Missing keys in {model_name} artifact: {sorted(missing_keys)}"
    )
    assert model_bundle["dataset"] == source
    assert int(model_bundle["seed"]) == seed
    assert model_bundle["model_name"] == model_name
    assert list(model_bundle["features"]) == features
    assert model_bundle["protocol_sha256"] == EXPECTED_PROTOCOL
    assert model_bundle["amendment_A1_sha256"] == EXPECTED_A1
    assert model_bundle["feature_schema_sha256"] == EXPECTED_SCHEMA

    expected_scaler = model_name == "LogisticRegression"
    assert bool(model_bundle["uses_standard_scaler"]) == expected_scaler

    estimator = model_bundle["model"]
    assert hasattr(estimator, "predict_proba"), (
        f"Embedded {model_name} estimator has no predict_proba method: "
        f"{type(estimator)}"
    )
    assert hasattr(estimator, "classes_"), (
        f"Embedded {model_name} estimator has no fitted classes_ attribute"
    )
    classes = np.asarray(estimator.classes_)
    assert classes.ndim == 1 and len(classes) == 2
    assert np.array_equal(np.sort(classes.astype(np.int64)), np.array([0, 1]))

    return estimator, float(model_bundle["threshold"])


def predict_attack_probability(estimator, input_matrix):
    """Return the probability column corresponding to attack label 1."""
    probability_matrix = np.asarray(
        estimator.predict_proba(input_matrix),
        dtype=np.float64,
    )
    assert probability_matrix.shape == (EXPECTED_ROWS_PER_JOB, 2)
    classes = np.asarray(estimator.classes_)
    attack_columns = np.flatnonzero(classes == 1)
    assert len(attack_columns) == 1
    return probability_matrix[:, int(attack_columns[0])]


def verify_source_model_chain():
    assert SOURCE_MODEL_SEAL_MANIFEST.exists()
    assert SOURCE_MODEL_SEAL_TEXT.exists()
    recorded = SOURCE_MODEL_SEAL_TEXT.read_text(encoding="utf-8").strip().split()[0]
    observed = sha256_file(SOURCE_MODEL_SEAL_MANIFEST)
    assert recorded == observed == EXPECTED_SOURCE_MODEL_SEAL
    seal = load_json(SOURCE_MODEL_SEAL_MANIFEST)
    assert seal["protocol_sha256"] == EXPECTED_PROTOCOL
    assert seal["amendment_A1_sha256"] == EXPECTED_A1
    assert seal["feature_schema_sha256"] == EXPECTED_SCHEMA
    assert seal["source_models"] == 60
    assert seal["target_datasets_evaluated"] in (0, [], None)
    assert seal["target_results_seen"] is False
    assert any(word in str(seal["status"]).upper() for word in ("PASS", "SEAL"))
    return seal


def verify_direction_record(direction_seal, source, target, seed):
    key = f"{source}__to__{target}__seed_{seed}"
    assert key in direction_seal["direction_records"], f"Missing direction: {key}"
    record = direction_seal["direction_records"][key]
    direction_dir = DIRECTION_ROOT / f"{source}__to__{target}" / f"seed_{seed}"
    index_path = direction_dir / "target_index.parquet"
    manifest_path = direction_dir / "direction_manifest.json"
    assert index_path.exists() and manifest_path.exists()
    assert sha256_file(index_path) == record["target_index_sha256"]
    assert sha256_file(manifest_path) == record["manifest_sha256"]
    manifest = load_json(manifest_path)
    assert manifest["target_index_sha256"] == record["target_index_sha256"]
    assert manifest["source_dataset"] == source
    assert manifest["target_dataset"] == target
    assert manifest["seed"] == seed
    assert manifest["final_target_rows"] == EXPECTED_ROWS_PER_JOB
    assert manifest["final_source_target_overlap"] == 0
    assert manifest["target_index_contains_label"] is False
    assert manifest["target_performance_calculated"] is False
    return index_path, manifest_path, manifest


def verify_existing_job(prediction_path, manifest_path, expected):
    present = [prediction_path.exists(), manifest_path.exists()]
    if not any(present):
        return None
    assert all(present), (
        f"Partial prediction artifacts exist in {prediction_path.parent}. "
        "Stop and audit before rerunning."
    )
    manifest = load_json(manifest_path)
    assert sha256_file(prediction_path) == manifest["prediction_file_sha256"]
    for key, value in expected.items():
        assert manifest[key] == value, (
            f"Existing prediction manifest mismatch for {key}: "
            f"{manifest[key]} != {value}"
        )
    frame = pd.read_parquet(prediction_path)
    assert list(frame.columns) == [
        "row_id", "attack_probability", "predicted_label"
    ]
    assert len(frame) == EXPECTED_ROWS_PER_JOB
    assert frame["row_id"].is_unique
    probability = frame["attack_probability"].to_numpy(dtype=np.float64)
    prediction = frame["predicted_label"].to_numpy(dtype=np.int8)
    assert np.isfinite(probability).all()
    assert ((probability >= 0.0) & (probability <= 1.0)).all()
    assert np.isin(prediction, [0, 1]).all()
    assert "target_label" not in manifest
    assert manifest["target_label_rows_read"] == 0
    assert manifest["target_performance_calculated"] is False
    del frame
    gc.collect()
    return manifest


print("=" * 86)
print("PAPER 3 — LABEL-BLIND CROSS-DATASET TARGET PREDICTIONS")
print("=" * 86)

assert A2_PATH.exists()
assert sha256_file(A2_PATH) == EXPECTED_A2
print("Amendment A2 verification: PASS")

source_model_seal = verify_source_model_chain()
print("Source-model seal verification: PASS")

candidate_seal = verify_payload_seal(
    CANDIDATE_SEAL_PATH,
    EXPECTED_CANDIDATE_SEAL,
    [
        "parent_amendment_A2_sha256",
        "candidate_index_sha256",
        "candidate_feature_hashes",
        "label_vault_hashes",
    ],
)
assert candidate_seal["parent_amendment_A2_sha256"] == EXPECTED_A2
print("Target candidate-panel seal verification: PASS")

direction_seal = verify_payload_seal(
    DIRECTION_SEAL_PATH,
    EXPECTED_DIRECTION_SEAL,
    [
        "parent_amendment_A2_sha256",
        "target_candidate_panel_seal_sha256",
        "direction_index_summary_sha256",
        "direction_records",
        "directed_transfers",
        "direction_specific_indices",
        "rows_per_index",
        "final_overlap_total",
        "target_predictions_calculated",
        "target_performance_calculated",
    ],
)
assert direction_seal["parent_amendment_A2_sha256"] == EXPECTED_A2
assert direction_seal["target_candidate_panel_seal_sha256"] == EXPECTED_CANDIDATE_SEAL
assert direction_seal["direction_specific_indices"] == 60
assert direction_seal["final_overlap_total"] == 0
assert direction_seal["target_predictions_calculated"] is False
assert direction_seal["target_performance_calculated"] is False
print("Direction-index seal verification: PASS")

script_path = Path(__file__).resolve()
script_sha = sha256_file(script_path) if script_path.exists() else "unavailable"

software_versions = {
    "python": platform.python_version(),
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "scikit_learn": sklearn.__version__,
    "xgboost": XGBOOST_VERSION,
    "joblib": joblib.__version__,
}

print("Prediction jobs:", 180)
print("Rows per job:", f"{EXPECTED_ROWS_PER_JOB:,}")
print("Target label-vault path present in this program: NO")
print("Target label rows read before prediction: 0")
print("Software versions:", software_versions)

# Load only the four prediction-visible feature panels. No label-vault path is
# defined or accessed anywhere in this script.
print("\nLoading prediction-visible candidate feature panels...")
candidate_frames = {}
candidate_feature_hashes = {}
for dataset in DATASETS:
    feature_path = CANDIDATE_ROOT / dataset / "target_candidate_features.parquet"
    assert feature_path.exists(), f"Missing: {feature_path}"
    observed_hash = sha256_file(feature_path)
    assert observed_hash == candidate_seal["candidate_feature_hashes"][dataset]
    frame = pd.read_parquet(feature_path)
    assert "Label" not in frame.columns
    assert "Attack" not in frame.columns
    assert frame["row_id"].is_unique
    assert len(frame) == 100_000
    frame = frame.set_index("row_id", verify_integrity=True)
    candidate_frames[dataset] = frame
    candidate_feature_hashes[dataset] = observed_hash
    print(f"  {dataset}: rows={len(frame):,}, target-label-columns=0 — PASS")

all_job_manifests = []
jobs_completed_now = 0
jobs_reused = 0
run_started = time.time()

for source in DATASETS:
    print("\n" + "-" * 86)
    print("SOURCE DATASET:", source)
    print("-" * 86)

    for seed in SEEDS:
        seed_dir = MODEL_ROOT / source / f"seed_{seed}"
        seed_manifest_path = seed_dir / "source_seed_complete.json"
        assert seed_manifest_path.exists()
        seed_manifest = load_json(seed_manifest_path)
        assert seed_manifest["dataset"] == source
        assert seed_manifest["seed"] == seed
        assert seed_manifest["protocol_sha256"] == EXPECTED_PROTOCOL
        assert seed_manifest["amendment_A1_sha256"] == EXPECTED_A1
        assert seed_manifest["feature_schema_sha256"] == EXPECTED_SCHEMA
        assert seed_manifest["status"] == "PASS"

        preprocessor_filename = "source_preprocessor.joblib"
        preprocessor_record = find_artifact_record(
            seed_manifest, preprocessor_filename
        )
        preprocessor_path = seed_dir / preprocessor_filename
        assert preprocessor_path.exists()
        if preprocessor_record is None:
            preprocessor_record = find_artifact_record_by_path(
                source_model_seal, preprocessor_path
            )
        if preprocessor_record is not None:
            assert Path(preprocessor_record["path"]).resolve() == preprocessor_path.resolve()
            assert sha256_file(preprocessor_path) == preprocessor_record["sha256"]
            preprocessor_sha = preprocessor_record["sha256"]
        else:
            # The source seal still fixes this file even when an older per-seed
            # manifest did not expose its path as a nested file record.
            preprocessor_sha = sha256_file(preprocessor_path)

        preprocessor = joblib.load(preprocessor_path)
        assert isinstance(preprocessor, dict)
        assert preprocessor["dataset"] == source
        assert preprocessor["seed"] == seed
        assert preprocessor["protocol_sha256"] == EXPECTED_PROTOCOL
        assert preprocessor["amendment_A1_sha256"] == EXPECTED_A1
        assert preprocessor["feature_schema_sha256"] == EXPECTED_SCHEMA
        features = list(preprocessor["features"])
        assert len(features) == 49
        assert hasattr(preprocessor["imputer"], "transform")
        assert hasattr(preprocessor["logistic_regression_scaler"], "transform")

        models_loaded = {}
        model_hashes = {}
        thresholds = {}
        for model_name in MODELS:
            model_filename = f"{model_name}_model.joblib"
            model_record = seed_manifest["models"][model_name]["files"]["model"]
            model_path = Path(model_record["path"])
            assert model_path.resolve() == (seed_dir / model_filename).resolve()
            assert sha256_file(model_path) == model_record["sha256"]
            model_bundle = joblib.load(model_path)
            estimator, bundled_threshold = unpack_source_model(
                model_bundle=model_bundle,
                model_name=model_name,
                source=source,
                seed=seed,
                features=features,
            )
            models_loaded[model_name] = estimator
            model_hashes[model_name] = model_record["sha256"]

            model_info = seed_manifest["models"][model_name]
            threshold = float(model_info["threshold_selection"]["threshold"])
            source_threshold = float(model_info["source_test_metrics"]["threshold"])
            calibration_threshold = float(model_info["calibration_metrics"]["threshold"])
            assert np.isclose(threshold, bundled_threshold, rtol=0.0, atol=1e-15)
            assert np.isclose(threshold, source_threshold, rtol=0.0, atol=1e-15)
            assert np.isclose(threshold, calibration_threshold, rtol=0.0, atol=1e-15)
            thresholds[model_name] = threshold

            expected_scaler = model_name == "LogisticRegression"
            assert bool(model_info["uses_standard_scaler"]) == expected_scaler
            assert bool(model_bundle["uses_standard_scaler"]) == expected_scaler

        for target in DATASETS:
            if target == source:
                continue

            index_path, direction_manifest_path, direction_manifest = (
                verify_direction_record(direction_seal, source, target, seed)
            )
            index_frame = pd.read_parquet(index_path)
            assert list(index_frame.columns) == ["row_id"]
            assert len(index_frame) == EXPECTED_ROWS_PER_JOB
            assert index_frame["row_id"].is_unique
            row_ids = index_frame["row_id"].to_numpy(dtype=np.int64)

            candidate = candidate_frames[target]
            missing_ids = np.setdiff1d(row_ids, candidate.index.to_numpy(), assume_unique=False)
            assert len(missing_ids) == 0
            selected = candidate.loc[row_ids]
            assert list(selected.columns) == features
            assert len(selected) == EXPECTED_ROWS_PER_JOB

            raw_matrix = selected.to_numpy(dtype=np.float64, copy=True)
            raw_matrix[~np.isfinite(raw_matrix)] = np.nan
            imputed = preprocessor["imputer"].transform(raw_matrix)
            assert imputed.shape == (EXPECTED_ROWS_PER_JOB, len(features))
            assert np.isfinite(imputed).all()
            scaled = preprocessor["logistic_regression_scaler"].transform(imputed)
            assert scaled.shape == imputed.shape
            assert np.isfinite(scaled).all()

            for model_name in MODELS:
                output_dir = (
                    PREDICTION_ROOT /
                    f"{source}__to__{target}" /
                    f"seed_{seed}"
                )
                prediction_path = output_dir / f"{model_name}_target_predictions.parquet"
                manifest_path = output_dir / f"{model_name}_prediction_manifest.json"

                expected_manifest_fields = {
                    "source_dataset": source,
                    "target_dataset": target,
                    "seed": seed,
                    "model": model_name,
                    "parent_amendment_A2_sha256": EXPECTED_A2,
                    "source_model_seal_sha256": EXPECTED_SOURCE_MODEL_SEAL,
                    "target_candidate_panel_seal_sha256": EXPECTED_CANDIDATE_SEAL,
                    "direction_index_seal_sha256": EXPECTED_DIRECTION_SEAL,
                    "target_index_sha256": direction_manifest["target_index_sha256"],
                    "target_candidate_feature_sha256": candidate_feature_hashes[target],
                    "source_preprocessor_sha256": preprocessor_sha,
                    "source_model_sha256": model_hashes[model_name],
                    "prediction_script_sha256": script_sha,
                    "target_rows": EXPECTED_ROWS_PER_JOB,
                    "target_label_rows_read": 0,
                    "target_performance_calculated": False,
                }

                old_manifest = verify_existing_job(
                    prediction_path, manifest_path, expected_manifest_fields
                )
                if old_manifest is not None:
                    all_job_manifests.append(old_manifest)
                    jobs_reused += 1
                    print(
                        f"  {source} → {target} | seed {seed} | {model_name}: "
                        "REUSED—prediction hash verified"
                    )
                    continue

                input_matrix = scaled if model_name == "LogisticRegression" else imputed
                probabilities = predict_attack_probability(
                    models_loaded[model_name],
                    input_matrix,
                )
                assert probabilities.shape == (EXPECTED_ROWS_PER_JOB,)
                assert np.isfinite(probabilities).all()
                assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()

                threshold = thresholds[model_name]
                predicted = (probabilities >= threshold).astype(np.int8)

                prediction_frame = pd.DataFrame({
                    "row_id": row_ids,
                    "attack_probability": probabilities,
                    "predicted_label": predicted,
                })
                assert prediction_frame["row_id"].is_unique
                atomic_to_parquet(prediction_frame, prediction_path)
                prediction_sha = sha256_file(prediction_path)

                manifest = {
                    "artifact": "paper3_label_blind_target_prediction",
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    **expected_manifest_fields,
                    "protocol_sha256": EXPECTED_PROTOCOL,
                    "amendment_A1_sha256": EXPECTED_A1,
                    "feature_schema_sha256": EXPECTED_SCHEMA,
                    "source_seed_manifest_sha256": sha256_file(seed_manifest_path),
                    "direction_manifest_sha256": sha256_file(direction_manifest_path),
                    "model_threshold": threshold,
                    "threshold_source": "source-only calibration set",
                    "preprocessing": (
                        "frozen source median imputer; frozen source StandardScaler "
                        "for LogisticRegression only"
                    ),
                    "target_adaptation": False,
                    "prediction_columns": [
                        "row_id", "attack_probability", "predicted_label"
                    ],
                    "prediction_contains_target_label": False,
                    "probability_gate_pass": True,
                    "prediction_file": str(prediction_path),
                    "prediction_file_sha256": prediction_sha,
                    "software_versions": software_versions,
                    "status": "PASS",
                }
                atomic_write_json(manifest_path, manifest)

                all_job_manifests.append(manifest)
                jobs_completed_now += 1
                print(
                    f"  {source} → {target} | seed {seed} | {model_name}: "
                    f"rows={EXPECTED_ROWS_PER_JOB:,}, labels-read=0 — PASS"
                )

                del probabilities, predicted, prediction_frame
                gc.collect()

            del selected, raw_matrix, imputed, scaled, index_frame, row_ids
            gc.collect()

        del preprocessor, models_loaded
        gc.collect()

assert len(all_job_manifests) == 180, (
    f"Expected 180 prediction manifests, observed {len(all_job_manifests)}"
)

inventory_rows = []
job_records = {}
for manifest in all_job_manifests:
    prediction_path = Path(manifest["prediction_file"])
    manifest_path = (
        prediction_path.parent /
        f"{manifest['model']}_prediction_manifest.json"
    )
    assert prediction_path.exists() and manifest_path.exists()
    assert sha256_file(prediction_path) == manifest["prediction_file_sha256"]
    manifest_sha = sha256_file(manifest_path)

    key = (
        f"{manifest['source_dataset']}__to__{manifest['target_dataset']}__"
        f"seed_{manifest['seed']}__{manifest['model']}"
    )
    assert key not in job_records
    job_records[key] = {
        "prediction_file_sha256": manifest["prediction_file_sha256"],
        "prediction_manifest_sha256": manifest_sha,
    }
    inventory_rows.append({
        "source_dataset": manifest["source_dataset"],
        "target_dataset": manifest["target_dataset"],
        "seed": manifest["seed"],
        "model": manifest["model"],
        "target_rows": manifest["target_rows"],
        "model_threshold": manifest["model_threshold"],
        "prediction_file_sha256": manifest["prediction_file_sha256"],
        "prediction_manifest_sha256": manifest_sha,
        "target_label_rows_read": manifest["target_label_rows_read"],
        "target_performance_calculated": manifest["target_performance_calculated"],
    })

inventory = pd.DataFrame(inventory_rows).sort_values(
    ["source_dataset", "target_dataset", "seed", "model"]
).reset_index(drop=True)
assert len(inventory) == 180
assert (inventory["target_rows"] == EXPECTED_ROWS_PER_JOB).all()
assert (inventory["target_label_rows_read"] == 0).all()
assert (~inventory["target_performance_calculated"]).all()

inventory_path = PREDICTION_ROOT / "target_prediction_inventory.csv"
if inventory_path.exists():
    old_inventory = pd.read_csv(inventory_path)
    pd.testing.assert_frame_equal(old_inventory, inventory, check_dtype=False)
else:
    atomic_to_csv(inventory, inventory_path)

seal_payload = {
    "artifact": "paper3_pre_label_target_prediction_seal",
    "parent_amendment_A2_sha256": EXPECTED_A2,
    "source_model_seal_sha256": EXPECTED_SOURCE_MODEL_SEAL,
    "target_candidate_panel_seal_sha256": EXPECTED_CANDIDATE_SEAL,
    "direction_index_seal_sha256": EXPECTED_DIRECTION_SEAL,
    "prediction_script_sha256": script_sha,
    "prediction_inventory_sha256": sha256_file(inventory_path),
    "prediction_jobs": 180,
    "rows_per_job": EXPECTED_ROWS_PER_JOB,
    "job_records": job_records,
    "target_label_rows_read": 0,
    "target_adaptation_performed": False,
    "target_performance_calculated": False,
}
prediction_seal_sha = hashlib.sha256(
    canonical_json_bytes(seal_payload)
).hexdigest()
seal_record = dict(seal_payload)
seal_record["seal_sha256"] = prediction_seal_sha
seal_record["status"] = "SEALED_BEFORE_TARGET_LABEL_JOIN"

seal_path = PREDICTION_ROOT / "pre_label_target_prediction_seal.json"
if seal_path.exists():
    old_seal = load_json(seal_path)
    assert old_seal == seal_record, "Existing prediction seal mismatch"
else:
    atomic_write_json(seal_path, seal_record)

elapsed_minutes = (time.time() - run_started) / 60.0

print("\n" + "=" * 86)
print("LABEL-BLIND TARGET PREDICTION AND PRE-LABEL SEAL: PASS")
print("Prediction jobs sealed: 180")
print("Completed during this run:", jobs_completed_now)
print("Reused and verified:", jobs_reused)
print("Rows per prediction job:", f"{EXPECTED_ROWS_PER_JOB:,}")
print("Target label rows read: 0")
print("Target adaptation performed: NO")
print("Target performance calculated: NO")
print("Pre-label target-prediction seal SHA-256:", prediction_seal_sha)
print("Prediction inventory:", inventory_path)
print("Elapsed minutes:", round(elapsed_minutes, 2))
print("=" * 86)
