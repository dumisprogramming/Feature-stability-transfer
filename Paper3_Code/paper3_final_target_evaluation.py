from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path
from datetime import datetime, timezone
from itertools import permutations, product
import hashlib
import json
import os
import platform
import time

import numpy as np
import pandas as pd
import scipy
from scipy.stats import spearmanr
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    recall_score,
)


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
EXPECTED_SOURCE_STABILITY_SEAL = (
    "3ec5641ed96a462ce2c76765b6b4483a"
    "1e48cad11d79981feb7afd1c43449340"
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
EXPECTED_PRE_LABEL_SEAL = (
    "7e9898212fda48e756b061529f46e9ca"
    "0d5c80512b2dc43c7da87f698ce7c618"
)
EXPECTED_PREDICTION_SCRIPT = (
    "bdf16dab915ea5994d9917b89f6902fc"
    "f20905ee8a07e5991eeda45a300265cb"
)

DATASETS = [
    "NF_BoT_IoT_v3",
    "NF_CSE_CIC_IDS2018_v3",
    "NF_ToN_IoT_v3",
    "NF_UNSW_NB15_v3",
]
MODELS = ["LogisticRegression", "RandomForest", "XGBoost"]
SEEDS = [2027, 2028, 2029, 2030, 2031]
EXPECTED_ROWS_PER_JOB = 80_000
EXPECTED_ROWS_PER_CLASS = 40_000

A2_PATH = (
    ROOT / "Paper3_Protocol" /
    "amendment_A2_target_evaluation_plan.json"
)
A3_PATH = (
    ROOT / "Paper3_Protocol" /
    "amendment_A3_confirmatory_analysis_plan.json"
)

MODEL_ROOT = ROOT / "Paper3_Source_Models"
SOURCE_MODEL_SEAL_PATH = MODEL_ROOT / "source_model_seal_manifest.json"
SOURCE_MODEL_SEAL_TEXT = MODEL_ROOT / "source_model_seal_manifest.sha256"
SOURCE_METRICS_PATH = MODEL_ROOT / "source_test_metrics_all_models.csv"

STABILITY_ROOT = ROOT / "Paper3_Source_Stability"
WARNING_SCORES_PATH = STABILITY_ROOT / "source_only_warning_scores.csv"

PANEL_ROOT = ROOT / "Paper3_Target_Panels"
CANDIDATE_SEAL_PATH = PANEL_ROOT / "target_candidate_panel_seal.json"
DIRECTION_SEAL_PATH = PANEL_ROOT / "direction_index_seal.json"
DIRECTION_ROOT = PANEL_ROOT / "direction_indices"
LABEL_VAULT_ROOT = ROOT / "Paper3_Target_Label_Vault"

PREDICTION_ROOT = ROOT / "Paper3_Target_Predictions"
PRE_LABEL_SEAL_PATH = PREDICTION_ROOT / "pre_label_target_prediction_seal.json"
PREDICTION_INVENTORY_PATH = PREDICTION_ROOT / "target_prediction_inventory.csv"

FINAL_ROOT = ROOT / "Paper3_Final_Evaluation"
FINAL_SEAL_PATH = FINAL_ROOT / "final_results_seal.json"

BOOTSTRAP_SEED = 53027
BOOTSTRAP_REPLICATES = 10_000
ALPHA = 0.05


A3_PAYLOAD = {
    "amendment_id": "P3_A3_CONFIRMATORY_ANALYSIS_PLAN",
    "parent_amendment_A2_sha256": EXPECTED_A2,
    "pre_label_target_prediction_seal_sha256": EXPECTED_PRE_LABEL_SEAL,
    "timing": (
        "Frozen after all 180 label-blind target predictions were sealed and "
        "before any target label was joined to a prediction."
    ),
    "outcome": {
        "primary": "balanced_accuracy_loss",
        "formula": (
            "source_test_balanced_accuracy - target_balanced_accuracy"
        ),
        "negative_values": "Retained without clipping.",
        "secondary": [
            "target_balanced_accuracy",
            "target_macro_f1",
            "target_mcc",
            "target_pr_auc_attack",
            "target_benign_recall",
            "target_attack_recall",
        ],
    },
    "warning_scores": {
        "primary": [
            "environmental_instability",
            "sampling_environment_gap",
        ],
        "baseline": "sampling_instability",
        "baseline_formula": "1 - sampling_stability",
    },
    "reporting_units": {
        "job_level": (
            "All 180 source-target-seed-model results are reported."
        ),
        "confirmatory_association": (
            "Twelve source-dataset-by-model units; outcome and warning scores "
            "are arithmetic means across five seeds, and outcomes are also "
            "averaged across the three eligible target datasets."
        ),
        "predictive_validation": (
            "Sixty source-dataset-by-seed-by-model units; each outcome is the "
            "arithmetic mean across the three eligible target datasets."
        ),
    },
    "confirmatory_association": {
        "effect": "Spearman rank correlation",
        "alternative": "positive one-sided association",
        "permutation": (
            "Exact permutation of source-dataset assignments within each model; "
            "4!^3 = 13,824 blocked permutations."
        ),
        "uncertainty": (
            "Model-stratified source-dataset bootstrap percentile 95% interval."
        ),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "multiplicity": (
            "Holm adjustment across the two primary warning predictors."
        ),
        "alpha": ALPHA,
        "support_rule": (
            "A primary association is supported only when rho > 0 and its "
            "Holm-adjusted one-sided p-value is below 0.05."
        ),
        "baseline_inference": (
            "Sampling-instability association is a prespecified comparator and "
            "is reported without inclusion in the primary Holm family."
        ),
    },
    "predictive_validation": {
        "cross_validation": (
            "Four-fold leave-one-source-dataset-out; all seeds and models from "
            "the held-out source dataset remain exclusively in the test fold."
        ),
        "estimator": "Ordinary least squares with no tuning and no clipping.",
        "continuous_preprocessing": (
            "Training-fold mean and population-standard-deviation scaling only."
        ),
        "known_covariate": (
            "Fixed model-identity indicators; source-dataset identity excluded."
        ),
        "specifications": [
            "mean_only",
            "model_only",
            "model_plus_sampling_instability",
            "model_plus_environmental_instability",
            "model_plus_sampling_environment_gap",
            "model_plus_primary_combined",
        ],
        "primary_score": "out-of-fold mean absolute error",
        "secondary_scores": [
            "out-of-fold root mean squared error",
            "out-of-fold Spearman correlation",
        ],
        "comparison_rule": (
            "Positive MAE improvement means lower MAE than the prespecified "
            "model-plus-sampling-instability baseline."
        ),
    },
    "secondary_analysis": (
        "Secondary outcomes and direction/model summaries are descriptive; no "
        "additional confirmatory significance claims are made."
    ),
    "confirmatory_rules": {
        "all_180_prediction_results_reported": True,
        "negative_results_reported": True,
        "no_model_selection_using_target_results": True,
        "no_threshold_change_using_target_results": True,
        "no_target_adaptation": True,
    },
    "scope_caution": (
        "Inference is bounded to the four evaluated NetFlow-v3 datasets and "
        "three fixed model families; it is not universal deployment validation."
    ),
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


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def atomic_to_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def record_contains_string(node, target):
    if isinstance(node, dict):
        return any(record_contains_string(v, target) for v in node.values())
    if isinstance(node, list):
        return any(record_contains_string(v, target) for v in node)
    return isinstance(node, str) and node == target


def verify_known_self_seal(path, expected_sha, payload_keys):
    assert path.exists(), f"Missing seal: {path}"
    record = load_json(path)
    payload = {key: record[key] for key in payload_keys}
    observed = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    assert observed == record["seal_sha256"] == expected_sha
    return record


def locate_source_stability_seal():
    assert STABILITY_ROOT.exists()
    matches = []
    for path in sorted(STABILITY_ROOT.rglob("*.json")):
        try:
            record = load_json(path)
        except Exception:
            continue
        if record.get("seal_sha256") == EXPECTED_SOURCE_STABILITY_SEAL:
            matches.append((path, record))
        elif sha256_file(path) == EXPECTED_SOURCE_STABILITY_SEAL:
            matches.append((path, record))
    assert len(matches) == 1, (
        "Expected exactly one source-stability seal matching the frozen hash; "
        f"observed {[str(p) for p, _ in matches]}"
    )
    path, record = matches[0]

    # Source-stability implementations may place created_utc outside the seal
    # payload. Accept only the two known metadata layouts and verify one exactly.
    internal_verified = False
    if record.get("seal_sha256") == EXPECTED_SOURCE_STABILITY_SEAL:
        for excluded in (
            {"seal_sha256", "status"},
            {"seal_sha256", "status", "created_utc"},
        ):
            payload = {k: v for k, v in record.items() if k not in excluded}
            observed = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
            if observed == EXPECTED_SOURCE_STABILITY_SEAL:
                internal_verified = True
                break
        assert internal_verified, "Source-stability internal seal mismatch"
    else:
        internal_verified = True

    return path, record


def verify_a2():
    assert A2_PATH.exists(), f"Missing A2: {A2_PATH}"
    assert sha256_file(A2_PATH) == EXPECTED_A2
    a2 = load_json(A2_PATH)
    assert a2["parent_protocol_sha256"] == EXPECTED_PROTOCOL
    assert a2["amendment_A1_sha256"] == EXPECTED_A1
    assert a2["feature_schema_sha256"] == EXPECTED_SCHEMA
    assert a2["source_model_seal_sha256"] == EXPECTED_SOURCE_MODEL_SEAL
    assert a2["source_stability_seal_sha256"] == EXPECTED_SOURCE_STABILITY_SEAL
    assert a2["paper3_target_results_seen_before_freeze"] is False
    assert a2["primary_outcome"] == {
        "formula": "source_test_balanced_accuracy - target_balanced_accuracy",
        "metric": "balanced_accuracy_loss",
    }
    assert a2["primary_warning_predictors"] == [
        "environmental_instability",
        "sampling_environment_gap",
    ]
    assert a2["baseline_warning_predictor"] == "sampling instability"
    design = a2["evaluation_design"]
    assert design["datasets"] == DATASETS
    assert design["models"] == MODELS
    assert design["source_seeds"] == SEEDS
    assert design["number_of_directions"] == 12
    assert design["prediction_jobs"] == 180
    assert a2["direction_specific_panel"]["final_rows_per_class"] == 40_000
    assert a2["direction_specific_panel"]["final_rows_per_direction_seed"] == 80_000
    assert a2["target_processing"]["target_adaptation"] is False
    assert a2["target_processing"]["target_threshold_calibration"] is False
    return a2


def verify_source_only_inputs():
    assert SOURCE_MODEL_SEAL_PATH.exists()
    assert SOURCE_MODEL_SEAL_TEXT.exists()
    recorded = SOURCE_MODEL_SEAL_TEXT.read_text(encoding="utf-8").strip().split()[0]
    assert recorded == sha256_file(SOURCE_MODEL_SEAL_PATH)
    assert recorded == EXPECTED_SOURCE_MODEL_SEAL
    source_model_seal = load_json(SOURCE_MODEL_SEAL_PATH)
    assert source_model_seal["protocol_sha256"] == EXPECTED_PROTOCOL
    assert source_model_seal["amendment_A1_sha256"] == EXPECTED_A1
    assert source_model_seal["feature_schema_sha256"] == EXPECTED_SCHEMA
    assert source_model_seal["source_models"] == 60
    assert source_model_seal["target_results_seen"] is False

    assert SOURCE_METRICS_PATH.exists()
    source_metrics_sha = sha256_file(SOURCE_METRICS_PATH)
    assert record_contains_string(source_model_seal, source_metrics_sha), (
        "Source metrics hash is not present in the source-model seal"
    )
    source_metrics = pd.read_csv(SOURCE_METRICS_PATH)
    required_source = {
        "dataset", "seed", "model", "balanced_accuracy",
    }
    assert required_source.issubset(source_metrics.columns)
    assert len(source_metrics) == 60
    assert not source_metrics.duplicated(["dataset", "seed", "model"]).any()
    assert set(source_metrics["dataset"]) == set(DATASETS)
    assert set(source_metrics["seed"]) == set(SEEDS)
    assert set(source_metrics["model"]) == set(MODELS)
    assert source_metrics["balanced_accuracy"].between(0.0, 1.0).all()

    stability_seal_path, stability_seal = locate_source_stability_seal()
    assert WARNING_SCORES_PATH.exists()
    warning_scores_sha = sha256_file(WARNING_SCORES_PATH)
    assert record_contains_string(stability_seal, warning_scores_sha), (
        "Warning-score hash is not present in the source-stability seal"
    )
    warning = pd.read_csv(WARNING_SCORES_PATH)
    required_warning = {
        "dataset", "seed", "model", "environmental_stability",
        "sampling_stability", "environmental_instability",
        "sampling_environment_gap",
    }
    assert required_warning.issubset(warning.columns)
    assert len(warning) == 60
    assert not warning.duplicated(["dataset", "seed", "model"]).any()
    assert set(warning["dataset"]) == set(DATASETS)
    assert set(warning["seed"]) == set(SEEDS)
    assert set(warning["model"]) == set(MODELS)
    assert np.allclose(
        warning["environmental_instability"],
        1.0 - warning["environmental_stability"],
        rtol=0.0,
        atol=1e-12,
    )
    assert np.allclose(
        warning["sampling_environment_gap"],
        warning["sampling_stability"] - warning["environmental_stability"],
        rtol=0.0,
        atol=1e-12,
    )
    warning = warning.copy()
    warning["sampling_instability"] = 1.0 - warning["sampling_stability"]

    return {
        "source_model_seal": source_model_seal,
        "source_metrics": source_metrics,
        "source_metrics_sha256": source_metrics_sha,
        "stability_seal_path": stability_seal_path,
        "stability_seal": stability_seal,
        "warning_scores": warning,
        "warning_scores_sha256": warning_scores_sha,
    }


def verify_pre_label_chain():
    candidate_seal = verify_known_self_seal(
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

    direction_seal = verify_known_self_seal(
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
    assert direction_seal["target_candidate_panel_seal_sha256"] == EXPECTED_CANDIDATE_SEAL
    assert direction_seal["directed_transfers"] == 12
    assert direction_seal["direction_specific_indices"] == 60
    assert direction_seal["rows_per_index"] == EXPECTED_ROWS_PER_JOB
    assert direction_seal["final_overlap_total"] == 0
    assert direction_seal["target_predictions_calculated"] is False
    assert direction_seal["target_performance_calculated"] is False

    pre_label_seal = load_json(PRE_LABEL_SEAL_PATH)
    pre_label_payload = {
        k: v for k, v in pre_label_seal.items()
        if k not in {"seal_sha256", "status"}
    }
    observed_pre_label = hashlib.sha256(
        canonical_json_bytes(pre_label_payload)
    ).hexdigest()
    assert observed_pre_label == pre_label_seal["seal_sha256"]
    assert observed_pre_label == EXPECTED_PRE_LABEL_SEAL
    assert pre_label_seal["prediction_script_sha256"] == EXPECTED_PREDICTION_SCRIPT
    assert pre_label_seal["prediction_jobs"] == 180
    assert pre_label_seal["rows_per_job"] == EXPECTED_ROWS_PER_JOB
    assert pre_label_seal["target_label_rows_read"] == 0
    assert pre_label_seal["target_adaptation_performed"] is False
    assert pre_label_seal["target_performance_calculated"] is False
    assert pre_label_seal["status"] == "SEALED_BEFORE_TARGET_LABEL_JOIN"
    assert pre_label_seal["parent_amendment_A2_sha256"] == EXPECTED_A2
    assert pre_label_seal["source_model_seal_sha256"] == EXPECTED_SOURCE_MODEL_SEAL
    assert pre_label_seal["target_candidate_panel_seal_sha256"] == EXPECTED_CANDIDATE_SEAL
    assert pre_label_seal["direction_index_seal_sha256"] == EXPECTED_DIRECTION_SEAL

    assert sha256_file(PREDICTION_INVENTORY_PATH) == pre_label_seal[
        "prediction_inventory_sha256"
    ]
    inventory = pd.read_csv(PREDICTION_INVENTORY_PATH)
    assert len(inventory) == 180
    assert not inventory.duplicated(
        ["source_dataset", "target_dataset", "seed", "model"]
    ).any()
    assert (inventory["target_rows"] == EXPECTED_ROWS_PER_JOB).all()
    assert (inventory["target_label_rows_read"] == 0).all()
    assert (~inventory["target_performance_calculated"].astype(bool)).all()

    expected_job_keys = set()
    direction_indices = {}
    direction_manifest_hashes = {}

    for source in DATASETS:
        for target in DATASETS:
            if target == source:
                continue
            for seed in SEEDS:
                direction_key = f"{source}__to__{target}__seed_{seed}"
                assert direction_key in direction_seal["direction_records"]
                direction_record = direction_seal["direction_records"][direction_key]
                direction_dir = (
                    DIRECTION_ROOT / f"{source}__to__{target}" / f"seed_{seed}"
                )
                index_path = direction_dir / "target_index.parquet"
                direction_manifest_path = direction_dir / "direction_manifest.json"
                assert sha256_file(index_path) == direction_record["target_index_sha256"]
                assert sha256_file(direction_manifest_path) == direction_record[
                    "manifest_sha256"
                ]
                direction_manifest = load_json(direction_manifest_path)
                assert direction_manifest["source_dataset"] == source
                assert direction_manifest["target_dataset"] == target
                assert direction_manifest["seed"] == seed
                assert direction_manifest["final_target_rows"] == EXPECTED_ROWS_PER_JOB
                assert direction_manifest["final_rows_per_class"] == EXPECTED_ROWS_PER_CLASS
                assert direction_manifest["final_source_target_overlap"] == 0
                assert direction_manifest["target_index_contains_label"] is False
                assert direction_manifest["target_performance_calculated"] is False

                index_frame = pd.read_parquet(index_path)
                assert list(index_frame.columns) == ["row_id"]
                assert len(index_frame) == EXPECTED_ROWS_PER_JOB
                assert index_frame["row_id"].is_unique
                direction_indices[direction_key] = index_frame[
                    "row_id"
                ].to_numpy(dtype=np.int64)
                direction_manifest_hashes[direction_key] = direction_record[
                    "manifest_sha256"
                ]

                for model in MODELS:
                    job_key = f"{direction_key}__{model}"
                    expected_job_keys.add(job_key)
                    assert job_key in pre_label_seal["job_records"]
                    job_record = pre_label_seal["job_records"][job_key]
                    prediction_dir = (
                        PREDICTION_ROOT /
                        f"{source}__to__{target}" /
                        f"seed_{seed}"
                    )
                    prediction_path = (
                        prediction_dir / f"{model}_target_predictions.parquet"
                    )
                    prediction_manifest_path = (
                        prediction_dir / f"{model}_prediction_manifest.json"
                    )
                    assert sha256_file(prediction_path) == job_record[
                        "prediction_file_sha256"
                    ]
                    assert sha256_file(prediction_manifest_path) == job_record[
                        "prediction_manifest_sha256"
                    ]
                    manifest = load_json(prediction_manifest_path)
                    assert manifest["source_dataset"] == source
                    assert manifest["target_dataset"] == target
                    assert manifest["seed"] == seed
                    assert manifest["model"] == model
                    assert manifest["target_rows"] == EXPECTED_ROWS_PER_JOB
                    assert manifest["target_label_rows_read"] == 0
                    assert manifest["target_performance_calculated"] is False
                    assert manifest["prediction_contains_target_label"] is False
                    assert manifest["target_adaptation"] is False
                    assert manifest["prediction_script_sha256"] == EXPECTED_PREDICTION_SCRIPT
                    assert manifest["prediction_file_sha256"] == job_record[
                        "prediction_file_sha256"
                    ]

    assert set(pre_label_seal["job_records"]) == expected_job_keys
    assert len(direction_indices) == 60

    return {
        "candidate_seal": candidate_seal,
        "direction_seal": direction_seal,
        "pre_label_seal": pre_label_seal,
        "inventory": inventory,
        "direction_indices": direction_indices,
        "direction_manifest_hashes": direction_manifest_hashes,
    }


def freeze_or_verify_a3():
    seal_sha = hashlib.sha256(canonical_json_bytes(A3_PAYLOAD)).hexdigest()
    record = dict(A3_PAYLOAD)
    record["seal_sha256"] = seal_sha
    record["status"] = "FROZEN_BEFORE_TARGET_LABEL_JOIN"

    if not A3_PATH.exists():
        atomic_write_json(A3_PATH, record)
        assert load_json(A3_PATH) == record
        return seal_sha, True

    observed = load_json(A3_PATH)
    assert observed == record, (
        "Existing A3 differs from the prespecified confirmatory plan"
    )
    return seal_sha, False


def verify_completed_results():
    if not FINAL_SEAL_PATH.exists():
        existing = [p for p in FINAL_ROOT.glob("*") if p.is_file()] \
            if FINAL_ROOT.exists() else []
        assert not existing, (
            "Partial final-evaluation artifacts exist without a final seal. "
            "Stop and audit before rerunning: " + str([p.name for p in existing])
        )
        return None

    seal = load_json(FINAL_SEAL_PATH)
    payload = {k: v for k, v in seal.items() if k not in {"seal_sha256", "status"}}
    observed = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    assert observed == seal["seal_sha256"]
    assert seal["parent_amendment_A2_sha256"] == EXPECTED_A2
    assert seal["pre_label_target_prediction_seal_sha256"] == EXPECTED_PRE_LABEL_SEAL
    assert seal["prediction_jobs_evaluated"] == 180
    for record in seal["artifacts"].values():
        path = Path(record["path"])
        assert path.exists()
        assert sha256_file(path) == record["sha256"]
    return seal


def binary_metrics(labels, predictions, probabilities):
    labels = np.asarray(labels, dtype=np.int8)
    predictions = np.asarray(predictions, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    assert labels.shape == predictions.shape == probabilities.shape
    assert np.isin(labels, [0, 1]).all()
    assert np.isin(predictions, [0, 1]).all()
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()

    tn = int(((labels == 0) & (predictions == 0)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    tp = int(((labels == 1) & (predictions == 1)).sum())
    assert tn + fp == EXPECTED_ROWS_PER_CLASS
    assert fn + tp == EXPECTED_ROWS_PER_CLASS

    return {
        "target_balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
        "target_macro_f1": float(f1_score(labels, predictions, average="macro")),
        "target_mcc": float(matthews_corrcoef(labels, predictions)),
        "target_pr_auc_attack": float(
            average_precision_score(labels, probabilities)
        ),
        "target_benign_recall": float(
            recall_score(labels, predictions, pos_label=0)
        ),
        "target_attack_recall": float(
            recall_score(labels, predictions, pos_label=1)
        ),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def stratified_bootstrap_rho(frame, predictor, outcome):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    grouped = [
        group.reset_index(drop=True)
        for _, group in frame.groupby("model", sort=True)
    ]
    assert len(grouped) == 3
    assert all(len(group) == 4 for group in grouped)
    values = []
    for _ in range(BOOTSTRAP_REPLICATES):
        samples = []
        for group in grouped:
            positions = rng.integers(0, len(group), size=len(group))
            samples.append(group.iloc[positions])
        sample = pd.concat(samples, ignore_index=True)
        rho = spearmanr(sample[predictor], sample[outcome]).statistic
        if np.isfinite(rho):
            values.append(float(rho))
    assert len(values) >= int(0.95 * BOOTSTRAP_REPLICATES)
    return (
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
        len(values),
    )


def exact_blocked_permutation_p(frame, predictor, outcome):
    ordered_groups = []
    for model in sorted(MODELS):
        group = frame.loc[frame["model"] == model].sort_values(
            "source_dataset"
        ).reset_index(drop=True)
        assert len(group) == 4
        ordered_groups.append(group)
    ordered = pd.concat(ordered_groups, ignore_index=True)
    observed = float(spearmanr(ordered[predictor], ordered[outcome]).statistic)
    assert np.isfinite(observed)

    permutation_set = list(permutations(range(4)))
    exceedances = 0
    total = 0
    for choices in product(permutation_set, repeat=3):
        permuted_parts = []
        for group_index, choice in enumerate(choices):
            values = ordered_groups[group_index][predictor].to_numpy()
            permuted_parts.append(values[list(choice)])
        permuted_predictor = np.concatenate(permuted_parts)
        rho = spearmanr(permuted_predictor, ordered[outcome]).statistic
        if np.isfinite(rho) and rho >= observed - 1e-15:
            exceedances += 1
        total += 1
    assert total == 13_824
    return observed, float(exceedances / total), exceedances, total


def holm_adjust(raw_p_values):
    names = list(raw_p_values)
    order = sorted(names, key=lambda name: raw_p_values[name])
    adjusted = {}
    running = 0.0
    count = len(order)
    for rank, name in enumerate(order):
        candidate = min(1.0, (count - rank) * raw_p_values[name])
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def build_primary_units(analysis):
    warning_columns = [
        "environmental_instability",
        "sampling_environment_gap",
        "sampling_instability",
        "environmental_stability",
        "sampling_stability",
    ]
    unit60 = analysis.groupby(
        ["source_dataset", "seed", "model"], as_index=False
    ).agg(
        mean_balanced_accuracy_loss=("balanced_accuracy_loss", "mean"),
        sd_balanced_accuracy_loss=("balanced_accuracy_loss", "std"),
        min_balanced_accuracy_loss=("balanced_accuracy_loss", "min"),
        max_balanced_accuracy_loss=("balanced_accuracy_loss", "max"),
        target_datasets=("target_dataset", "nunique"),
        **{column: (column, "first") for column in warning_columns},
    )
    assert len(unit60) == 60
    assert (unit60["target_datasets"] == 3).all()

    unit12 = unit60.groupby(
        ["source_dataset", "model"], as_index=False
    ).agg(
        mean_balanced_accuracy_loss=("mean_balanced_accuracy_loss", "mean"),
        sd_seed_mean_balanced_accuracy_loss=(
            "mean_balanced_accuracy_loss", "std"
        ),
        seeds=("seed", "nunique"),
        environmental_instability=("environmental_instability", "mean"),
        sampling_environment_gap=("sampling_environment_gap", "mean"),
        sampling_instability=("sampling_instability", "mean"),
        environmental_stability=("environmental_stability", "mean"),
        sampling_stability=("sampling_stability", "mean"),
    )
    assert len(unit12) == 12
    assert (unit12["seeds"] == 5).all()
    return unit60, unit12


def run_confirmatory_associations(unit12):
    outcome = "mean_balanced_accuracy_loss"
    predictors = [
        "environmental_instability",
        "sampling_environment_gap",
        "sampling_instability",
    ]
    rows = []
    raw_primary = {}
    for predictor in predictors:
        rho, p_value, exceedances, total = exact_blocked_permutation_p(
            unit12, predictor, outcome
        )
        ci_low, ci_high, finite_bootstraps = stratified_bootstrap_rho(
            unit12, predictor, outcome
        )
        is_primary = predictor in {
            "environmental_instability", "sampling_environment_gap"
        }
        if is_primary:
            raw_primary[predictor] = p_value
        rows.append({
            "predictor": predictor,
            "role": "primary" if is_primary else "baseline",
            "n_source_model_units": 12,
            "spearman_rho": rho,
            "bootstrap_95_ci_low": ci_low,
            "bootstrap_95_ci_high": ci_high,
            "finite_bootstrap_replicates": finite_bootstraps,
            "exact_one_sided_p": p_value,
            "permutation_exceedances": exceedances,
            "permutations": total,
        })

    adjusted = holm_adjust(raw_primary)
    for row in rows:
        predictor = row["predictor"]
        if predictor in adjusted:
            row["holm_adjusted_p_primary_family"] = adjusted[predictor]
            row["association_supported_at_0_05"] = bool(
                row["spearman_rho"] > 0.0 and adjusted[predictor] < ALPHA
            )
        else:
            row["holm_adjusted_p_primary_family"] = np.nan
            row["association_supported_at_0_05"] = False
    return pd.DataFrame(rows)


def design_matrix(frame, continuous_columns, include_model):
    parts = []
    names = []
    if include_model:
        for model in ["RandomForest", "XGBoost"]:
            parts.append((frame["model"].to_numpy() == model).astype(float))
            names.append(f"model_{model}")
    for column in continuous_columns:
        parts.append(frame[column].to_numpy(dtype=float))
        names.append(column)
    if not parts:
        return np.empty((len(frame), 0), dtype=float), names
    return np.column_stack(parts), names


def run_loso_validation(unit60):
    outcome = "mean_balanced_accuracy_loss"
    specifications = {
        "mean_only": {"model": False, "continuous": []},
        "model_only": {"model": True, "continuous": []},
        "model_plus_sampling_instability": {
            "model": True,
            "continuous": ["sampling_instability"],
        },
        "model_plus_environmental_instability": {
            "model": True,
            "continuous": ["environmental_instability"],
        },
        "model_plus_sampling_environment_gap": {
            "model": True,
            "continuous": ["sampling_environment_gap"],
        },
        "model_plus_primary_combined": {
            "model": True,
            "continuous": [
                "environmental_instability",
                "sampling_environment_gap",
            ],
        },
    }

    prediction_rows = []
    for held_out_source in DATASETS:
        train = unit60.loc[
            unit60["source_dataset"] != held_out_source
        ].copy()
        test = unit60.loc[
            unit60["source_dataset"] == held_out_source
        ].copy()
        assert len(train) == 45 and len(test) == 15
        y_train = train[outcome].to_numpy(dtype=float)
        y_test = test[outcome].to_numpy(dtype=float)

        for specification, config in specifications.items():
            continuous = config["continuous"]
            train_work = train.copy()
            test_work = test.copy()
            for column in continuous:
                mean = float(train[column].mean())
                std = float(train[column].std(ddof=0))
                if not np.isfinite(std) or std == 0.0:
                    std = 1.0
                train_work[column] = (train[column] - mean) / std
                test_work[column] = (test[column] - mean) / std

            if specification == "mean_only":
                predicted = np.full(len(test), y_train.mean(), dtype=float)
            else:
                x_train, train_names = design_matrix(
                    train_work, continuous, config["model"]
                )
                x_test, test_names = design_matrix(
                    test_work, continuous, config["model"]
                )
                assert train_names == test_names
                estimator = LinearRegression()
                estimator.fit(x_train, y_train)
                predicted = estimator.predict(x_test)

            for position, (_, row) in enumerate(test.iterrows()):
                prediction_rows.append({
                    "specification": specification,
                    "held_out_source_dataset": held_out_source,
                    "source_dataset": row["source_dataset"],
                    "seed": int(row["seed"]),
                    "model": row["model"],
                    "actual_mean_balanced_accuracy_loss": float(y_test[position]),
                    "predicted_mean_balanced_accuracy_loss": float(
                        predicted[position]
                    ),
                })

    predictions = pd.DataFrame(prediction_rows)
    assert len(predictions) == 60 * len(specifications)
    score_rows = []
    for specification, frame in predictions.groupby("specification", sort=True):
        actual = frame["actual_mean_balanced_accuracy_loss"].to_numpy()
        predicted = frame["predicted_mean_balanced_accuracy_loss"].to_numpy()
        rho = spearmanr(actual, predicted).statistic
        score_rows.append({
            "specification": specification,
            "oof_units": len(frame),
            "oof_mae": float(mean_absolute_error(actual, predicted)),
            "oof_rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
            "oof_spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
        })
    scores = pd.DataFrame(score_rows).sort_values("specification").reset_index(
        drop=True
    )
    baseline_mae = float(
        scores.loc[
            scores["specification"] == "model_plus_sampling_instability",
            "oof_mae",
        ].iloc[0]
    )
    scores["mae_improvement_vs_sampling_baseline"] = (
        baseline_mae - scores["oof_mae"]
    )
    return predictions, scores


def artifact_record(path, rows=None):
    record = {
        "path": str(path),
        "sha256": sha256_file(path),
    }
    if rows is not None:
        record["rows"] = int(rows)
    return record


def main():
    print("=" * 92)
    print("PAPER 3 — FINAL TARGET EVALUATION AND CONFIRMATORY ANALYSIS")
    print("=" * 92)

    started = time.time()
    a2 = verify_a2()
    print("Amendment A2 verification: PASS")

    source_inputs = verify_source_only_inputs()
    print("Source-model and source-stability chain: PASS")

    pre_label = verify_pre_label_chain()
    print("All 180 sealed target predictions verified before label access: PASS")
    print("Target label rows read so far: 0")

    a3_sha, created_now = freeze_or_verify_a3()
    print("Amendment A3 confirmatory-analysis seal SHA-256:", a3_sha)

    if created_now:
        print("Amendment A3 timing: FROZEN BEFORE TARGET LABEL JOIN")
        print("Target label rows read: 0")
        print("Target performance calculated: NO")
        print("\nFIRST-RUN GATE: PASS")
        print("Run this same script one more time to perform the authorized label reveal.")
        print("=" * 92)
        return

    completed = verify_completed_results()
    if completed is not None:
        print("Amendment A3 verification: PASS")
        print("Final target evaluation already completed and sealed: PASS")
        print("Final-results seal SHA-256:", completed["seal_sha256"])
        print("No target labels were reread during this verification run.")
        print("=" * 92)
        return

    print("Amendment A3 verification: PASS")
    print("Target-label reveal is now authorized by the frozen A2→prediction→A3 chain.")

    # This is the first semantic read of the separated target-label vault.
    label_frames = {}
    unique_label_rows_read = 0
    for dataset in DATASETS:
        label_path = (
            LABEL_VAULT_ROOT / dataset / "target_candidate_labels.parquet"
        )
        assert label_path.exists()
        assert sha256_file(label_path) == pre_label["candidate_seal"][
            "label_vault_hashes"
        ][dataset]
        labels = pd.read_parquet(label_path)
        assert list(labels.columns) == ["row_id", "Label"]
        assert len(labels) == 100_000
        assert labels["row_id"].is_unique
        assert labels["Label"].value_counts().to_dict() == {
            0: 50_000,
            1: 50_000,
        }
        labels = labels.set_index("row_id", verify_integrity=True)
        label_frames[dataset] = labels
        unique_label_rows_read += len(labels)
    assert unique_label_rows_read == 400_000
    print("Target label vaults opened after A3 freeze: PASS")

    target_metric_rows = []
    evaluated_label_associations = 0
    for source in DATASETS:
        print("\nSOURCE DATASET:", source)
        for target in DATASETS:
            if target == source:
                continue
            target_labels = label_frames[target]
            for seed in SEEDS:
                direction_key = f"{source}__to__{target}__seed_{seed}"
                row_ids = pre_label["direction_indices"][direction_key]
                selected_labels = target_labels.loc[row_ids, "Label"].to_numpy(
                    dtype=np.int8
                )
                assert selected_labels.shape == (EXPECTED_ROWS_PER_JOB,)
                assert pd.Series(selected_labels).value_counts().to_dict() == {
                    0: EXPECTED_ROWS_PER_CLASS,
                    1: EXPECTED_ROWS_PER_CLASS,
                }

                for model in MODELS:
                    job_key = f"{direction_key}__{model}"
                    job_record = pre_label["pre_label_seal"]["job_records"][job_key]
                    prediction_dir = (
                        PREDICTION_ROOT /
                        f"{source}__to__{target}" /
                        f"seed_{seed}"
                    )
                    prediction_path = (
                        prediction_dir / f"{model}_target_predictions.parquet"
                    )
                    assert sha256_file(prediction_path) == job_record[
                        "prediction_file_sha256"
                    ]
                    predictions = pd.read_parquet(prediction_path)
                    assert list(predictions.columns) == [
                        "row_id", "attack_probability", "predicted_label"
                    ]
                    assert len(predictions) == EXPECTED_ROWS_PER_JOB
                    assert np.array_equal(
                        predictions["row_id"].to_numpy(dtype=np.int64),
                        row_ids,
                    )
                    metrics = binary_metrics(
                        selected_labels,
                        predictions["predicted_label"].to_numpy(dtype=np.int8),
                        predictions["attack_probability"].to_numpy(
                            dtype=np.float64
                        ),
                    )
                    target_metric_rows.append({
                        "source_dataset": source,
                        "target_dataset": target,
                        "seed": seed,
                        "model": model,
                        "target_rows": EXPECTED_ROWS_PER_JOB,
                        "target_benign_rows": EXPECTED_ROWS_PER_CLASS,
                        "target_attack_rows": EXPECTED_ROWS_PER_CLASS,
                        **metrics,
                        "prediction_file_sha256": job_record[
                            "prediction_file_sha256"
                        ],
                    })
                    evaluated_label_associations += EXPECTED_ROWS_PER_JOB
        print(f"  {source}: 45 target jobs evaluated — PASS")

    target_metrics = pd.DataFrame(target_metric_rows).sort_values(
        ["source_dataset", "target_dataset", "seed", "model"]
    ).reset_index(drop=True)
    assert len(target_metrics) == 180
    assert evaluated_label_associations == 14_400_000

    source_metrics = source_inputs["source_metrics"][[
        "dataset", "seed", "model", "balanced_accuracy"
    ]].rename(columns={
        "dataset": "source_dataset",
        "balanced_accuracy": "source_test_balanced_accuracy",
    })
    warning = source_inputs["warning_scores"].rename(
        columns={"dataset": "source_dataset"}
    )
    analysis = target_metrics.merge(
        source_metrics,
        on=["source_dataset", "seed", "model"],
        how="left",
        validate="many_to_one",
    ).merge(
        warning,
        on=["source_dataset", "seed", "model"],
        how="left",
        validate="many_to_one",
    )
    assert len(analysis) == 180
    assert not analysis[[
        "source_test_balanced_accuracy",
        "environmental_instability",
        "sampling_environment_gap",
        "sampling_instability",
    ]].isna().any().any()
    analysis["balanced_accuracy_loss"] = (
        analysis["source_test_balanced_accuracy"] -
        analysis["target_balanced_accuracy"]
    )
    # Negative loss values are intentionally retained as frozen in A3.

    unit60, unit12 = build_primary_units(analysis)
    associations = run_confirmatory_associations(unit12)
    loso_predictions, loso_scores = run_loso_validation(unit60)

    summary_metrics = [
        "target_balanced_accuracy",
        "target_macro_f1",
        "target_mcc",
        "target_pr_auc_attack",
        "target_benign_recall",
        "target_attack_recall",
        "balanced_accuracy_loss",
    ]
    aggregation = {}
    for column in summary_metrics:
        aggregation[f"{column}_mean"] = (column, "mean")
        aggregation[f"{column}_sd"] = (column, "std")
        aggregation[f"{column}_min"] = (column, "min")
        aggregation[f"{column}_max"] = (column, "max")
    direction_summary = analysis.groupby(
        ["source_dataset", "target_dataset", "model"], as_index=False
    ).agg(seeds=("seed", "nunique"), **aggregation)
    assert len(direction_summary) == 36
    assert (direction_summary["seeds"] == 5).all()

    target_metrics_path = FINAL_ROOT / "target_metrics_all_180.csv"
    analysis_path = FINAL_ROOT / "transfer_analysis_all_180.csv"
    unit60_path = FINAL_ROOT / "source_seed_model_primary_units_60.csv"
    unit12_path = FINAL_ROOT / "source_model_confirmatory_units_12.csv"
    association_path = FINAL_ROOT / "confirmatory_association_results.csv"
    loso_prediction_path = FINAL_ROOT / "loso_oof_predictions.csv"
    loso_score_path = FINAL_ROOT / "loso_model_scores.csv"
    direction_summary_path = FINAL_ROOT / "direction_model_summary_36.csv"
    summary_path = FINAL_ROOT / "final_analysis_summary.json"

    atomic_to_csv(target_metrics, target_metrics_path)
    atomic_to_csv(analysis, analysis_path)
    atomic_to_csv(unit60, unit60_path)
    atomic_to_csv(unit12, unit12_path)
    atomic_to_csv(associations, association_path)
    atomic_to_csv(loso_predictions, loso_prediction_path)
    atomic_to_csv(loso_scores, loso_score_path)
    atomic_to_csv(direction_summary, direction_summary_path)

    global_summary = {
        metric: {
            "mean": float(analysis[metric].mean()),
            "sd": float(analysis[metric].std()),
            "min": float(analysis[metric].min()),
            "max": float(analysis[metric].max()),
        }
        for metric in summary_metrics
    }
    summary = {
        "artifact": "paper3_final_analysis_summary",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "parent_amendment_A2_sha256": EXPECTED_A2,
        "amendment_A3_confirmatory_analysis_seal_sha256": a3_sha,
        "pre_label_target_prediction_seal_sha256": EXPECTED_PRE_LABEL_SEAL,
        "prediction_jobs_evaluated": 180,
        "unique_target_label_rows_read": unique_label_rows_read,
        "evaluated_job_label_associations": evaluated_label_associations,
        "target_adaptation_performed": False,
        "threshold_changed_using_target_results": False,
        "global_descriptive_summary": global_summary,
        "confirmatory_associations": associations.replace(
            {np.nan: None}
        ).to_dict(orient="records"),
        "loso_scores": loso_scores.replace(
            {np.nan: None}
        ).to_dict(orient="records"),
        "scope_caution": A3_PAYLOAD["scope_caution"],
        "negative_results_reporting_rule": (
            "All prespecified results are retained regardless of direction or "
            "statistical significance."
        ),
        "status": "PASS",
    }
    atomic_write_json(summary_path, summary)

    artifacts = {
        "target_metrics_all_180": artifact_record(target_metrics_path, 180),
        "transfer_analysis_all_180": artifact_record(analysis_path, 180),
        "source_seed_model_primary_units_60": artifact_record(unit60_path, 60),
        "source_model_confirmatory_units_12": artifact_record(unit12_path, 12),
        "confirmatory_association_results": artifact_record(
            association_path, len(associations)
        ),
        "loso_oof_predictions": artifact_record(
            loso_prediction_path, len(loso_predictions)
        ),
        "loso_model_scores": artifact_record(loso_score_path, len(loso_scores)),
        "direction_model_summary_36": artifact_record(direction_summary_path, 36),
        "final_analysis_summary": artifact_record(summary_path),
    }

    script_path = Path(__file__).resolve()
    software_versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
    }
    final_payload = {
        "artifact": "paper3_final_results_seal",
        "parent_protocol_sha256": EXPECTED_PROTOCOL,
        "amendment_A1_sha256": EXPECTED_A1,
        "feature_schema_sha256": EXPECTED_SCHEMA,
        "parent_amendment_A2_sha256": EXPECTED_A2,
        "amendment_A3_confirmatory_analysis_seal_sha256": a3_sha,
        "source_model_seal_sha256": EXPECTED_SOURCE_MODEL_SEAL,
        "source_stability_seal_sha256": EXPECTED_SOURCE_STABILITY_SEAL,
        "target_candidate_panel_seal_sha256": EXPECTED_CANDIDATE_SEAL,
        "direction_index_seal_sha256": EXPECTED_DIRECTION_SEAL,
        "pre_label_target_prediction_seal_sha256": EXPECTED_PRE_LABEL_SEAL,
        "final_evaluation_script_sha256": sha256_file(script_path),
        "prediction_jobs_evaluated": 180,
        "unique_target_label_rows_read": unique_label_rows_read,
        "evaluated_job_label_associations": evaluated_label_associations,
        "target_adaptation_performed": False,
        "target_thresholds_changed": False,
        "all_negative_results_retained": True,
        "artifacts": artifacts,
        "software_versions": software_versions,
    }
    final_seal_sha = hashlib.sha256(
        canonical_json_bytes(final_payload)
    ).hexdigest()
    final_record = dict(final_payload)
    final_record["seal_sha256"] = final_seal_sha
    final_record["status"] = "FINAL_TARGET_EVALUATION_SEALED"
    atomic_write_json(FINAL_SEAL_PATH, final_record)

    elapsed_minutes = (time.time() - started) / 60.0
    print("\n" + "=" * 92)
    print("PAPER 3 FINAL TARGET EVALUATION AND RESULTS SEAL: PASS")
    print("Prediction jobs evaluated: 180")
    print("Unique target-label vault rows read:", f"{unique_label_rows_read:,}")
    print(
        "Evaluated job-label associations:",
        f"{evaluated_label_associations:,}",
    )
    print("Target adaptation performed: NO")
    print("Target thresholds changed: NO")
    print("A3 confirmatory-analysis seal SHA-256:", a3_sha)
    print("Final-results seal SHA-256:", final_seal_sha)
    print("Results directory:", FINAL_ROOT)
    print("Elapsed minutes:", round(elapsed_minutes, 2))
    print("=" * 92)


if __name__ == "__main__":
    main()
