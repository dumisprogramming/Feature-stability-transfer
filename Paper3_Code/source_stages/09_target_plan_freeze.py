from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
from itertools import permutations

ROOT = Path(
    "/content/drive/MyDrive/research/researchdata/"
    "NIDS_Research/Paper3_DataSet"
)

PROTOCOL_DIR = ROOT / "Paper3_Protocol"
MATERIALIZED_DIR = ROOT / "Paper3_Materialized"
MODEL_DIR = ROOT / "Paper3_Source_Models"
STABILITY_DIR = ROOT / "Paper3_Source_Stability"

EXPECTED_HASHES = {
    "protocol": (
        "e39a5007a7f2cbdc348d5179ce593af"
        "5766bc6688f5baa3a46a56399f683a488"
    ),
    "schema": (
        "67d61eebb3377e3053f943f1e4c03123"
        "7b3ef35dd06296d8538fe552b16be0c6"
    ),
    "amendment_A1": (
        "19c30a37d62d36f5b3c9bb795f64e105"
        "99dcef78a9c51bf96ca80b58d7db4b47"
    ),
    "source_model_seal": (
        "81eebf7b013fec0ca1802ebb54894fe73"
        "d7023b2c6450da09c72af3b3f5a3182"
    ),
    "source_stability_seal": (
        "3ec5641ed96a462ce2c76765b6b4483a"
        "1e48cad11d79981feb7afd1c43449340"
    ),
}

ARTIFACTS = {
    "protocol": (
        PROTOCOL_DIR
        / "paper3_protocol_frozen_v1.json"
    ),
    "schema": (
        MATERIALIZED_DIR
        / "feature_schema_v1.json"
    ),
    "amendment_A1": (
        PROTOCOL_DIR
        / "amendment_A1_duplicate_group_disjoint.json"
    ),
    "source_model_seal": (
        MODEL_DIR
        / "source_model_seal_manifest.json"
    ),
    "source_stability_seal": (
        STABILITY_DIR
        / "source_only_stability_seal_manifest.json"
    ),
}

DATASETS = [
    "NF_BoT_IoT_v3",
    "NF_CSE_CIC_IDS2018_v3",
    "NF_ToN_IoT_v3",
    "NF_UNSW_NB15_v3",
]


def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


checks = {
    name: sha256_file(path)
    == EXPECTED_HASHES[name]
    for name, path in ARTIFACTS.items()
}

print("=" * 76)
print("PAPER 3 — TARGET-EVALUATION PLAN FREEZE")
print("=" * 76)
print(f"Artifact-chain verification: {checks}")

if not all(checks.values()):
    raise RuntimeError(
        "Artifact-chain verification failed."
    )

directions = [
    {
        "source": source,
        "target": target,
    }
    for source, target in permutations(
        DATASETS,
        2,
    )
]

amendment_file = (
    PROTOCOL_DIR
    / "amendment_A2_target_evaluation_plan.json"
)

hash_file = (
    PROTOCOL_DIR
    / "amendment_A2_target_evaluation_plan.sha256"
)

if amendment_file.exists():
    amendment_hash = sha256_file(
        amendment_file
    )

    recorded_hash = (
        hash_file.read_text(
            encoding="utf-8"
        ).strip()
    )

    if amendment_hash != recorded_hash:
        raise RuntimeError(
            "Existing Amendment A2 hash mismatch."
        )

else:
    amendment = {
        "amendment_id": (
            "P3_A2_TARGET_EVALUATION_PLAN"
        ),
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "timing": (
            "Frozen after all source-only warning scores "
            "were cryptographically sealed and before any "
            "Paper 3 cross-dataset target performance was "
            "calculated."
        ),
        "parent_protocol_sha256": (
            EXPECTED_HASHES["protocol"]
        ),
        "amendment_A1_sha256": (
            EXPECTED_HASHES["amendment_A1"]
        ),
        "feature_schema_sha256": (
            EXPECTED_HASHES["schema"]
        ),
        "source_model_seal_sha256": (
            EXPECTED_HASHES[
                "source_model_seal"
            ]
        ),
        "source_stability_seal_sha256": (
            EXPECTED_HASHES[
                "source_stability_seal"
            ]
        ),
        "evaluation_design": {
            "datasets": DATASETS,
            "directed_transfers": directions,
            "number_of_directions": 12,
            "source_seeds": [
                2027,
                2028,
                2029,
                2030,
                2031,
            ],
            "models": [
                "LogisticRegression",
                "RandomForest",
                "XGBoost",
            ],
            "prediction_jobs": 180,
        },
        "target_candidate_panel": {
            "selection_seed": 42027,
            "candidate_rows_per_class": 50000,
            "total_candidate_rows_per_target": (
                100000
            ),
            "selection": (
                "Deterministic lowest row-hash priority "
                "within each target label."
            ),
            "replacement": False,
        },
        "direction_specific_panel": {
            "final_rows_per_class": 40000,
            "final_rows_per_direction_seed": (
                80000
            ),
            "shared_across_models": True,
            "cross_dataset_duplicate_rule": (
                "Exclude target feature signatures matching "
                "any source train, calibration, source-test, "
                "or environmental signature for that source "
                "dataset and seed."
            ),
            "insufficient_rows_rule": (
                "Hard stop if fewer than 40000 rows remain "
                "for either class; do not reduce the quota "
                "after viewing target outcomes."
            ),
        },
        "target_processing": {
            "predictors": 49,
            "target_fitted_imputation": False,
            "target_fitted_scaling": False,
            "target_threshold_calibration": False,
            "use_frozen_source_imputer": True,
            "use_frozen_source_scaler_for_LR": True,
            "use_frozen_source_threshold": True,
            "target_adaptation": False,
        },
        "label_separation": {
            "candidate_features_and_labels_saved_separately": True,
            "prediction_code_must_not_read_label_vault": True,
            "predictions_sealed_before_label_join": True,
            "target_labels_used_for": (
                "Balanced evaluation-panel construction "
                "and final evaluation only."
            ),
        },
        "primary_outcome": {
            "metric": "balanced_accuracy_loss",
            "formula": (
                "source_test_balanced_accuracy "
                "- target_balanced_accuracy"
            ),
        },
        "secondary_outcomes": [
            "target balanced accuracy",
            "target Macro F1",
            "target MCC",
            "target PR-AUC",
            "benign recall",
            "attack recall",
        ],
        "primary_warning_predictors": [
            "environmental_instability",
            "sampling_environment_gap",
        ],
        "baseline_warning_predictor": (
            "sampling instability"
        ),
        "confirmatory_rules": {
            "all_60_warning_scores_already_sealed": True,
            "no_model_selection_using_target_results": True,
            "no_threshold_change_using_target_results": True,
            "all_180_prediction_results_reported": True,
            "negative_results_reported": True,
        },
        "paper3_target_results_seen_before_freeze": False,
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

    amendment_hash = sha256_file(
        amendment_file
    )

    hash_file.write_text(
        amendment_hash,
        encoding="utf-8",
    )

print("Target-evaluation plan: FROZEN")
print(f"Directed transfers: {len(directions)}")
print("Prediction jobs: 180")
print(f"Amendment A2 SHA-256: {amendment_hash}")
print("Paper 3 target results previously calculated: NO")
print("=" * 76)