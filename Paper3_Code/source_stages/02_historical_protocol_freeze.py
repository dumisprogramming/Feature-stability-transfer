from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

ROOT = Path(
    "/content/drive/MyDrive/research/researchdata/"
    "NIDS_Research/Paper3_DataSet"
)

PROTOCOL_DIR = ROOT / "Paper3_Protocol"
PROTOCOL_DIR.mkdir(parents=True, exist_ok=True)

protocol = {
    "protocol_id": "P3_FEATURE_STABILITY_V1",
    "status": "FROZEN_BEFORE_MODELLING",
    "created_utc": datetime.now(timezone.utc).isoformat(),

    "working_title": (
        "Does Source-Side Feature Stability Warn of "
        "Cross-Dataset NIDS Failure?"
    ),

    "primary_question": (
        "Does feature-importance stability across genuine source "
        "temporal environments predict cross-dataset failure better "
        "than stability across random source samples?"
    ),

    "datasets": {
        "NF_BoT_IoT_v3": {
            "rows": 16933808,
            "eligible_6h_blocks": 5,
        },
        "NF_CSE_CIC_IDS2018_v3": {
            "rows": 20115529,
            "eligible_6h_blocks": 13,
        },
        "NF_ToN_IoT_v3": {
            "rows": 27520260,
            "eligible_6h_blocks": 21,
        },
        "NF_UNSW_NB15_v3": {
            "rows": 2365424,
            "eligible_6h_blocks": 4,
        },
    },

    "task": {
        "type": "binary_classification",
        "benign_label": 0,
        "attack_label": 1,
        "transfer_directions": 12,
    },

    "environment_definition": {
        "timestamp": "FLOW_START_MILLISECONDS",
        "timezone": "UTC",
        "primary_unit": "non_overlapping_6_hour_block",
        "block_operation": "floor timestamp to 6-hour boundary",
        "minimum_benign_per_block": 2000,
        "minimum_attack_per_block": 2000,
        "included_blocks": "all eligible blocks",
        "total_eligible_blocks": 43,
        "calendar_day_analysis": "secondary_sensitivity",
    },

    "features": {
        "available_predictors": 53,
        "excluded_from_models": [
            "FLOW_START_MILLISECONDS",
            "FLOW_END_MILLISECONDS",
            "IPV4_SRC_ADDR",
            "IPV4_DST_ADDR",
        ],
        "primary_predictor_count": 49,
        "timestamp_usage": "environment_definition_only",
        "ip_address_usage": "excluded_completely",
        "label_columns": [
            "Label",
            "Attack",
        ],
    },

    "sampling": {
        "environment_sample_per_block": {
            "benign": 2000,
            "attack": 2000,
            "replacement": False,
        },
        "sampling_stability_control": (
            "Use the same observations as the environmental samples, "
            "but randomly redistribute observations within each class "
            "into matched pseudo-blocks."
        ),
        "source_training_per_class": 25000,
        "source_calibration_per_class": 5000,
        "source_test_per_class": 5000,
        "sampling_method": "deterministic_seeded_without_replacement",
        "seeds": [2027, 2028, 2029, 2030, 2031],
    },

    "preprocessing": {
        "fit_using": "source_training_only",
        "infinite_values": "convert_to_missing",
        "imputation": "source_training_median",
        "logistic_regression_scaling": "StandardScaler",
        "tree_model_scaling": "none",
        "target_adaptation": False,
        "target_imputation_refitting": False,
        "target_scaling_refitting": False,
    },

    "models": {
        "LogisticRegression": {
            "solver": "liblinear",
            "max_iter": 2000,
            "class_weight": None,
        },
        "RandomForest": {
            "n_estimators": 300,
            "max_features": "sqrt",
            "min_samples_leaf": 2,
            "n_jobs": -1,
        },
        "XGBoost": {
            "n_estimators": 600,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
        },
    },

    "threshold": {
        "selection_data": "source_calibration_only",
        "rule": "maximize balanced accuracy",
        "tie_break": "threshold closest to 0.5",
        "target_recalibration": False,
    },

    "feature_importance": {
        "primary_method": "permutation_importance",
        "scoring": "balanced_accuracy",
        "permutation_repeats": 3,
        "primary_stability_metric": (
            "mean pairwise Spearman rank correlation"
        ),
        "secondary_stability_metric": "top_10_Jaccard",
        "shap_usage": "XGBoost verification only",
    },

    "warning_scores": {
        "environmental_instability": "1 - environmental_stability",
        "false_reassurance_gap": (
            "sampling_stability - environmental_stability"
        ),
        "interpretation": (
            "Higher values indicate greater source-only warning risk."
        ),
    },

    "outcomes": {
        "primary_transfer_failure": (
            "source balanced accuracy - target balanced accuracy"
        ),
        "secondary_metrics": [
            "Macro F1",
            "MCC",
            "PR-AUC",
        ],
        "target_threshold": (
            "Use frozen source-calibrated threshold without adaptation"
        ),
    },

    "confirmatory_rules": {
        "all_source_warning_scores_frozen_before_target_evaluation": True,
        "all_source_models_frozen_before_target_evaluation": True,
        "target_features_not_used_for_warning_construction": True,
        "target_labels_not_used_for_warning_construction": True,
        "negative_and_null_results_reported": True,
    },

    "scope_separation": {
        "semantic_feature_group_masking": False,
        "artificial_feature_loss": False,
        "selective_classification": False,
        "target_domain_adaptation": False,
        "new_deep_learning_architecture": False,
    },

    "robustness_checks": [
        "calendar-day stability where sufficient",
        "top-10 Jaccard stability",
        "XGBoost SHAP verification",
        "exact-feature-duplicate sensitivity audit",
    ],
}

protocol_text = json.dumps(
    protocol,
    indent=2,
    sort_keys=True,
    ensure_ascii=False,
)

protocol_file = (
    PROTOCOL_DIR / "paper3_protocol_frozen_v1.json"
)

protocol_file.write_text(
    protocol_text,
    encoding="utf-8",
)

protocol_hash = hashlib.sha256(
    protocol_text.encode("utf-8")
).hexdigest()

hash_file = (
    PROTOCOL_DIR / "paper3_protocol_frozen_v1.sha256"
)

hash_file.write_text(
    protocol_hash + "\n",
    encoding="utf-8",
)

# Verification
saved_text = protocol_file.read_text(encoding="utf-8")
verified_hash = hashlib.sha256(
    saved_text.encode("utf-8")
).hexdigest()

print("=" * 72)
print("PAPER 3 PROTOCOL FREEZE")
print("=" * 72)
print(f"Protocol: {protocol_file}")
print(f"SHA-256: {protocol_hash}")
print(
    "Verification:",
    "PASS" if verified_hash == protocol_hash else "FAIL",
)
print("Status: FROZEN BEFORE MODELLING")