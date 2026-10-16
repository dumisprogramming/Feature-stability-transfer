from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost

from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    recall_score,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(
    "/content/drive/MyDrive/research/researchdata/"
    "NIDS_Research/Paper3_DataSet"
)

PROTOCOL_DIR = ROOT / "Paper3_Protocol"
MATERIALIZED_DIR = ROOT / "Paper3_Materialized"
SANITIZED_DIR = ROOT / "Paper3_Sanitized"
MODEL_DIR = ROOT / "Paper3_Source_Models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

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


def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for part in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(part)

    return digest.hexdigest()


def atomic_joblib(obj, path):
    temporary = path.with_suffix(".temporary.joblib")
    joblib.dump(obj, temporary, compress=3)
    temporary.replace(path)


def atomic_parquet(frame, path):
    temporary = path.with_suffix(".temporary.parquet")

    frame.to_parquet(
        temporary,
        index=False,
        engine="pyarrow",
        compression="zstd",
    )

    temporary.replace(path)


def select_balanced_accuracy_threshold(y_true, probabilities):
    fpr, tpr, thresholds = roc_curve(
        y_true,
        probabilities,
        drop_intermediate=False,
    )

    valid = (
        np.isfinite(thresholds)
        & (thresholds >= 0)
        & (thresholds <= 1)
    )

    fpr = fpr[valid]
    tpr = tpr[valid]
    thresholds = thresholds[valid]

    balanced_accuracies = (
        tpr + (1.0 - fpr)
    ) / 2.0

    best_value = float(
        np.max(balanced_accuracies)
    )

    best_indices = np.flatnonzero(
        np.isclose(
            balanced_accuracies,
            best_value,
            rtol=0,
            atol=1e-12,
        )
    )

    # Frozen tie-break: threshold closest to 0.5
    distances = np.abs(
        thresholds[best_indices] - 0.5
    )

    chosen_local = best_indices[
        np.argmin(distances)
    ]

    return {
        "threshold": float(
            thresholds[chosen_local]
        ),
        "calibration_balanced_accuracy": (
            best_value
        ),
        "number_of_tied_optima": int(
            len(best_indices)
        ),
    }


def calculate_metrics(y_true, probabilities, threshold):
    predictions = (
        probabilities >= threshold
    ).astype(np.int8)

    return {
        "threshold": float(threshold),
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                predictions,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "mcc": float(
            matthews_corrcoef(
                y_true,
                predictions,
            )
        ),
        "pr_auc_attack": float(
            average_precision_score(
                y_true,
                probabilities,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        ),
        "benign_recall": float(
            recall_score(
                y_true,
                predictions,
                pos_label=0,
                zero_division=0,
            )
        ),
        "attack_recall": float(
            recall_score(
                y_true,
                predictions,
                pos_label=1,
                zero_division=0,
            )
        ),
    }


def predict_probability(model, features):
    probabilities = model.predict_proba(features)
    return probabilities[:, 1]


# ----------------------------------------------------------------
# Verify protocol chain
# ----------------------------------------------------------------

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
    raise RuntimeError("Protocol hash mismatch.")

if sha256_file(schema_file) != EXPECTED_SCHEMA_HASH:
    raise RuntimeError("Feature-schema hash mismatch.")

if sha256_file(amendment_file) != EXPECTED_AMENDMENT_HASH:
    raise RuntimeError("Amendment A1 hash mismatch.")

with open(schema_file, "r", encoding="utf-8") as file:
    FEATURES = json.load(file)["predictors_in_order"]

software_versions = {
    "python_note": "Google Colab runtime",
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "scikit_learn": sklearn.__version__,
    "xgboost": xgboost.__version__,
    "joblib": joblib.__version__,
}

print("=" * 78)
print("PAPER 3 — SOURCE-ONLY MODEL TRAINING")
print("=" * 78)
print("Protocol chain verification: PASS")
print(f"Predictors: {len(FEATURES)}")
print(f"Software versions: {software_versions}")


all_result_rows = []
all_training_problems = []


for dataset in DATASETS:

    print(f"\n{'-' * 78}")
    print(f"DATASET: {dataset}")
    print(f"{'-' * 78}")

    for seed in SEEDS:

        seed_start = time.time()

        input_dir = (
            SANITIZED_DIR
            / dataset
            / f"seed_{seed}"
        )

        output_dir = (
            MODEL_DIR
            / dataset
            / f"seed_{seed}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        completion_file = (
            output_dir / "source_seed_complete.json"
        )

        if completion_file.exists():
            with open(
                completion_file,
                "r",
                encoding="utf-8",
            ) as file:
                completed = json.load(file)

            for model_name, model_result in completed[
                "models"
            ].items():
                all_result_rows.append({
                    "dataset": dataset,
                    "seed": seed,
                    "model": model_name,
                    **model_result["source_test_metrics"],
                })

            print(f"  Seed {seed}: SKIPPED—already completed")
            continue

        sanitized_manifest_file = (
            input_dir / "sanitization_manifest.json"
        )

        with open(
            sanitized_manifest_file,
            "r",
            encoding="utf-8",
        ) as file:
            sanitized_manifest = json.load(file)

        if sanitized_manifest[
            "maximum_final_overlap"
        ] != 0:
            raise RuntimeError(
                f"{dataset}, seed {seed}: overlap is not zero."
            )

        required_files = {
            name: input_dir / f"{name}.parquet"
            for name in [
                "train",
                "calibration",
                "source_test",
            ]
        }

        for name, path in required_files.items():
            expected_hash = sanitized_manifest[
                "output_files"
            ][name]["sha256"]

            if sha256_file(path) != expected_hash:
                raise RuntimeError(
                    f"{dataset}, seed {seed}: "
                    f"{name} hash mismatch."
                )

        train = pd.read_parquet(
            required_files["train"]
        )

        calibration = pd.read_parquet(
            required_files["calibration"]
        )

        source_test = pd.read_parquet(
            required_files["source_test"]
        )

        X_train_raw = train[
            FEATURES
        ].to_numpy(dtype=np.float32)

        y_train = train[
            "Label"
        ].to_numpy(dtype=np.int8)

        X_calibration_raw = calibration[
            FEATURES
        ].to_numpy(dtype=np.float32)

        y_calibration = calibration[
            "Label"
        ].to_numpy(dtype=np.int8)

        X_source_test_raw = source_test[
            FEATURES
        ].to_numpy(dtype=np.float32)

        y_source_test = source_test[
            "Label"
        ].to_numpy(dtype=np.int8)

        # Fit only on source training data
        imputer = SimpleImputer(
            strategy="median",
        )

        X_train = imputer.fit_transform(
            X_train_raw
        ).astype(np.float32)

        X_calibration = imputer.transform(
            X_calibration_raw
        ).astype(np.float32)

        X_source_test = imputer.transform(
            X_source_test_raw
        ).astype(np.float32)

        if (
            np.isnan(X_train).any()
            or np.isnan(X_calibration).any()
            or np.isnan(X_source_test).any()
        ):
            raise RuntimeError(
                f"{dataset}, seed {seed}: "
                "missing values remain after imputation."
            )

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(
            X_train
        ).astype(np.float32)

        X_calibration_scaled = scaler.transform(
            X_calibration
        ).astype(np.float32)

        X_source_test_scaled = scaler.transform(
            X_source_test
        ).astype(np.float32)

        preprocessor_artifact = {
            "dataset": dataset,
            "seed": seed,
            "features": FEATURES,
            "imputer": imputer,
            "logistic_regression_scaler": scaler,
            "protocol_sha256": EXPECTED_PROTOCOL_HASH,
            "amendment_A1_sha256": (
                EXPECTED_AMENDMENT_HASH
            ),
            "feature_schema_sha256": (
                EXPECTED_SCHEMA_HASH
            ),
        }

        preprocessor_file = (
            output_dir / "source_preprocessor.joblib"
        )

        atomic_joblib(
            preprocessor_artifact,
            preprocessor_file,
        )

        model_definitions = {
            "LogisticRegression": {
                "model": LogisticRegression(
                    solver="liblinear",
                    max_iter=2000,
                    class_weight=None,
                    random_state=seed,
                ),
                "scaled": True,
            },

            "RandomForest": {
                "model": RandomForestClassifier(
                    n_estimators=300,
                    max_features="sqrt",
                    min_samples_leaf=2,
                    class_weight=None,
                    random_state=seed,
                    n_jobs=-1,
                ),
                "scaled": False,
            },

            "XGBoost": {
                "model": XGBClassifier(
                    n_estimators=600,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    random_state=seed,
                    n_jobs=-1,
                    verbosity=0,
                ),
                "scaled": False,
            },
        }

        seed_model_results = {}

        for model_name, definition in model_definitions.items():

            model_start = time.time()
            model = definition["model"]
            uses_scaling = definition["scaled"]

            if uses_scaling:
                fit_features = X_train_scaled
                calibration_features = (
                    X_calibration_scaled
                )
                source_test_features = (
                    X_source_test_scaled
                )
            else:
                fit_features = X_train
                calibration_features = X_calibration
                source_test_features = X_source_test

            with warnings.catch_warnings(
                record=True
            ) as captured_warnings:
                warnings.simplefilter("always")

                model.fit(
                    fit_features,
                    y_train,
                )

            warning_messages = [
                str(item.message)
                for item in captured_warnings
            ]

            convergence_warnings = [
                item for item in captured_warnings
                if issubclass(
                    item.category,
                    ConvergenceWarning,
                )
            ]

            calibration_probabilities = (
                predict_probability(
                    model,
                    calibration_features,
                )
            )

            threshold_result = (
                select_balanced_accuracy_threshold(
                    y_calibration,
                    calibration_probabilities,
                )
            )

            threshold = threshold_result[
                "threshold"
            ]

            source_test_probabilities = (
                predict_probability(
                    model,
                    source_test_features,
                )
            )

            calibration_metrics = calculate_metrics(
                y_calibration,
                calibration_probabilities,
                threshold,
            )

            source_test_metrics = calculate_metrics(
                y_source_test,
                source_test_probabilities,
                threshold,
            )

            calibration_predictions = pd.DataFrame({
                "row_id": calibration[
                    "row_id"
                ].to_numpy(dtype=np.int64),
                "y_true": y_calibration,
                "probability_attack": (
                    calibration_probabilities
                ),
                "y_pred": (
                    calibration_probabilities
                    >= threshold
                ).astype(np.int8),
            })

            source_test_predictions = pd.DataFrame({
                "row_id": source_test[
                    "row_id"
                ].to_numpy(dtype=np.int64),
                "y_true": y_source_test,
                "probability_attack": (
                    source_test_probabilities
                ),
                "y_pred": (
                    source_test_probabilities
                    >= threshold
                ).astype(np.int8),
            })

            model_file = (
                output_dir
                / f"{model_name}_model.joblib"
            )

            calibration_prediction_file = (
                output_dir
                / f"{model_name}_calibration_predictions.parquet"
            )

            source_test_prediction_file = (
                output_dir
                / f"{model_name}_source_test_predictions.parquet"
            )

            model_artifact = {
                "dataset": dataset,
                "seed": seed,
                "model_name": model_name,
                "model": model,
                "uses_standard_scaler": uses_scaling,
                "threshold": threshold,
                "features": FEATURES,
                "protocol_sha256": (
                    EXPECTED_PROTOCOL_HASH
                ),
                "amendment_A1_sha256": (
                    EXPECTED_AMENDMENT_HASH
                ),
                "feature_schema_sha256": (
                    EXPECTED_SCHEMA_HASH
                ),
            }

            atomic_joblib(
                model_artifact,
                model_file,
            )

            atomic_parquet(
                calibration_predictions,
                calibration_prediction_file,
            )

            atomic_parquet(
                source_test_predictions,
                source_test_prediction_file,
            )

            model_elapsed = (
                time.time() - model_start
            )

            model_gate_pass = (
                len(convergence_warnings) == 0
                and np.isfinite(
                    calibration_probabilities
                ).all()
                and np.isfinite(
                    source_test_probabilities
                ).all()
                and 0 <= threshold <= 1
            )

            if not model_gate_pass:
                all_training_problems.append(
                    f"{dataset}, seed {seed}, "
                    f"{model_name}: model gate failed"
                )

            model_result = {
                "model_name": model_name,
                "uses_standard_scaler": (
                    uses_scaling
                ),
                "threshold_selection": (
                    threshold_result
                ),
                "calibration_metrics": (
                    calibration_metrics
                ),
                "source_test_metrics": (
                    source_test_metrics
                ),
                "fit_warning_count": int(
                    len(warning_messages)
                ),
                "fit_warnings": warning_messages,
                "convergence_warning_count": int(
                    len(convergence_warnings)
                ),
                "elapsed_seconds": round(
                    model_elapsed,
                    3,
                ),
                "model_gate_pass": (
                    model_gate_pass
                ),
                "files": {
                    "model": {
                        "path": str(model_file),
                        "sha256": sha256_file(
                            model_file
                        ),
                    },
                    "calibration_predictions": {
                        "path": str(
                            calibration_prediction_file
                        ),
                        "sha256": sha256_file(
                            calibration_prediction_file
                        ),
                    },
                    "source_test_predictions": {
                        "path": str(
                            source_test_prediction_file
                        ),
                        "sha256": sha256_file(
                            source_test_prediction_file
                        ),
                    },
                },
            }

            seed_model_results[
                model_name
            ] = model_result

            all_result_rows.append({
                "dataset": dataset,
                "seed": seed,
                "model": model_name,
                **source_test_metrics,
            })

            print(
                f"  Seed {seed} | {model_name}: "
                f"threshold={threshold:.6f}, "
                f"source BA="
                f"{source_test_metrics['balanced_accuracy']:.4f}, "
                f"Macro F1="
                f"{source_test_metrics['macro_f1']:.4f}, "
                f"MCC={source_test_metrics['mcc']:.4f}, "
                f"warnings={len(warning_messages)}, "
                f"gate="
                f"{'PASS' if model_gate_pass else 'FAIL'}"
            )

        seed_completion = {
            "dataset": dataset,
            "seed": seed,
            "created_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "protocol_sha256": (
                EXPECTED_PROTOCOL_HASH
            ),
            "amendment_A1_sha256": (
                EXPECTED_AMENDMENT_HASH
            ),
            "feature_schema_sha256": (
                EXPECTED_SCHEMA_HASH
            ),
            "input_sanitization_manifest_sha256": (
                sha256_file(
                    sanitized_manifest_file
                )
            ),
            "preprocessor": {
                "path": str(preprocessor_file),
                "sha256": sha256_file(
                    preprocessor_file
                ),
            },
            "software_versions": software_versions,
            "models": seed_model_results,
            "elapsed_seconds": round(
                time.time() - seed_start,
                3,
            ),
            "status": (
                "PASS"
                if all(
                    item["model_gate_pass"]
                    for item in seed_model_results.values()
                )
                else "FAIL"
            ),
        }

        completion_file.write_text(
            json.dumps(
                seed_completion,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        print(
            f"  Seed {seed} completed in "
            f"{(time.time() - seed_start) / 60:.1f} min"
        )


# ----------------------------------------------------------------
# Source-results summary and cryptographic seal
# ----------------------------------------------------------------

results_table = pd.DataFrame(all_result_rows)

results_file = (
    MODEL_DIR / "source_test_metrics_all_models.csv"
)

results_table.to_csv(
    results_file,
    index=False,
)

seal_entries = []

for dataset in DATASETS:
    for seed in SEEDS:
        completion_file = (
            MODEL_DIR
            / dataset
            / f"seed_{seed}"
            / "source_seed_complete.json"
        )

        if not completion_file.exists():
            raise RuntimeError(
                f"Missing completion file: "
                f"{dataset}, seed {seed}"
            )

        with open(
            completion_file,
            "r",
            encoding="utf-8",
        ) as file:
            completion = json.load(file)

        if completion["status"] != "PASS":
            all_training_problems.append(
                f"{dataset}, seed {seed}: "
                "completion status failed"
            )

        seal_entries.append({
            "relative_path": str(
                completion_file.relative_to(
                    MODEL_DIR
                )
            ),
            "sha256": sha256_file(
                completion_file
            ),
        })

        preprocessor_path = Path(
            completion["preprocessor"]["path"]
        )

        seal_entries.append({
            "relative_path": str(
                preprocessor_path.relative_to(
                    MODEL_DIR
                )
            ),
            "sha256": sha256_file(
                preprocessor_path
            ),
        })

        for model_result in completion[
            "models"
        ].values():
            for artifact in model_result[
                "files"
            ].values():
                artifact_path = Path(
                    artifact["path"]
                )

                seal_entries.append({
                    "relative_path": str(
                        artifact_path.relative_to(
                            MODEL_DIR
                        )
                    ),
                    "sha256": sha256_file(
                        artifact_path
                    ),
                })

seal_entries.append({
    "relative_path": results_file.name,
    "sha256": sha256_file(results_file),
})

seal_manifest = {
    "seal_id": "P3_SOURCE_MODELS_SEAL_V1",
    "created_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "protocol_sha256": EXPECTED_PROTOCOL_HASH,
    "amendment_A1_sha256": EXPECTED_AMENDMENT_HASH,
    "feature_schema_sha256": EXPECTED_SCHEMA_HASH,
    "source_models": 60,
    "target_datasets_evaluated": 0,
    "target_results_seen": False,
    "artifacts": sorted(
        seal_entries,
        key=lambda item: item["relative_path"],
    ),
    "training_problems": all_training_problems,
    "status": (
        "PASS"
        if not all_training_problems
        else "FAIL"
    ),
}

seal_file = (
    MODEL_DIR / "source_model_seal_manifest.json"
)

seal_file.write_text(
    json.dumps(
        seal_manifest,
        indent=2,
        sort_keys=True,
    ),
    encoding="utf-8",
)

seal_hash = sha256_file(seal_file)

(
    MODEL_DIR
    / "source_model_seal_manifest.sha256"
).write_text(
    seal_hash,
    encoding="utf-8",
)

print("\n" + "=" * 78)
print(
    "SOURCE-ONLY MODEL TRAINING AND SEAL: "
    f"{seal_manifest['status']}"
)
print(f"Models trained: {len(results_table)}")
print(f"Problems: {all_training_problems}")
print(f"Source-model seal SHA-256: {seal_hash}")
print(f"Source metrics: {results_file}")
print("=" * 78)

if seal_manifest["status"] != "PASS":
    raise RuntimeError(
        "Source-model seal failed. Do not continue."
    )