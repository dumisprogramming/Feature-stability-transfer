from pathlib import Path
from datetime import datetime, timezone
from itertools import combinations
import hashlib
import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(
    "/content/drivesyz/MyDrive/research/researchdata/"
    "NIDS_Research/Paper3_DataSet"
)

# Correct the path if the accidental "drivesyz" text is present.
ROOT = Path(
    "/content/drive/MyDrive/research/researchdata/"
    "NIDS_Research/Paper3_DataSet"
)

PROTOCOL_DIR = ROOT / "Paper3_Protocol"
MATERIALIZED_DIR = ROOT / "Paper3_Materialized"
SANITIZED_DIR = ROOT / "Paper3_Sanitized"
MODEL_DIR = ROOT / "Paper3_Source_Models"
STABILITY_DIR = ROOT / "Paper3_Source_Stability"
STABILITY_DIR.mkdir(parents=True, exist_ok=True)

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

EXPECTED_SOURCE_MODEL_SEAL = (
    "81eebf7b013fec0ca1802ebb54894fe73"
    "d7023b2c6450da09c72af3b3f5a3182"
)

DATASETS = [
    "NF_BoT_IoT_v3",
    "NF_CSE_CIC_IDS2018_v3",
    "NF_ToN_IoT_v3",
    "NF_UNSW_NB15_v3",
]

SEEDS = [2027, 2028, 2029, 2030, 2031]

MODELS = [
    "LogisticRegression",
    "XGBoost",
    "RandomForest",
]

N_REPEATS = 3
TOP_K = 10


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


def stable_random_state(*parts):
    text = "|".join(str(part) for part in parts)

    digest = hashlib.sha256(
        text.encode("utf-8")
    ).digest()

    return int.from_bytes(
        digest[:4],
        byteorder="little",
        signed=False,
    )


def threshold_scorer(threshold):
    def scorer(estimator, X, y):
        probabilities = estimator.predict_proba(X)[:, 1]

        predictions = (
            probabilities >= threshold
        ).astype(np.int8)

        return balanced_accuracy_score(
            y,
            predictions,
        )

    return scorer


def calculate_pairwise_stability(
    importance_frame,
    feature_names,
    group_type,
):
    subset = importance_frame[
        importance_frame["group_type"] == group_type
    ].copy()

    pivot = subset.pivot(
        index="group_id",
        columns="feature",
        values="importance_mean",
    )

    pivot = pivot.reindex(
        columns=feature_names
    )

    if pivot.isna().any().any():
        raise RuntimeError(
            f"Missing importance values for {group_type}"
        )

    group_ids = pivot.index.tolist()
    pairwise_rows = []

    for first, second in combinations(group_ids, 2):
        first_values = pivot.loc[
            first
        ].to_numpy(dtype=np.float64)

        second_values = pivot.loc[
            second
        ].to_numpy(dtype=np.float64)

        # Average ranks preserve ties.
        first_ranks = (
            pd.Series(-first_values)
            .rank(method="average")
            .to_numpy(dtype=np.float64)
        )

        second_ranks = (
            pd.Series(-second_values)
            .rank(method="average")
            .to_numpy(dtype=np.float64)
        )

        first_std = float(
            np.std(first_ranks)
        )

        second_std = float(
            np.std(second_ranks)
        )

        undefined_spearman = (
            first_std == 0
            or second_std == 0
        )

        if undefined_spearman:
            # No ranking information is treated as zero stability.
            spearman = 0.0
        else:
            spearman = float(
                np.corrcoef(
                    first_ranks,
                    second_ranks,
                )[0, 1]
            )

        first_order = sorted(
            range(len(feature_names)),
            key=lambda index: (
                -first_values[index],
                feature_names[index],
            ),
        )[:TOP_K]

        second_order = sorted(
            range(len(feature_names)),
            key=lambda index: (
                -second_values[index],
                feature_names[index],
            ),
        )[:TOP_K]

        first_top = {
            feature_names[index]
            for index in first_order
        }

        second_top = {
            feature_names[index]
            for index in second_order
        }

        top_k_jaccard = (
            len(first_top & second_top)
            / len(first_top | second_top)
        )

        pairwise_rows.append({
            "group_type": group_type,
            "group_1": str(first),
            "group_2": str(second),
            "spearman": spearman,
            "top10_jaccard": float(
                top_k_jaccard
            ),
            "undefined_spearman": bool(
                undefined_spearman
            ),
        })

    pairwise_frame = pd.DataFrame(
        pairwise_rows
    )

    if pairwise_frame.empty:
        raise RuntimeError(
            f"Fewer than two groups for {group_type}"
        )

    summary = {
        "groups": int(len(group_ids)),
        "pairs": int(len(pairwise_frame)),
        "mean_spearman": float(
            pairwise_frame["spearman"].mean()
        ),
        "median_spearman": float(
            pairwise_frame["spearman"].median()
        ),
        "mean_top10_jaccard": float(
            pairwise_frame[
                "top10_jaccard"
            ].mean()
        ),
        "undefined_spearman_pairs": int(
            pairwise_frame[
                "undefined_spearman"
            ].sum()
        ),
    }

    return pairwise_frame, summary


# ----------------------------------------------------------------
# Verify the entire source-only artifact chain
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

source_model_seal_file = (
    MODEL_DIR / "source_model_seal_manifest.json"
)

checks = {
    "protocol": (
        sha256_file(protocol_file)
        == EXPECTED_PROTOCOL_HASH
    ),
    "schema": (
        sha256_file(schema_file)
        == EXPECTED_SCHEMA_HASH
    ),
    "amendment_A1": (
        sha256_file(amendment_file)
        == EXPECTED_AMENDMENT_HASH
    ),
    "source_model_seal": (
        sha256_file(source_model_seal_file)
        == EXPECTED_SOURCE_MODEL_SEAL
    ),
}

if not all(checks.values()):
    raise RuntimeError(
        f"Artifact-chain verification failed: {checks}"
    )

with open(schema_file, "r", encoding="utf-8") as file:
    FEATURES = json.load(file)[
        "predictors_in_order"
    ]

print("=" * 78)
print("PAPER 3 — SOURCE-ONLY FEATURE-STABILITY ANALYSIS")
print("=" * 78)
print(f"Artifact-chain verification: {checks}")
print(f"Predictors: {len(FEATURES)}")
print(f"Permutation repeats: {N_REPEATS}")


warning_rows = []
stability_problems = []


for dataset in DATASETS:

    print(f"\n{'-' * 78}")
    print(f"DATASET: {dataset}")
    print(f"{'-' * 78}")

    for seed in SEEDS:

        seed_start = time.time()

        environment_file = (
            SANITIZED_DIR
            / dataset
            / f"seed_{seed}"
            / "environment.parquet"
        )

        environment_data = pd.read_parquet(
            environment_file
        )

        expected_features = set(FEATURES)

        if not expected_features.issubset(
            environment_data.columns
        ):
            raise RuntimeError(
                f"{dataset}, seed {seed}: "
                "missing environment predictors."
            )

        actual_counts = (
            environment_data.groupby([
                "actual_environment_id",
                "Label",
            ]).size()
        )

        pseudo_counts = (
            environment_data.groupby([
                "pseudo_environment_id",
                "Label",
            ]).size()
        )

        if not (actual_counts == 2000).all():
            raise RuntimeError(
                "Actual environment groups are not balanced."
            )

        if not (pseudo_counts == 2000).all():
            raise RuntimeError(
                "Pseudo-environment groups are not balanced."
            )

        preprocessor_file = (
            MODEL_DIR
            / dataset
            / f"seed_{seed}"
            / "source_preprocessor.joblib"
        )

        preprocessor_artifact = joblib.load(
            preprocessor_file
        )

        if (
            preprocessor_artifact[
                "protocol_sha256"
            ] != EXPECTED_PROTOCOL_HASH
        ):
            raise RuntimeError(
                "Preprocessor protocol mismatch."
            )

        X_environment_raw = environment_data[
            FEATURES
        ].to_numpy(dtype=np.float32)

        y_environment = environment_data[
            "Label"
        ].to_numpy(dtype=np.int8)

        X_environment = (
            preprocessor_artifact["imputer"]
            .transform(X_environment_raw)
            .astype(np.float32)
        )

        X_environment_scaled = (
            preprocessor_artifact[
                "logistic_regression_scaler"
            ]
            .transform(X_environment)
            .astype(np.float32)
        )

        group_definitions = []

        for group_id in sorted(
            environment_data[
                "actual_environment_id"
            ].unique()
        ):
            positions = np.flatnonzero(
                environment_data[
                    "actual_environment_id"
                ].to_numpy()
                == group_id
            )

            group_definitions.append({
                "group_type": "environmental",
                "group_id": str(group_id),
                "positions": positions,
            })

        for group_id in sorted(
            environment_data[
                "pseudo_environment_id"
            ].unique()
        ):
            positions = np.flatnonzero(
                environment_data[
                    "pseudo_environment_id"
                ].to_numpy()
                == group_id
            )

            group_definitions.append({
                "group_type": "sampling",
                "group_id": str(group_id),
                "positions": positions,
            })

        for model_name in MODELS:

            model_output_dir = (
                STABILITY_DIR
                / dataset
                / f"seed_{seed}"
                / model_name
            )

            model_output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            completion_file = (
                model_output_dir
                / "stability_complete.json"
            )

            pfi_file = (
                model_output_dir
                / "permutation_importance.parquet"
            )

            pairwise_file = (
                model_output_dir
                / "pairwise_stability.parquet"
            )

            if completion_file.exists():
                with open(
                    completion_file,
                    "r",
                    encoding="utf-8",
                ) as file:
                    completed = json.load(file)

                if sha256_file(
                    pfi_file
                ) != completed[
                    "permutation_importance_sha256"
                ]:
                    raise RuntimeError(
                        "Completed PFI hash mismatch."
                    )

                if sha256_file(
                    pairwise_file
                ) != completed[
                    "pairwise_stability_sha256"
                ]:
                    raise RuntimeError(
                        "Completed pairwise hash mismatch."
                    )

                warning_rows.append(
                    completed["warning_scores"]
                )

                print(
                    f"  Seed {seed} | {model_name}: "
                    "SKIPPED—already completed"
                )
                continue

            source_model_completion_file = (
                MODEL_DIR
                / dataset
                / f"seed_{seed}"
                / "source_seed_complete.json"
            )

            with open(
                source_model_completion_file,
                "r",
                encoding="utf-8",
            ) as file:
                source_completion = json.load(file)

            model_record = source_completion[
                "models"
            ][model_name]

            model_file = Path(
                model_record[
                    "files"
                ]["model"]["path"]
            )

            if sha256_file(model_file) != model_record[
                "files"
            ]["model"]["sha256"]:
                raise RuntimeError(
                    "Source model artifact hash mismatch."
                )

            model_artifact = joblib.load(
                model_file
            )

            model = model_artifact["model"]
            threshold = float(
                model_artifact["threshold"]
            )

            uses_scaling = bool(
                model_artifact[
                    "uses_standard_scaler"
                ]
            )

            if uses_scaling:
                X_for_model = X_environment_scaled
            else:
                X_for_model = X_environment

            scorer = threshold_scorer(
                threshold
            )

            if pfi_file.exists():
                pfi_results = pd.read_parquet(
                    pfi_file
                )
            else:
                pfi_results = pd.DataFrame()

            completed_groups = set()

            if not pfi_results.empty:
                counts = (
                    pfi_results.groupby([
                        "group_type",
                        "group_id",
                    ]).size()
                )

                completed_groups = {
                    (
                        str(group_type),
                        str(group_id),
                    )
                    for (
                        group_type,
                        group_id,
                    ), count in counts.items()
                    if count == len(FEATURES)
                }

                incomplete_groups = {
                    (
                        str(group_type),
                        str(group_id),
                    )
                    for (
                        group_type,
                        group_id,
                    ), count in counts.items()
                    if count != len(FEATURES)
                }

                if incomplete_groups:
                    remove_mask = pfi_results.apply(
                        lambda row: (
                            str(row["group_type"]),
                            str(row["group_id"]),
                        ) in incomplete_groups,
                        axis=1,
                    )

                    pfi_results = pfi_results[
                        ~remove_mask
                    ].reset_index(drop=True)

            groups_finished_now = 0
            total_groups = len(group_definitions)

            for group in group_definitions:

                group_key = (
                    group["group_type"],
                    str(group["group_id"]),
                )

                if group_key in completed_groups:
                    continue

                positions = group["positions"]

                X_group = X_for_model[positions]
                y_group = y_environment[positions]

                label_counts = np.bincount(
                    y_group,
                    minlength=2,
                )

                if not np.array_equal(
                    label_counts,
                    np.array([2000, 2000]),
                ):
                    raise RuntimeError(
                        f"Unbalanced PFI group: {group_key}"
                    )

                baseline_ba = float(
                    scorer(
                        model,
                        X_group,
                        y_group,
                    )
                )

                random_state = stable_random_state(
                    dataset,
                    seed,
                    model_name,
                    group["group_type"],
                    group["group_id"],
                    "PFI",
                )

                pfi = permutation_importance(
                    estimator=model,
                    X=X_group,
                    y=y_group,
                    scoring=scorer,
                    n_repeats=N_REPEATS,
                    random_state=random_state,
                    n_jobs=1,
                )

                group_rows = []

                for feature_index, feature in enumerate(
                    FEATURES
                ):
                    group_rows.append({
                        "dataset": dataset,
                        "seed": seed,
                        "model": model_name,
                        "group_type": (
                            group["group_type"]
                        ),
                        "group_id": str(
                            group["group_id"]
                        ),
                        "feature": feature,
                        "baseline_balanced_accuracy": (
                            baseline_ba
                        ),
                        "importance_mean": float(
                            pfi.importances_mean[
                                feature_index
                            ]
                        ),
                        "importance_std": float(
                            pfi.importances_std[
                                feature_index
                            ]
                        ),
                        "repeat_1": float(
                            pfi.importances[
                                feature_index,
                                0,
                            ]
                        ),
                        "repeat_2": float(
                            pfi.importances[
                                feature_index,
                                1,
                            ]
                        ),
                        "repeat_3": float(
                            pfi.importances[
                                feature_index,
                                2,
                            ]
                        ),
                        "rows": int(len(y_group)),
                        "benign_rows": 2000,
                        "attack_rows": 2000,
                        "permutation_random_state": (
                            random_state
                        ),
                    })

                group_frame = pd.DataFrame(
                    group_rows
                )

                pfi_results = pd.concat(
                    [
                        pfi_results,
                        group_frame,
                    ],
                    ignore_index=True,
                )

                atomic_parquet(
                    pfi_results,
                    pfi_file,
                )

                completed_groups.add(group_key)
                groups_finished_now += 1

                if (
                    groups_finished_now % 5 == 0
                    or len(completed_groups)
                    == total_groups
                ):
                    print(
                        f"    Seed {seed} | {model_name}: "
                        f"{len(completed_groups)}/"
                        f"{total_groups} groups completed"
                    )

            expected_result_rows = (
                total_groups * len(FEATURES)
            )

            if len(pfi_results) != expected_result_rows:
                raise RuntimeError(
                    f"Unexpected PFI row count: "
                    f"{len(pfi_results)} != "
                    f"{expected_result_rows}"
                )

            environmental_pairs, environmental_summary = (
                calculate_pairwise_stability(
                    pfi_results,
                    FEATURES,
                    "environmental",
                )
            )

            sampling_pairs, sampling_summary = (
                calculate_pairwise_stability(
                    pfi_results,
                    FEATURES,
                    "sampling",
                )
            )

            environmental_pairs[
                "dataset"
            ] = dataset

            environmental_pairs["seed"] = seed

            environmental_pairs[
                "model"
            ] = model_name

            sampling_pairs["dataset"] = dataset
            sampling_pairs["seed"] = seed
            sampling_pairs["model"] = model_name

            pairwise_results = pd.concat(
                [
                    environmental_pairs,
                    sampling_pairs,
                ],
                ignore_index=True,
            )

            atomic_parquet(
                pairwise_results,
                pairwise_file,
            )

            environmental_stability = float(
                environmental_summary[
                    "mean_spearman"
                ]
            )

            sampling_stability = float(
                sampling_summary[
                    "mean_spearman"
                ]
            )

            warning_scores = {
                "dataset": dataset,
                "seed": seed,
                "model": model_name,
                "environmental_stability": (
                    environmental_stability
                ),
                "sampling_stability": (
                    sampling_stability
                ),
                "environmental_instability": float(
                    1.0
                    - environmental_stability
                ),
                "sampling_environment_gap": float(
                    sampling_stability
                    - environmental_stability
                ),
                "environmental_top10_jaccard": float(
                    environmental_summary[
                        "mean_top10_jaccard"
                    ]
                ),
                "sampling_top10_jaccard": float(
                    sampling_summary[
                        "mean_top10_jaccard"
                    ]
                ),
                "environmental_groups": int(
                    environmental_summary["groups"]
                ),
                "sampling_groups": int(
                    sampling_summary["groups"]
                ),
                "environmental_pairs": int(
                    environmental_summary["pairs"]
                ),
                "sampling_pairs": int(
                    sampling_summary["pairs"]
                ),
                "undefined_environmental_pairs": int(
                    environmental_summary[
                        "undefined_spearman_pairs"
                    ]
                ),
                "undefined_sampling_pairs": int(
                    sampling_summary[
                        "undefined_spearman_pairs"
                    ]
                ),
                "mean_environmental_baseline_ba": float(
                    pfi_results[
                        pfi_results["group_type"]
                        == "environmental"
                    ][
                        "baseline_balanced_accuracy"
                    ].mean()
                ),
                "mean_sampling_baseline_ba": float(
                    pfi_results[
                        pfi_results["group_type"]
                        == "sampling"
                    ][
                        "baseline_balanced_accuracy"
                    ].mean()
                ),
            }

            stability_gate_pass = (
                np.isfinite(
                    list(
                        warning_scores.values()
                    )[3:]
                ).all()
                and warning_scores[
                    "environmental_groups"
                ] >= 4
                and warning_scores[
                    "sampling_groups"
                ] >= 4
            )

            if not stability_gate_pass:
                stability_problems.append(
                    f"{dataset}, seed {seed}, "
                    f"{model_name}: stability gate failed"
                )

            completion = {
                "dataset": dataset,
                "seed": seed,
                "model": model_name,
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
                "source_model_seal_sha256": (
                    EXPECTED_SOURCE_MODEL_SEAL
                ),
                "threshold": threshold,
                "permutation_repeats": N_REPEATS,
                "features": len(FEATURES),
                "environmental_summary": (
                    environmental_summary
                ),
                "sampling_summary": sampling_summary,
                "warning_scores": warning_scores,
                "permutation_importance_sha256": (
                    sha256_file(pfi_file)
                ),
                "pairwise_stability_sha256": (
                    sha256_file(pairwise_file)
                ),
                "stability_gate_pass": (
                    stability_gate_pass
                ),
                "status": (
                    "PASS"
                    if stability_gate_pass
                    else "FAIL"
                ),
            }

            completion_file.write_text(
                json.dumps(
                    completion,
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

            warning_rows.append(warning_scores)

            print(
                f"  Seed {seed} | {model_name}: "
                f"S_env={environmental_stability:.4f}, "
                f"S_sampling={sampling_stability:.4f}, "
                f"gap="
                f"{warning_scores['sampling_environment_gap']:.4f}, "
                f"env-top10="
                f"{warning_scores['environmental_top10_jaccard']:.4f}, "
                f"gate="
                f"{'PASS' if stability_gate_pass else 'FAIL'}"
            )

        print(
            f"  Seed {seed} stability elapsed: "
            f"{(time.time() - seed_start) / 60:.1f} min"
        )


# ----------------------------------------------------------------
# Seal all source-only warning scores before target evaluation
# ----------------------------------------------------------------

warning_table = pd.DataFrame(warning_rows)

warning_table = warning_table.sort_values([
    "dataset",
    "seed",
    "model",
]).reset_index(drop=True)

if len(warning_table) != 60:
    raise RuntimeError(
        f"Expected 60 warning rows, found "
        f"{len(warning_table)}."
    )

warning_file = (
    STABILITY_DIR
    / "source_only_warning_scores.csv"
)

warning_table.to_csv(
    warning_file,
    index=False,
)

seal_entries = []

for dataset in DATASETS:
    for seed in SEEDS:
        for model_name in MODELS:
            model_output_dir = (
                STABILITY_DIR
                / dataset
                / f"seed_{seed}"
                / model_name
            )

            for filename in [
                "stability_complete.json",
                "permutation_importance.parquet",
                "pairwise_stability.parquet",
            ]:
                path = model_output_dir / filename

                if not path.exists():
                    raise RuntimeError(
                        f"Missing stability artifact: {path}"
                    )

                seal_entries.append({
                    "relative_path": str(
                        path.relative_to(
                            STABILITY_DIR
                        )
                    ),
                    "sha256": sha256_file(path),
                })

seal_entries.append({
    "relative_path": warning_file.name,
    "sha256": sha256_file(warning_file),
})

stability_seal = {
    "seal_id": "P3_SOURCE_ONLY_STABILITY_SEAL_V1",
    "created_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "protocol_sha256": EXPECTED_PROTOCOL_HASH,
    "amendment_A1_sha256": EXPECTED_AMENDMENT_HASH,
    "feature_schema_sha256": EXPECTED_SCHEMA_HASH,
    "source_model_seal_sha256": (
        EXPECTED_SOURCE_MODEL_SEAL
    ),
    "warning_score_rows": int(
        len(warning_table)
    ),
    "target_datasets_evaluated": 0,
    "target_results_seen": False,
    "stability_problems": stability_problems,
    "artifacts": sorted(
        seal_entries,
        key=lambda item: item["relative_path"],
    ),
    "status": (
        "PASS"
        if not stability_problems
        else "FAIL"
    ),
}

stability_seal_file = (
    STABILITY_DIR
    / "source_only_stability_seal_manifest.json"
)

stability_seal_file.write_text(
    json.dumps(
        stability_seal,
        indent=2,
        sort_keys=True,
    ),
    encoding="utf-8",
)

stability_seal_hash = sha256_file(
    stability_seal_file
)

(
    STABILITY_DIR
    / "source_only_stability_seal_manifest.sha256"
).write_text(
    stability_seal_hash,
    encoding="utf-8",
)

print("\n" + "=" * 78)
print(
    "SOURCE-ONLY FEATURE-STABILITY SEAL: "
    f"{stability_seal['status']}"
)
print(f"Warning-score rows: {len(warning_table)}")
print(f"Problems: {stability_problems}")
print(
    "Source-only stability seal SHA-256: "
    f"{stability_seal_hash}"
)
print(f"Warning scores: {warning_file}")
print("=" * 78)

if stability_seal["status"] != "PASS":
    raise RuntimeError(
        "Source-only stability seal failed. "
        "Do not evaluate targets."
    )