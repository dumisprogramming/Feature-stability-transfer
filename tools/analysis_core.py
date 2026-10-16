"""Analysis functions copied verbatim from the supplied final-evaluation script.

Original SHA-256: ef9b7b01a25a1efa72160bbf4a03f264a391b20fa52b9e87fc9d59a4d028734e
Only the seven named function definitions are copied, avoiding Colab mounting,
label access and top-level writes in the historical script. This module replays
statistics from supplied result tables; it does not recreate source predictions.
"""
from itertools import permutations, product
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
DATASETS = ["NF_BoT_IoT_v3", "NF_CSE_CIC_IDS2018_v3", "NF_ToN_IoT_v3", "NF_UNSW_NB15_v3"]
MODELS = ["LogisticRegression", "RandomForest", "XGBoost"]
BOOTSTRAP_SEED = 53027
BOOTSTRAP_REPLICATES = 10000
ALPHA = 0.05


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
