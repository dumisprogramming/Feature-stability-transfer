#!/usr/bin/env python3
"""Build publication figures and manuscript tables for Paper 3.

The program is deliberately read-only with respect to the sealed result directory.
It verifies the recorded hashes, reconstructs the prespecified 12 confirmatory
units, checks the reported statistics, and writes a separate manuscript-assets
directory. It runs both in Google Colab and in a local Python environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_VERSION = "1.0.0"
DEFAULT_DRIVE_INPUT = Path(
    "/content/drive/MyDrive/research/researchdata/NIDS_Research/"
    "Paper3_DataSet/Paper3_Final_Evaluation"
)
DEFAULT_DRIVE_OUTPUT = Path(
    "/content/drive/MyDrive/research/researchdata/NIDS_Research/"
    "Paper3_DataSet/Paper3_Manuscript_Assets"
)

REQUIRED_FILES = (
    "transfer_analysis_all_180.csv",
    "direction_model_summary_36.csv",
    "loso_model_scores.csv",
    "confirmatory_association_results.csv",
    "final_analysis_summary.json",
    "final_results_seal.json",
)

DATASET_ORDER = [
    "NF_BoT_IoT_v3",
    "NF_CSE_CIC_IDS2018_v3",
    "NF_ToN_IoT_v3",
    "NF_UNSW_NB15_v3",
]
DATASET_LABEL = {
    "NF_BoT_IoT_v3": "BoT-IoT",
    "NF_CSE_CIC_IDS2018_v3": "CSE-CIC-IDS2018",
    "NF_ToN_IoT_v3": "ToN-IoT",
    "NF_UNSW_NB15_v3": "UNSW-NB15",
}
MODEL_ORDER = ["LogisticRegression", "RandomForest", "XGBoost"]
MODEL_LABEL = {
    "LogisticRegression": "Logistic regression",
    "RandomForest": "Random forest",
    "XGBoost": "XGBoost",
}
MODEL_SHORT = {
    "LogisticRegression": "LR",
    "RandomForest": "RF",
    "XGBoost": "XGB",
}
MODEL_COLOR = {
    "LogisticRegression": "#0072B2",
    "RandomForest": "#D55E00",
    "XGBoost": "#009E73",
}
MODEL_MARKER = {
    "LogisticRegression": "o",
    "RandomForest": "s",
    "XGBoost": "^",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(payload: dict) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.input_dir:
        input_dir = Path(args.input_dir).expanduser().resolve()
    elif DEFAULT_DRIVE_INPUT.exists():
        input_dir = DEFAULT_DRIVE_INPUT
    else:
        input_dir = (Path.cwd() / "upload").resolve()

    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    elif input_dir == DEFAULT_DRIVE_INPUT:
        output_dir = DEFAULT_DRIVE_OUTPUT
    else:
        output_dir = (Path.cwd() / "outputs" / "paper3_publication_assets").resolve()
    return input_dir, output_dir


def verify_inputs(input_dir: Path) -> dict:
    missing = [name for name in REQUIRED_FILES if not (input_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required result files: {missing}")

    seal = json.loads((input_dir / "final_results_seal.json").read_text("utf-8"))
    if seal.get("status") != "FINAL_TARGET_EVALUATION_SEALED":
        raise AssertionError("Final result status is not sealed")

    expected_self_hash = seal.get("seal_sha256")
    # The originating evaluation program hashes the payload before adding both
    # the human-readable status and the seal_sha256 fields.
    seal_without_hash = {
        key: value for key, value in seal.items() if key not in {"seal_sha256", "status"}
    }
    observed_self_hash = canonical_json_sha256(seal_without_hash)
    if observed_self_hash != expected_self_hash:
        raise AssertionError(
            "Final seal self-hash mismatch: "
            f"expected {expected_self_hash}, observed {observed_self_hash}"
        )

    filename_to_artifact = {
        "transfer_analysis_all_180.csv": "transfer_analysis_all_180",
        "direction_model_summary_36.csv": "direction_model_summary_36",
        "loso_model_scores.csv": "loso_model_scores",
        "confirmatory_association_results.csv": "confirmatory_association_results",
        "final_analysis_summary.json": "final_analysis_summary",
    }
    verified = {}
    for filename, artifact_name in filename_to_artifact.items():
        expected = seal["artifacts"][artifact_name]["sha256"]
        observed = sha256_file(input_dir / filename)
        if observed != expected:
            raise AssertionError(
                f"Hash mismatch for {filename}: expected {expected}, observed {observed}"
            )
        verified[filename] = observed
    return {"seal": seal, "verified_hashes": verified}


def load_and_reconcile(input_dir: Path) -> dict:
    transfer = pd.read_csv(input_dir / "transfer_analysis_all_180.csv")
    direction_model = pd.read_csv(input_dir / "direction_model_summary_36.csv")
    loso = pd.read_csv(input_dir / "loso_model_scores.csv")
    associations = pd.read_csv(input_dir / "confirmatory_association_results.csv")
    summary = json.loads((input_dir / "final_analysis_summary.json").read_text("utf-8"))

    if len(transfer) != 180 or len(direction_model) != 36:
        raise AssertionError("Unexpected transfer or direction-model row count")
    if len(loso) != 6 or len(associations) != 3:
        raise AssertionError("Unexpected LOSO or association row count")
    if set(transfer["source_dataset"]) != set(DATASET_ORDER):
        raise AssertionError("Unexpected source dataset set")
    if set(transfer["target_dataset"]) != set(DATASET_ORDER):
        raise AssertionError("Unexpected target dataset set")
    if set(transfer["model"]) != set(MODEL_ORDER):
        raise AssertionError("Unexpected model set")
    if (transfer["source_dataset"] == transfer["target_dataset"]).any():
        raise AssertionError("Self-transfer rows were found")

    reported = summary["global_descriptive_summary"]
    checks = {
        "target_balanced_accuracy": transfer["target_balanced_accuracy"].mean(),
        "target_macro_f1": transfer["target_macro_f1"].mean(),
        "target_mcc": transfer["target_mcc"].mean(),
        "target_pr_auc_attack": transfer["target_pr_auc_attack"].mean(),
        "target_benign_recall": transfer["target_benign_recall"].mean(),
        "target_attack_recall": transfer["target_attack_recall"].mean(),
        "balanced_accuracy_loss": transfer["balanced_accuracy_loss"].mean(),
    }
    for metric, observed in checks.items():
        expected = reported[metric]["mean"]
        if not math.isclose(float(observed), float(expected), abs_tol=1e-12):
            raise AssertionError(f"Summary mismatch for {metric}")

    confirmatory_units = (
        transfer.groupby(["source_dataset", "model"], as_index=False)
        .agg(
            environmental_instability=("environmental_instability", "mean"),
            sampling_environment_gap=("sampling_environment_gap", "mean"),
            sampling_instability=("sampling_instability", "mean"),
            balanced_accuracy_loss=("balanced_accuracy_loss", "mean"),
        )
        .sort_values(["source_dataset", "model"])
        .reset_index(drop=True)
    )
    for predictor in (
        "environmental_instability",
        "sampling_environment_gap",
        "sampling_instability",
    ):
        observed = confirmatory_units[predictor].corr(
            confirmatory_units["balanced_accuracy_loss"], method="spearman"
        )
        expected = associations.loc[
            associations["predictor"] == predictor, "spearman_rho"
        ].iloc[0]
        if not math.isclose(float(observed), float(expected), abs_tol=1e-12):
            raise AssertionError(f"Reconstructed Spearman mismatch for {predictor}")

    return {
        "transfer": transfer,
        "direction_model": direction_model,
        "loso": loso,
        "associations": associations,
        "summary": summary,
        "confirmatory_units": confirmatory_units,
    }


def publication_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "figure.titlesize": 11.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.55,
            "grid.alpha": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
        }
    )


def save_figure(fig: mpl.figure.Figure, stem: Path) -> list[Path]:
    outputs = []
    for suffix, kwargs in (
        (".png", {"dpi": 600}),
        (".pdf", {}),
        (".svg", {}),
    ):
        path = stem.with_suffix(suffix)
        fig.savefig(path, facecolor="white", **kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs


def figure_1_heatmaps(direction_model: pd.DataFrame, figure_dir: Path) -> list[Path]:
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.9), constrained_layout=True)
    cmap = mpl.colormaps["cividis"].copy()
    cmap.set_bad("#E6E6E6")
    vmin, vmax = 0.10, 0.60

    for ax, model in zip(axes, MODEL_ORDER):
        subset = direction_model[direction_model["model"] == model]
        matrix = np.full((4, 4), np.nan)
        for _, row in subset.iterrows():
            i = DATASET_ORDER.index(row["source_dataset"])
            j = DATASET_ORDER.index(row["target_dataset"])
            matrix[i, j] = row["balanced_accuracy_loss_mean"]
        image = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_title(MODEL_LABEL[model])
        ax.set_xticks(range(4), [DATASET_LABEL[x] for x in DATASET_ORDER], rotation=38, ha="right")
        ax.set_yticks(range(4), [DATASET_LABEL[x] for x in DATASET_ORDER])
        ax.set_xlabel("Target dataset")
        if ax is axes[0]:
            ax.set_ylabel("Source dataset")
        for i in range(4):
            for j in range(4):
                value = matrix[i, j]
                if np.isnan(value):
                    ax.text(j, i, "—", ha="center", va="center", color="#6B6B6B")
                else:
                    norm = (value - vmin) / (vmax - vmin)
                    color = "white" if norm < 0.35 else "black"
                    ax.text(j, i, f"{value:.3f}", ha="center", va="center", color=color, fontsize=8.2)
        ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2)
        ax.tick_params(which="minor", bottom=False, left=False)

    colorbar = fig.colorbar(image, ax=axes, shrink=0.78, pad=0.02)
    colorbar.set_label("Mean balanced-accuracy loss")
    fig.suptitle("Cross-dataset degradation is strongly direction- and model-dependent")
    return save_figure(fig, figure_dir / "Figure_1_Transfer_Loss_Heatmaps")


def figure_2_source_target(direction_model: pd.DataFrame, transfer: pd.DataFrame, figure_dir: Path) -> list[Path]:
    source_means = (
        transfer.groupby(["source_dataset", "model"], as_index=False)[
            "source_test_balanced_accuracy"
        ]
        .mean()
        .rename(columns={"source_test_balanced_accuracy": "source_ba"})
    )
    points = direction_model.merge(source_means, on=["source_dataset", "model"], how="left")

    fig, ax = plt.subplots(figsize=(6.2, 5.4), constrained_layout=True)
    for model in MODEL_ORDER:
        subset = points[points["model"] == model]
        ax.scatter(
            subset["source_ba"],
            subset["target_balanced_accuracy_mean"],
            s=48,
            marker=MODEL_MARKER[model],
            facecolor=MODEL_COLOR[model],
            edgecolor="white",
            linewidth=0.7,
            alpha=0.9,
            label=MODEL_LABEL[model],
        )
    ax.plot([0.25, 1.0], [0.25, 1.0], linestyle="--", color="#555555", linewidth=1.0, label="No degradation")
    ax.set_xlim(0.25, 1.01)
    ax.set_ylim(0.25, 1.01)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Mean source-test balanced accuracy")
    ax.set_ylabel("Mean target balanced accuracy")
    ax.set_title("All 36 direction–model means fall below source performance")
    ax.grid(True)
    ax.legend(loc="lower right", frameon=False)
    ax.text(
        0.28,
        0.96,
        "Mean loss across 180 jobs: 0.382",
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#BDBDBD"},
    )
    return save_figure(fig, figure_dir / "Figure_2_Source_vs_Target_Balanced_Accuracy")


def figure_3_warning_associations(
    units: pd.DataFrame, associations: pd.DataFrame, figure_dir: Path
) -> list[Path]:
    panels = [
        ("environmental_instability", "Environmental instability", True),
        ("sampling_environment_gap", "Sampling–environment gap", True),
        ("sampling_instability", "Sampling instability", False),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.1, 3.75), constrained_layout=True, sharey=True)

    for ax, (predictor, label, is_primary) in zip(axes, panels):
        for model in MODEL_ORDER:
            subset = units[units["model"] == model]
            ax.scatter(
                subset[predictor],
                subset["balanced_accuracy_loss"],
                s=55,
                marker=MODEL_MARKER[model],
                color=MODEL_COLOR[model],
                edgecolor="white",
                linewidth=0.7,
                label=MODEL_LABEL[model],
                zorder=3,
            )
        x = units[predictor].to_numpy(float)
        y = units["balanced_accuracy_loss"].to_numpy(float)
        if np.ptp(x) > 0:
            coefficient = np.polyfit(x, y, deg=1)
            xx = np.linspace(x.min(), x.max(), 100)
            ax.plot(xx, np.polyval(coefficient, xx), color="#707070", linewidth=1.1, zorder=2)
        record = associations.loc[associations["predictor"] == predictor].iloc[0]
        p_value = (
            record["holm_adjusted_p_primary_family"]
            if is_primary
            else record["exact_one_sided_p"]
        )
        p_label = "Holm p" if is_primary else "Exact p"
        ax.set_title(f"{label}\n$\\rho$ = {record['spearman_rho']:.3f}; {p_label} = {p_value:.3f}")
        ax.set_xlabel(label)
        ax.grid(True)
        ax.set_ylim(0.27, 0.51)
    axes[0].set_ylabel("Mean balanced-accuracy loss")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle("Source-side stability scores do not provide a reliable warning signal")
    return save_figure(fig, figure_dir / "Figure_3_Stability_Warning_Associations")


def clean_specification(name: str) -> str:
    mapping = {
        "mean_only": "Mean only",
        "model_only": "Model only",
        "model_plus_sampling_instability": "Model + sampling instability",
        "model_plus_environmental_instability": "Model + environmental instability",
        "model_plus_sampling_environment_gap": "Model + sampling–environment gap",
        "model_plus_primary_combined": "Model + combined primary warnings",
    }
    return mapping.get(name, name.replace("_", " "))


def figure_4_loso(loso: pd.DataFrame, figure_dir: Path) -> list[Path]:
    frame = loso.sort_values("oof_mae", ascending=False).copy()
    frame["label"] = frame["specification"].map(clean_specification)
    baseline = float(
        loso.loc[
            loso["specification"] == "model_plus_sampling_instability", "oof_mae"
        ].iloc[0]
    )
    colors = []
    for specification in frame["specification"]:
        if specification == "model_only":
            colors.append("#0072B2")
        elif specification == "model_plus_sampling_instability":
            colors.append("#8C8C8C")
        elif specification in {
            "model_plus_environmental_instability",
            "model_plus_sampling_environment_gap",
            "model_plus_primary_combined",
        }:
            colors.append("#D55E00")
        else:
            colors.append("#B3B3B3")

    fig, ax = plt.subplots(figsize=(7.3, 4.4), constrained_layout=True)
    y = np.arange(len(frame))
    bars = ax.barh(y, frame["oof_mae"], color=colors, height=0.66)
    ax.set_yticks(y, frame["label"])
    ax.set_xlim(0, 0.09)
    ax.set_xlabel("Out-of-fold mean absolute error (lower is better)")
    ax.set_title("Warning scores did not improve leave-one-dataset-out prediction")
    ax.axvline(baseline, color="#555555", linestyle="--", linewidth=1.0, label="Sampling-instability baseline")
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    for bar, value in zip(bars, frame["oof_mae"]):
        ax.text(value + 0.001, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=8.5)
    ax.legend(loc="lower right", frameon=False)
    return save_figure(fig, figure_dir / "Figure_4_LOSO_MAE_Comparison")


def direction_label(source: str, target: str) -> str:
    return f"{DATASET_LABEL[source]} → {DATASET_LABEL[target]}"


def figure_5_recall_imbalance(transfer: pd.DataFrame, figure_dir: Path) -> list[Path]:
    frame = (
        transfer.groupby(["source_dataset", "target_dataset"], as_index=False)
        .agg(
            benign_recall=("target_benign_recall", "mean"),
            attack_recall=("target_attack_recall", "mean"),
        )
    )
    frame["gap"] = (frame["benign_recall"] - frame["attack_recall"]).abs()
    frame["direction"] = [
        direction_label(source, target)
        for source, target in zip(frame["source_dataset"], frame["target_dataset"])
    ]
    frame = frame.sort_values("attack_recall", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8.0, 5.7), constrained_layout=True)
    y = np.arange(len(frame))
    for row_number, row in frame.iterrows():
        ax.plot(
            [row["benign_recall"], row["attack_recall"]],
            [row_number, row_number],
            color="#B8B8B8",
            linewidth=1.5,
            zorder=1,
        )
    ax.scatter(frame["benign_recall"], y, color="#0072B2", s=42, label="Benign recall", zorder=3)
    ax.scatter(frame["attack_recall"], y, color="#D55E00", s=42, label="Attack recall", zorder=3)
    ax.set_yticks(y, frame["direction"])
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Mean recall across models and seeds")
    ax.set_title("Several transfer directions detect one class while largely missing the other")
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=False,
    )
    return save_figure(fig, figure_dir / "Figure_5_Class_Recall_Imbalance")


def build_tables(data: dict, seal: dict, table_dir: Path) -> dict[str, pd.DataFrame]:
    transfer = data["transfer"]
    direction_model = data["direction_model"]
    associations = data["associations"].copy()
    loso = data["loso"].copy()

    design = pd.DataFrame(
        [
            ("Evaluated datasets", 4, "Common NetFlow-v3 schema"),
            ("Directed transfers", 12, "All ordered source–target pairs; no self-transfer"),
            ("Model families", 3, "Logistic regression, random forest, XGBoost"),
            ("Seeds per source–model", 5, "2027–2031"),
            ("Common predictors", 49, "Identifiers and labels excluded"),
            ("Target prediction jobs", 180, "4 sources × 3 targets × 3 models × 5 seeds"),
            ("Rows per target job", 80000, "40,000 benign and 40,000 attack"),
            ("Unique target-label rows", 400000, "Labels revealed only after A3 freeze"),
            ("Evaluated job–label associations", 14400000, "180 jobs × 80,000 rows"),
            ("Confirmatory association units", 12, "Source dataset × model, averaged over seeds and targets"),
            ("Exact blocked permutations", 13824, "4!³ permutations within model"),
            ("Bootstrap replicates", 10000, "Model-stratified source-dataset bootstrap"),
            ("Target adaptation", "No", "No target-fitted transformation or retraining"),
            ("Target threshold changes", "No", "Source-calibrated thresholds remained frozen"),
        ],
        columns=["Design element", "Value", "Definition"],
    )

    metric_columns = [
        ("source_test_balanced_accuracy", "Source BA"),
        ("target_balanced_accuracy", "Target BA"),
        ("balanced_accuracy_loss", "BA loss"),
        ("target_macro_f1", "Macro F1"),
        ("target_mcc", "MCC"),
        ("target_pr_auc_attack", "PR-AUC"),
        ("target_benign_recall", "Benign recall"),
        ("target_attack_recall", "Attack recall"),
    ]
    model_rows = []
    for model in MODEL_ORDER:
        subset = transfer[transfer["model"] == model]
        row = {"Model": MODEL_LABEL[model], "Jobs": len(subset)}
        for metric, label in metric_columns:
            row[label] = subset[metric].mean()
        model_rows.append(row)
    model_table = pd.DataFrame(model_rows)

    directions = (
        transfer.groupby(["source_dataset", "target_dataset"], as_index=False)
        .agg(
            Jobs=("balanced_accuracy_loss", "size"),
            **{
                "Target BA mean": ("target_balanced_accuracy", "mean"),
                "Target BA SD": ("target_balanced_accuracy", "std"),
                "BA loss mean": ("balanced_accuracy_loss", "mean"),
                "BA loss SD": ("balanced_accuracy_loss", "std"),
                "Macro F1 mean": ("target_macro_f1", "mean"),
                "MCC mean": ("target_mcc", "mean"),
                "PR-AUC mean": ("target_pr_auc_attack", "mean"),
                "Benign recall mean": ("target_benign_recall", "mean"),
                "Attack recall mean": ("target_attack_recall", "mean"),
            },
        )
    )
    directions.insert(
        0,
        "Direction",
        [
            direction_label(source, target)
            for source, target in zip(directions["source_dataset"], directions["target_dataset"])
        ],
    )
    directions = directions.drop(columns=["source_dataset", "target_dataset"]).sort_values(
        "BA loss mean", ascending=False
    )

    association_table = associations.rename(
        columns={
            "predictor": "Predictor",
            "role": "Role",
            "n_source_model_units": "n",
            "spearman_rho": "Spearman rho",
            "bootstrap_95_ci_low": "95% CI low",
            "bootstrap_95_ci_high": "95% CI high",
            "exact_one_sided_p": "Exact one-sided p",
            "holm_adjusted_p_primary_family": "Holm-adjusted p",
            "association_supported_at_0_05": "Supported at 0.05",
        }
    )[
        [
            "Predictor",
            "Role",
            "n",
            "Spearman rho",
            "95% CI low",
            "95% CI high",
            "Exact one-sided p",
            "Holm-adjusted p",
            "Supported at 0.05",
        ]
    ]
    association_table["Predictor"] = association_table["Predictor"].str.replace("_", " ")
    association_table["Supported at 0.05"] = association_table["Supported at 0.05"].map(
        {True: "Yes", False: "No"}
    )

    loso_table = loso.rename(
        columns={
            "specification": "Specification",
            "oof_units": "OOF units",
            "oof_mae": "OOF MAE",
            "oof_rmse": "OOF RMSE",
            "oof_spearman_rho": "OOF Spearman rho",
            "mae_improvement_vs_sampling_baseline": "MAE improvement vs sampling baseline",
        }
    )
    loso_table["Specification"] = loso_table["Specification"].map(clean_specification)
    loso_table = loso_table.sort_values("OOF MAE", ascending=True).reset_index(drop=True)

    direction_model_table = direction_model.copy()
    direction_model_table.insert(
        0,
        "Direction",
        [
            direction_label(source, target)
            for source, target in zip(
                direction_model_table["source_dataset"], direction_model_table["target_dataset"]
            )
        ],
    )
    direction_model_table["Model label"] = direction_model_table["model"].map(MODEL_LABEL)
    selected = [
        "Direction",
        "Model label",
        "seeds",
        "target_balanced_accuracy_mean",
        "target_balanced_accuracy_sd",
        "balanced_accuracy_loss_mean",
        "balanced_accuracy_loss_sd",
        "target_macro_f1_mean",
        "target_mcc_mean",
        "target_pr_auc_attack_mean",
        "target_benign_recall_mean",
        "target_attack_recall_mean",
    ]
    direction_model_table = direction_model_table[selected].rename(
        columns={
            "Model label": "Model",
            "seeds": "Seeds",
            "target_balanced_accuracy_mean": "Target BA mean",
            "target_balanced_accuracy_sd": "Target BA SD",
            "balanced_accuracy_loss_mean": "BA loss mean",
            "balanced_accuracy_loss_sd": "BA loss SD",
            "target_macro_f1_mean": "Macro F1 mean",
            "target_mcc_mean": "MCC mean",
            "target_pr_auc_attack_mean": "PR-AUC mean",
            "target_benign_recall_mean": "Benign recall mean",
            "target_attack_recall_mean": "Attack recall mean",
        }
    )

    headline = pd.DataFrame(
        [
            ("Mean source-test balanced accuracy", transfer["source_test_balanced_accuracy"].mean()),
            ("Mean target balanced accuracy", transfer["target_balanced_accuracy"].mean()),
            ("Mean balanced-accuracy loss", transfer["balanced_accuracy_loss"].mean()),
            ("Mean target Macro F1", transfer["target_macro_f1"].mean()),
            ("Mean target MCC", transfer["target_mcc"].mean()),
            ("Mean target attack PR-AUC", transfer["target_pr_auc_attack"].mean()),
            ("Jobs with BA loss > 0.30", int((transfer["balanced_accuracy_loss"] > 0.30).sum())),
            ("Jobs with target BA < 0.50", int((transfer["target_balanced_accuracy"] < 0.50).sum())),
            ("Jobs with negative MCC", int((transfer["target_mcc"] < 0).sum())),
            ("Final-results seal", seal["seal_sha256"]),
        ],
        columns=["Finding", "Value"],
    )

    tables = {
        "Table_1_Experimental_Design": design,
        "Table_2_Model_Performance": model_table,
        "Table_3_Directional_Transfer": directions,
        "Table_4_Confirmatory_Associations": association_table,
        "Table_5_LOSO_Validation": loso_table,
        "Table_S1_Direction_Model": direction_model_table,
        "Headline_Findings": headline,
    }
    for name, frame in tables.items():
        frame.to_csv(table_dir / f"{name}.csv", index=False)
    return tables


def frame_payload(frame: pd.DataFrame) -> dict:
    clean = frame.copy().replace({np.nan: None})
    return {"columns": list(clean.columns), "rows": clean.values.tolist()}


def write_workbook_payload(
    path: Path,
    tables: dict[str, pd.DataFrame],
    data: dict,
    verification: dict,
    input_dir: Path,
) -> None:
    payload = {
        "metadata": {
            "artifact": "paper3_publication_tables",
            "script_version": SCRIPT_VERSION,
            "source_directory": str(input_dir),
            "final_results_seal_sha256": verification["seal"]["seal_sha256"],
            "protocol_sha256": verification["seal"]["parent_protocol_sha256"],
            "amendment_A1_sha256": verification["seal"]["amendment_A1_sha256"],
            "amendment_A2_sha256": verification["seal"]["parent_amendment_A2_sha256"],
            "amendment_A3_sha256": verification["seal"][
                "amendment_A3_confirmatory_analysis_seal_sha256"
            ],
            "verified_input_hashes": verification["verified_hashes"],
        },
        "tables": {name: frame_payload(frame) for name, frame in tables.items()},
        "raw_transfer": frame_payload(data["transfer"]),
        "raw_direction_model": frame_payload(data["direction_model"]),
        "raw_associations": frame_payload(data["associations"]),
        "raw_loso": frame_payload(data["loso"]),
        "confirmatory_units": frame_payload(data["confirmatory_units"]),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_readme(output_dir: Path, seal_hash: str) -> None:
    text = f"""# Paper 3 manuscript assets

These assets were generated from the sealed Paper 3 final evaluation.

- Final-results seal: `{seal_hash}`
- Builder version: `{SCRIPT_VERSION}`
- Confirmatory results were retained regardless of direction or statistical significance.
- No target adaptation or target-driven threshold change was performed.

## Main figures

1. **Transfer-loss heatmaps:** mean balanced-accuracy loss for all 12 directions and three models.
2. **Source versus target balanced accuracy:** 36 direction–model means against the no-degradation line.
3. **Stability warning associations:** the two primary predictors and sampling-instability baseline at the 12-unit confirmatory level.
4. **LOSO predictive validation:** out-of-fold MAE for all prespecified warning specifications.
5. **Class-recall imbalance:** benign and attack recall by transfer direction, averaged over models and seeds.

Each figure is supplied as 600-dpi PNG, editable-vector SVG, and vector PDF.

## Manuscript tables

- Table 1: Experimental design and safeguards.
- Table 2: Performance by model across 60 target jobs per model.
- Table 3: Performance by directed transfer across 15 jobs per direction.
- Table 4: Prespecified confirmatory association tests.
- Table 5: Leave-one-source-dataset-out predictive validation.
- Table S1: Direction-by-model descriptive results.

## Safe interpretation

Under the evaluated four-dataset protocol, source-side feature stability did not provide reliable evidence of cross-dataset degradation and did not improve leave-one-dataset-out prediction beyond model identity alone. This statement is bounded to the evaluated NetFlow-v3 datasets, binary task, fixed models, and frozen protocol.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def write_captions(output_dir: Path) -> None:
    text = """# Publication-ready figure captions and table notes

## Figure captions

**Figure 1. Cross-dataset balanced-accuracy degradation by source, target, and model.** Each cell reports the mean source-test minus target balanced accuracy across five confirmatory seeds. The diagonal is excluded because self-transfer was not evaluated. Larger values indicate greater degradation. The common color scale permits direct comparison across logistic regression, random forest, and XGBoost.

**Figure 2. Source-test versus cross-dataset target balanced accuracy.** Each point represents one directed-transfer and model combination averaged across five seeds (36 units). The dashed equality line represents no degradation. Every evaluated unit falls below the equality line; across all 180 target jobs, mean balanced-accuracy loss was 0.382.

**Figure 3. Association between source-only stability warnings and cross-dataset degradation.** Points are the 12 prespecified source-dataset × model confirmatory units, with outcomes averaged across five seeds and three targets. Lines are descriptive least-squares fits and are not the inferential test. Environmental instability and the sampling–environment gap were the two primary predictors; displayed primary p values are Holm adjusted. Sampling instability was the prespecified baseline and displays its exact one-sided p value. None of the associations met the 0.05 support criterion.

**Figure 4. Leave-one-source-dataset-out prediction error for prespecified warning models.** Bars report out-of-fold mean absolute error across 60 source × seed × model units; lower values are better. The dashed line marks the sampling-instability baseline. Model identity alone achieved the lowest MAE (0.059), whereas adding the combined primary warnings increased MAE to 0.082.

**Figure 5. Benign-versus-attack recall imbalance across directed transfers.** Points report recall averaged across the three models and five seeds (15 target jobs per direction). Connecting lines show the within-direction separation between benign and attack recall. Several directions retain high recall for one class while largely missing the other, illustrating why balanced accuracy and class-specific recall are necessary alongside aggregate discrimination metrics.

## Table notes

**Table 1.** All target predictions were generated and sealed before target labels were opened. No target adaptation, target-fitted preprocessing, or target-driven threshold change was permitted.

**Table 2.** Values are arithmetic means across 60 target jobs per model (12 directed transfers × five seeds). BA denotes balanced accuracy; PR-AUC is reported for the attack class.

**Table 3.** Values are arithmetic means across 15 jobs per direction (three models × five seeds). Directions are ordered from largest to smallest mean balanced-accuracy loss.

**Table 4.** Confidence intervals are model-stratified bootstrap intervals based on 10,000 replicates. Exact positive one-sided p values use all 13,824 blocked permutations within model. Holm adjustment applies to the two primary predictors only; the sampling-instability baseline is reported without family-wise adjustment.

**Table 5.** Predictive specifications were evaluated with four-fold leave-one-source-dataset-out validation over 60 source × seed × model units. Positive MAE improvement indicates lower error than the sampling-instability baseline.

**Supplementary Table S1.** Each direction–model row summarizes five confirmatory seeds.
"""
    (output_dir / "Figure_Captions_and_Table_Notes.md").write_text(
        text, encoding="utf-8"
    )


def build_manifest(output_dir: Path, input_dir: Path, seal_hash: str) -> Path:
    files = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name not in {"assets_manifest.json", "Paper3_Publication_Assets.zip"}:
            files.append(
                {
                    "path": path.relative_to(output_dir).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    manifest = {
        "artifact": "paper3_manuscript_assets_manifest",
        "builder_version": SCRIPT_VERSION,
        "source_directory": str(input_dir),
        "parent_final_results_seal_sha256": seal_hash,
        "files": files,
    }
    path = output_dir / "assets_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", help="Directory containing the six sealed result files")
    parser.add_argument("--output-dir", help="Separate directory for publication assets")
    args = parser.parse_args()

    input_dir, output_dir = resolve_paths(args)
    figure_dir = output_dir / "figures"
    table_dir = output_dir / "tables"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("PAPER 3 — PUBLICATION FIGURES AND MANUSCRIPT TABLES")
    print("=" * 88)
    verification = verify_inputs(input_dir)
    print("Final-results seal verification: PASS")
    data = load_and_reconcile(input_dir)
    print("Numerical reconciliation: PASS")
    print("Reconstructed confirmatory-unit correlations: PASS")

    publication_style()
    figure_outputs = []
    figure_outputs += figure_1_heatmaps(data["direction_model"], figure_dir)
    figure_outputs += figure_2_source_target(data["direction_model"], data["transfer"], figure_dir)
    figure_outputs += figure_3_warning_associations(
        data["confirmatory_units"], data["associations"], figure_dir
    )
    figure_outputs += figure_4_loso(data["loso"], figure_dir)
    figure_outputs += figure_5_recall_imbalance(data["transfer"], figure_dir)
    print(f"Publication figures written: {len(figure_outputs)} files (5 figures × 3 formats)")

    tables = build_tables(data, verification["seal"], table_dir)
    data["confirmatory_units"].to_csv(
        table_dir / "Figure_3_Confirmatory_Units_12.csv", index=False
    )
    print(f"Manuscript tables written: {len(tables)} CSV files")

    payload_path = output_dir / "paper3_workbook_payload.json"
    write_workbook_payload(payload_path, tables, data, verification, input_dir)
    write_readme(output_dir, verification["seal"]["seal_sha256"])
    write_captions(output_dir)

    script_destination = output_dir / Path(__file__).name
    if Path(__file__).resolve() != script_destination.resolve():
        shutil.copy2(Path(__file__), script_destination)

    manifest_path = build_manifest(
        output_dir, input_dir, verification["seal"]["seal_sha256"]
    )
    print(f"Workbook payload: {payload_path}")
    print(f"Manifest: {manifest_path}")
    print("ASSET GENERATION: PASS")
    print("=" * 88)


if __name__ == "__main__":
    main()
