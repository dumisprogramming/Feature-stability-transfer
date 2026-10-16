# Saved result tables

Rows below describe stored analysis units, not independent raw network flows.

| File | Rows | Meaning |
|---|---:|---|
| `confirmatory_association_results.csv` | 3 | Two primary predictors plus the sampling-instability comparator |
| `direction_model_summary_36.csv` | 36 | 12 directions × 3 models, averaged over five seeds |
| `loso_model_scores.csv` | 6 | One row per prediction specification |
| `loso_oof_predictions.csv` | 360 | 60 units × six prespecified prediction specifications |
| `source_model_confirmatory_units_12.csv` | 12 | Four sources × three models, averaged over seeds and target datasets |
| `source_seed_model_primary_units_60.csv` | 60 | Four sources × five seeds × three models, averaged over target datasets |
| `target_metrics_all_180.csv` | 180 | One row per target-evaluation job |
| `transfer_analysis_all_180.csv` | 180 | One row per directed-transfer × seed × model job, joined with source metrics and warning scores |

## Conventions

- BA is balanced accuracy, on a 0–1 scale. Loss is source-test BA minus target BA.
- Macro F1, PR-AUC and recalls are also stored as proportions; MCC ranges from −1 to 1.
- Confidence limits refer to the original model-stratified source bootstrap; exact p-values refer to blocked permutations.
- Only the two primary predictors are in the Holm correction family. Blank baseline adjusted p-values are intentional.
- OOF means out-of-fold. LOSO holds out the source role.
- Positive MAE improvement is relative to the model-plus-sampling-instability baseline.
- `final_analysis_summary.json` is a saved summary; `final_results_seal.json` records hashes and provenance links.

## Exact CSV columns

### confirmatory_association_results.csv

`predictor`, `role`, `n_source_model_units`, `spearman_rho`, `bootstrap_95_ci_low`, `bootstrap_95_ci_high`, `finite_bootstrap_replicates`, `exact_one_sided_p`, `permutation_exceedances`, `permutations`, `holm_adjusted_p_primary_family`, `association_supported_at_0_05`

### direction_model_summary_36.csv

`source_dataset`, `target_dataset`, `model`, `seeds`, `target_balanced_accuracy_mean`, `target_balanced_accuracy_sd`, `target_balanced_accuracy_min`, `target_balanced_accuracy_max`, `target_macro_f1_mean`, `target_macro_f1_sd`, `target_macro_f1_min`, `target_macro_f1_max`, `target_mcc_mean`, `target_mcc_sd`, `target_mcc_min`, `target_mcc_max`, `target_pr_auc_attack_mean`, `target_pr_auc_attack_sd`, `target_pr_auc_attack_min`, `target_pr_auc_attack_max`, `target_benign_recall_mean`, `target_benign_recall_sd`, `target_benign_recall_min`, `target_benign_recall_max`, `target_attack_recall_mean`, `target_attack_recall_sd`, `target_attack_recall_min`, `target_attack_recall_max`, `balanced_accuracy_loss_mean`, `balanced_accuracy_loss_sd`, `balanced_accuracy_loss_min`, `balanced_accuracy_loss_max`

### loso_model_scores.csv

`specification`, `oof_units`, `oof_mae`, `oof_rmse`, `oof_spearman_rho`, `mae_improvement_vs_sampling_baseline`

### loso_oof_predictions.csv

`specification`, `held_out_source_dataset`, `source_dataset`, `seed`, `model`, `actual_mean_balanced_accuracy_loss`, `predicted_mean_balanced_accuracy_loss`

### source_model_confirmatory_units_12.csv

`source_dataset`, `model`, `mean_balanced_accuracy_loss`, `sd_seed_mean_balanced_accuracy_loss`, `seeds`, `environmental_instability`, `sampling_environment_gap`, `sampling_instability`, `environmental_stability`, `sampling_stability`

### source_seed_model_primary_units_60.csv

`source_dataset`, `seed`, `model`, `mean_balanced_accuracy_loss`, `sd_balanced_accuracy_loss`, `min_balanced_accuracy_loss`, `max_balanced_accuracy_loss`, `target_datasets`, `environmental_instability`, `sampling_environment_gap`, `sampling_instability`, `environmental_stability`, `sampling_stability`

### target_metrics_all_180.csv

`source_dataset`, `target_dataset`, `seed`, `model`, `target_rows`, `target_benign_rows`, `target_attack_rows`, `target_balanced_accuracy`, `target_macro_f1`, `target_mcc`, `target_pr_auc_attack`, `target_benign_recall`, `target_attack_recall`, `tn`, `fp`, `fn`, `tp`, `prediction_file_sha256`

### transfer_analysis_all_180.csv

`source_dataset`, `target_dataset`, `seed`, `model`, `target_rows`, `target_benign_rows`, `target_attack_rows`, `target_balanced_accuracy`, `target_macro_f1`, `target_mcc`, `target_pr_auc_attack`, `target_benign_recall`, `target_attack_recall`, `tn`, `fp`, `fn`, `tp`, `prediction_file_sha256`, `source_test_balanced_accuracy`, `environmental_stability`, `sampling_stability`, `environmental_instability`, `sampling_environment_gap`, `environmental_top10_jaccard`, `sampling_top10_jaccard`, `environmental_groups`, `sampling_groups`, `environmental_pairs`, `sampling_pairs`, `undefined_environmental_pairs`, `undefined_sampling_pairs`, `mean_environmental_baseline_ba`, `mean_sampling_baseline_ba`, `sampling_instability`, `balanced_accuracy_loss`

