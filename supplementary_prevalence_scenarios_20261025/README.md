# Completed prevalence sensitivity calculations

These post hoc scenarios use verified confusion counts from 180 fixed-model unstratified evaluations. The tested attack fractions are 0.1%, 0.5%, 1%, 5%, 10% and 50%. Each scenario holds the observed attack recall (TPR) and false-positive rate (FPR) fixed. This assumes only a change in the class prior, not a change in conditional traffic distributions or attack-family composition. These fractions are hypothetical; they are not estimates of deployment prevalence.

Precision is pi * TPR / [pi * TPR + (1 - pi) * FPR]. Expected false alerts per million flows are 1,000,000 * (1 - pi) * FPR. Balanced accuracy does not change under this assumed class-prior shift. We report equal-job means of per-job precision, not precision obtained by pooling different datasets. No confidence intervals or significance tests are claimed.

| Assumed attack rate | Classifier | Mean scenario precision | Mean false alerts per million flows |
|---|---|---:|---:|
| 0.1% | LogisticRegression | 0.264% | 327,523 |
| 0.1% | RandomForest | 0.192% | 280,570 |
| 0.1% | XGBoost | 0.209% | 270,783 |
| 0.5% | LogisticRegression | 1.296% | 326,211 |
| 0.5% | RandomForest | 0.949% | 279,447 |
| 0.5% | XGBoost | 1.033% | 269,698 |
| 1% | LogisticRegression | 2.532% | 324,572 |
| 1% | RandomForest | 1.876% | 278,043 |
| 1% | XGBoost | 2.037% | 268,343 |
| 5% | LogisticRegression | 11.052% | 311,458 |
| 5% | RandomForest | 8.649% | 266,809 |
| 5% | XGBoost | 9.233% | 257,501 |
| 10% | LogisticRegression | 19.726% | 295,065 |
| 10% | RandomForest | 15.940% | 252,766 |
| 10% | XGBoost | 16.776% | 243,948 |
| 50% | LogisticRegression | 63.639% | 163,925 |
| 50% | RandomForest | 56.097% | 140,426 |
| 50% | XGBoost | 56.447% | 135,527 |

At the 0.1% assumed attack rate, mean scenario precision is below 0.3% for all three models. This calculation illustrates how the observed false-positive rates can dominate an alert stream at a low class prior. It does not estimate actual production alert counts or validate the class-conditional invariance assumption.

Run `python verify_prevalence.py` to check all 1,080 rows against the supplied original integer confusion counts. Source and input-file hashes are recorded in prevalence_scenarios.json. No new detector prediction, model fit or raw-data scan was performed. These calculations extend the operational sensitivity discussion; genuine production-prevalence validation remains uncompleted.
