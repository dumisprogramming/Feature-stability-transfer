# Recorded and tested environments

## Original final-evaluation record

These strings are copied from the historical final-results seal; that exact environment has not been recreated here.

```json
{
  "numpy": "2.1.3",
  "pandas": "2.2.3",
  "python": "3.13.15",
  "scikit_learn": "1.6.1",
  "scipy": "1.16.3"
}
```

The recovered notebook training-cell output additionally records XGBoost 3.4.1 and joblib 1.5.3, alongside NumPy 2.1.3, pandas 2.2.3 and scikit-learn 1.6.1. That saved training-cell execution reused all twenty completed dataset/seed checkpoints; its printed model total is not evidence of retraining during that saved run. The received four target scripts import additional packages such as PyArrow and Google Colab; their original full dependency lockfile is absent. Do not infer that the final-evaluation version record is a complete training environment.

## Environment used for portable checks

Python 3.12.14

- numpy: 2.3.5
- pandas: 2.2.3
- scipy: 1.17.0
- scikit-learn: 1.8.0
- matplotlib: 3.10.8

- pyarrow: 26.0.0 (added for recovered stability checks; not a claim about the historical Colab runtime)
