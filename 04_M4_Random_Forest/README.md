# 04_M4_Random_Forest — Random Forest churn model (Muskan)

## What was done
- Trained a Random Forest on the same data, features, split (80/20, stratified, `random_state=42`) and preprocessing as M3 (Logistic Regression) and M5 (XGBoost).
- Ran a limited search (20 random combinations, 5-fold CV on the training set only, scored by ROC-AUC). Test set used once.
- Evaluated Accuracy, Precision, Recall, F1, ROC-AUC; made a confusion matrix, ROC curve (vs Logistic Regression) and feature importance.

## Test-set results (1,409 customers, threshold 0.5)
| Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| 0.7537 | 0.5242 | 0.7834 | 0.6281 | 0.8442 |

Confusion matrix: TN 769, FP 266, FN 81, TP 293.
Best parameters: n_estimators=300, max_depth=6, min_samples_leaf=5, max_features=0.3, class_weight="balanced_subsample".

## Files
| File | Purpose |
|---|---|
| `04_random_forest.ipynb` | Full notebook (run top to bottom) |
| `random_forest.pkl` | Saved pipeline (preprocessing + model); accepts raw customer columns |
| `random_forest_metrics.csv / .json` | Metrics, best parameters, confusion matrix |
| `random_forest_test_predictions.csv` | customerID, actual_churn, rf_predicted_churn, rf_churn_probability (shared test rows, same order as M5's file) |
| `random_forest_feature_importance.csv` | Importance per one-hot feature |
| `random_forest_feature_importance_grouped.csv` | Importance rolled up per original column |
| `figures/` | Confusion matrix, ROC curve, feature importance |
| `telco_with_ids.csv` | M2 output used as input (copied so the notebook runs standalone) |

## How to run
1. Keep `telco_with_ids.csv` next to the notebook (or in `../02_M2_EDA/`).
2. `pip install pandas numpy scikit-learn matplotlib seaborn joblib`
3. Kernel → Restart & Run All. Takes about a minute or two.
4. To use the model: `joblib.load("random_forest.pkl").predict_proba(raw_customer_dataframe)[:, 1]`
   (the .pkl was saved with scikit-learn 1.8.0; use the same version if loading fails).

## What the next members need
- **M6:** `random_forest_metrics.csv`, `random_forest_test_predictions.csv`, `random_forest.pkl`. Test rows are identical to M5's, so join on `customerID`.
- **M7:** `random_forest_feature_importance*.csv`, `figures/`, `random_forest.pkl`.
