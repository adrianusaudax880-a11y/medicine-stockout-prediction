# Medicine Stockout Prediction

Predicting next-month medicine stockouts at health facilities using historical
consumption, stock balance, and receipt data — so facility staff and supply
chain teams can restock proactively instead of reactively.

## Problem Statement

Health facilities frequently run out of essential medicines, disrupting patient
care. This project builds a machine learning model that predicts whether a
given product at a given facility will experience a **stockout next month**,
using the current and historical months' consumption, opening/closing balance,
and receipt data.

- **Target variable:** `stockout_next_month` (1 = stockout, 0 = no stockout)
- **Unit of prediction:** one product, at one facility, for one month
- **Input:** monthly consumption/balance/receipt records with engineered lag
  features (previous month, 3-month rolling averages)

## Repository Structure

```
.
├── README.md
├── data/
│   └── medicine_stockout_prediction.csv
├── notebooks/
│   └── Medicine_Stockout_Prediction_FULL.ipynb   # Weeks 1-3 in one continuous notebook
├── reports/
│   └── Final_Technical_Report.docx
├── app.py
├── xgb_stockout_model.pkl
├── contribution_log.xlsx
└── requirements.txt
```

## Setup

```bash
git clone <repo-url>
cd medicine-stockout-prediction
pip install -r requirements.txt
jupyter notebook
```

Open `notebooks/Medicine_Stockout_Prediction_FULL.ipynb` and run all cells
**top to bottom in order** — later sections depend on variables created
earlier in the notebook (data split, class-imbalance handling, etc.).

## Methodology

| Week | Focus | Key steps |
|---|---|---|
| 1 | Problem Definition & EDA | Data dictionary, missing-value checks, class balance, distribution plots |
| 2 | Preprocessing & Baseline | Missing-value handling, feature selection, time-based Train/Validation/Test split, outlier handling (`log1p`), class-imbalance handling (`scale_pos_weight`, SMOTE), Logistic Regression baseline, first XGBoost model |
| 3 | Advanced Modeling & Tuning | Random Forest as a third model family, hyperparameter tuning with `GridSearchCV` restricted to the Validation fold (`PredefinedSplit`), final Test-set evaluation, error analysis by product/facility |

**Note on the train/test split:** rows are split **by date**, not randomly, so
the model is always evaluated on months it has never seen — this avoids the
model "seeing the future" during training.

## Results Summary

The best-performing model (XGBoost with `scale_pos_weight`) reached a ROC-AUC of
0.840 on the untouched Test set. The Logistic Regression baseline, despite the
lowest ROC-AUC, achieved the highest recall (0.72) on the Stockout class at the
default threshold — catching more real stockouts than any tree-based model.
Hyperparameter tuning via GridSearchCV produced a negligible change (0.838),
suggesting the hand-chosen settings were already close to optimal for this
feature set.

| Model | ROC-AUC (Test) | Recall — Stockout |
|---|---|---|
| Logistic Regression (baseline) | 0.819 | 0.72 |
| XGBoost + scale_pos_weight | **0.840** | 0.63 |
| XGBoost + SMOTE | 0.831 | 0.52 |
| Random Forest + class_weight | 0.833 | 0.55 |
| XGBoost (tuned via GridSearchCV) | 0.838 | 0.58 |

Dataset: 5,653 rows / 19 columns, 5,501 rows after cleaning, 13 model features.
Overall stockout rate: ~16.5% (933 of 5,653 records).

## Error Analysis Highlights

On the Test set, the final tuned model produced 198 true positives, 145 false
negatives (missed stockouts), 235 false positives, and 1,412 true negatives.
The lowest-recall products include Paracetamol (Acetaminophen) 250mg Dispersible
(35% recall), Salbutamol 100mcg/dose Aerosol Inhaler (36%), and Povidone Iodine
10% Solution (41%) — likely due to irregular, demand-driven consumption patterns
that lag-based features capture less reliably. A handful of facilities (e.g.
facility 10008 at 30% recall) are also consistently harder to predict, flagging
them for additional manual review rather than full automation.

## Team

This is a group project by a team of 11. See `contribution_log.xlsx` for each
member's role and task assignment across all four weeks.

## Next Steps

- Try LightGBM as an additional model for comparison
- Consider adding `hf_pk` / `productID` as categorical features
- Package the best model behind a simple demo (e.g. Streamlit) for the final
  presentation
