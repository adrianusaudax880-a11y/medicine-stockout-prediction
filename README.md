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
│   ├── Week1_Problem_Definition_EDA.ipynb
│   └── Week3_Medicine_Stockout_Prediction.ipynb   # includes Week 2 + Week 3 work
├── reports/
│   └── final_report.pdf                            # (add once written)
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

Open `notebooks/Week3_Medicine_Stockout_Prediction.ipynb` and run all cells
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

*(Fill in with the final numbers from the notebook's Section 16 comparison
table before submitting — e.g. best model, its ROC-AUC, precision/recall on
the Stockout class.)*

| Model | ROC-AUC (Test) |
|---|---|
| Logistic Regression (baseline) | — |
| XGBoost + scale_pos_weight | — |
| XGBoost + SMOTE | — |
| Random Forest | — |
| XGBoost (tuned) | — |

## Error Analysis Highlights

*(Fill in 2–3 sentences from Section 18–19 of the notebook — e.g. which
products/facilities the model struggles with most, and why.)*

## Team

This is a group project by a team of 10. See `contribution_log.xlsx` for each
member's individual contribution across all four weeks.

## Next Steps

- Try LightGBM as an additional model for comparison
- Consider adding `hf_pk` / `productID` as categorical features
- Package the best model behind a simple demo (e.g. Streamlit) for the final
  presentation
