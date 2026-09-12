# Bank Marketing Prediction

A machine learning project that predicts whether a bank customer will subscribe to a term deposit, using data from a Portuguese bank's phone marketing campaign.

## Dataset

Bank Marketing dataset from the UCI Machine Learning Repository:
https://archive.ics.uci.edu/dataset/222/bank+marketing

File used: `bank-additional-full.csv`. Each row is one phone call with customer details, campaign details, economic indicators, and a target column `y` (yes/no).

## Results

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Random Forest | 86.4% | 0.818 |
| XGBoost | 84.9% | 0.811 |

Random Forest did slightly better.

Classification report (Random Forest):

| Class | Precision | Recall | F1 |
|---|---|---|---|
| no  | 0.95 | 0.89 | 0.92 |
| yes | 0.43 | 0.64 | 0.51 |

Top features were euribor3m, nr.employed, and emp.var.rate, so the economic indicators mattered more than the customer's personal details.

## Files

| File | What it is |
|---|---|
| `train.py` | main script |
| `bank-additional-full.csv` | dataset |
| `feature_importance.png` | feature importance chart |
| `pca_plot.png` | PCA scatter plot |

## Requirements

Python 3.12+, plus:

```
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
```

## How to run

```
python train.py
```
