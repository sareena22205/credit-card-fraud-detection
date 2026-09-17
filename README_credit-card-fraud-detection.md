# Credit Card Fraud Detection using Machine Learning

Classification models to detect fraudulent credit card transactions on a highly imbalanced dataset.

## Overview
Built and compared multiple ML classifiers to flag fraudulent transactions in a dataset of ~495,000 records, where fraud cases represent a tiny fraction of the total. Focused on handling class imbalance and maximizing detection sensitivity without sacrificing precision.

## Tech Stack
- Python
- Scikit-learn (Logistic Regression, Random Forest, XGBoost)
- Pandas, NumPy
- SQL (data querying/staging)
- imbalanced-learn (SMOTE)

## Approach
1. **EDA & Feature Engineering** — explored transaction patterns, distributions, and correlations across ~495K transactions.
2. **Class Imbalance Handling** — applied SMOTE (Synthetic Minority Oversampling Technique) to rebalance the training data.
3. **Model Training** — trained and tuned Logistic Regression, Random Forest, and XGBoost classifiers.
4. **Validation** — used cross-validation to ensure model robustness and avoid overfitting to the resampled data.

## Results
- Achieved **96% accuracy** across the best-performing model.
- Improved fraud-case detection sensitivity by **18%** after SMOTE and feature engineering, compared to the baseline model.

## Project Structure
```
credit-card-fraud-detection/
├── data/                  # Raw and processed transaction data
├── notebooks/             # EDA and experimentation notebooks
├── src/
│   ├── preprocessing.py   # Cleaning, feature engineering, SMOTE
│   ├── train.py           # Model training and cross-validation
│   └── evaluate.py        # Metrics, confusion matrix, ROC-AUC
├── requirements.txt
└── README.md
```

## How to Run
```bash
pip install -r requirements.txt
python src/preprocessing.py
python src/train.py
python src/evaluate.py
```

## Author
Sareena Ashfaque Shaikh — [LinkedIn](https://linkedin.com/in/sareena-ashfaque-shaikh)
