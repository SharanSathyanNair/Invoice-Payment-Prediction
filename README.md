# Invoice Payment Prediction

A machine learning system that predicts whether invoices will be paid late 
and estimates how many days late payment will arrive.

## Pipeline
1. generate_realistic_data.py — generates invoice and payment data
2. load_and_prepare.py — joins tables and splits into 3 buckets
3. feature_engineering_real.py — computes features for ML
4. train_real_model.py — trains Logistic Regression and XGBoost
5. predict_open_invoices.py — scores open invoices by risk level
6. predict_days_late.py — estimates days late for high risk invoices

## Models
- Logistic Regression (baseline)
- XGBoost Classifier (late/on-time prediction)
- XGBoost Regressor (days late estimation)

## Tech Stack
Python, pandas, scikit-learn, XGBoost, imbalanced-learn, matplotlib