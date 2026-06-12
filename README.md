# Invoice Payment Prediction

An end-to-end ML system that predicts invoice payment delays and estimates 
how many days late payment will arrive — modelled on enterprise AR platforms 
like Tesorio, Kolleno, and Chaser.

## What it does
- Classifies open invoices as High / Medium / Low payment risk
- Estimates how many days late each high risk invoice will be paid
- Predicts whether partially paid invoices will fully settle or become bad debt
- Displays all predictions in an interactive Streamlit dashboard

## Model Performance
| Model | Metric | Score |
| Logistic Regression | ROC-AUC | 0.9775 |
| XGBoost Classifier | ROC-AUC | 0.9661 |
| XGBoost Classifier | Accuracy | 91% |
| XGBoost Regressor | MAE | 4.5 days |
| Bucket 3 Classifier | ROC-AUC | 1.0 |

## Pipeline — run in this order
1. `Src/generate_realistic_data.py` — generates invoice and payment data
2. `Src/load_and_prepare.py` — joins tables, splits into 3 buckets
3. `Src/feature_engineering_real.py` — engineers 11 features for ML
4. `Src/train_real_model.py` — trains and evaluates LR + XGBoost
5. `Src/predict_open_invoices.py` — scores open invoices by risk level
6. `Src/predict_days_late.py` — estimates days late for high risk invoices
7. `Src/bucket3_model.py` — predicts partial payment resolution
8. `Src/save_models.py` — saves all trained models to disk
9. `app.py` — launches Str