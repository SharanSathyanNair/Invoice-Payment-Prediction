import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# ── Load data ─────────────────────────────────────────────────
bucket1 = pd.read_csv('data/bucket1_features.csv')
bucket2 = pd.read_csv('data/bucket2_features.csv', 
                       parse_dates=['invoice_date', 'due_date'])

feature_cols = [
    'invoice_amount', 'payment_terms_days', 'invoice_age_at_due',
    'invoice_month', 'invoice_quarter', 'is_end_of_month',
    'avg_days_to_pay', 'std_days_to_pay', 'max_days_late_ever',
    'pct_invoices_paid_late', 'last_3_avg'
]

# ── Train final model on ALL of Bucket 1 ─────────────────────
# No train/test split here — we use 100% of settled data
# to train the final model before predicting open invoices
X_train = bucket1[feature_cols]
y_train = bucket1['is_late']

smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_bal)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train_bal)

print("Final model trained on", len(X_train_bal), "rows")

# ── Predict Bucket 2 ──────────────────────────────────────────
X_predict = bucket2[feature_cols]
X_predict_scaled = scaler.transform(X_predict)

probabilities = model.predict_proba(X_predict_scaled)[:, 1]

# ── Build results table ───────────────────────────────────────
results = bucket2[['invoice_id', 'customer_id', 
                    'invoice_date', 'due_date', 
                    'invoice_amount']].copy()

results['late_probability'] = (probabilities * 100).round(1)
results['risk_level'] = pd.cut(
    probabilities,
    bins=[0, 0.3, 0.6, 1.0],
    labels=['Low', 'Medium', 'High']
)
results['days_until_due'] = (results['due_date'] - pd.Timestamp.today()).dt.days.astype(int)

# Sort by highest risk first
results = results.sort_values('late_probability', ascending=False)

# ── Print results ─────────────────────────────────────────────
print("\n── Open Invoice Risk Scores ──")
print(results.to_string(index=False))

print("\n── Risk Summary ──")
print(results['risk_level'].value_counts())

print("\n── High Risk Invoices (action needed) ──")
high_risk = results[results['risk_level'] == 'High']
medium_risk = results[results['risk_level'] == 'Medium']
low_risk = results[results['risk_level'] == 'Low']
print(f"{len(high_risk)} invoices flagged as high risk")
print(high_risk[['invoice_id', 'customer_id', 
                  'invoice_amount', 'late_probability', 
                  'days_until_due']].to_string(index=False))

# ── Save ──────────────────────────────────────────────────────
results.to_csv('data/open_invoice_predictions.csv', index=False)
print("\nPredictions saved to data/open_invoice_predictions.csv")