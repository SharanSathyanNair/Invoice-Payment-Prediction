import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# ── Load data ─────────────────────────────────────────────────
bucket1 = pd.read_csv('data/bucket1_features.csv')

# Train regressor only on invoices that were actually late
# No point training on on-time invoices — we already know
# they won't be late, nothing to predict
late_invoices = bucket1[bucket1['is_late'] == 1].copy()

print("Late invoices for regression training:", len(late_invoices))
print("Avg days late:", round(late_invoices['days_late'].mean(), 1))
print("Max days late:", late_invoices['days_late'].max())
print("Min days late:", late_invoices['days_late'].min())

feature_cols = [
    'invoice_amount', 'payment_terms_days', 'invoice_age_at_due',
    'invoice_month', 'invoice_quarter', 'is_end_of_month',
    'avg_days_to_pay', 'std_days_to_pay', 'max_days_late_ever',
    'pct_invoices_paid_late', 'last_3_avg'
]

X = late_invoices[feature_cols]
y = late_invoices['days_late']

# ── Train / test split ────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Train XGBoost Regressor ───────────────────────────────────
regressor = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    random_state=42
)
regressor.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────
y_pred = regressor.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"\nModel MAE: {round(mae, 1)} days")
print("Meaning: predictions are off by avg", round(mae, 1), "days")

# ── Predict on high risk open invoices ────────────────────────
predictions = pd.read_csv('data/open_invoice_predictions.csv')
high_risk = predictions[predictions['risk_level'] == 'High'].copy()

bucket2 = pd.read_csv('data/bucket2_features.csv')
high_risk_features = bucket2[
    bucket2['invoice_id'].isin(high_risk['invoice_id'])
][feature_cols]

days_late_pred = regressor.predict(high_risk_features)
days_late_pred = np.maximum(days_late_pred, 1).round().astype(int)

# ── Build final output ────────────────────────────────────────
high_risk = high_risk.copy()
high_risk['estimated_days_late'] = days_late_pred
high_risk['estimated_payment_date'] = (
    pd.to_datetime(high_risk['due_date']) + 
    pd.to_timedelta(high_risk['estimated_days_late'], unit='D')
)

# ── Print final actionable output ────────────────────────────
print("\n── High Risk Invoices — Full Prediction ──")
print(f"{'Invoice':<12} {'Customer':<10} {'Amount':>10} {'Late%':>7} {'Est.Days Late':>14} {'Est.Payment Date':<20} {'Action'}")
print("-" * 90)

for _, row in high_risk.iterrows():
    if row['late_probability'] >= 80:
        action = "Call today"
    elif row['late_probability'] >= 60:
        action = "Send reminder"
    else:
        action = "Monitor"
    print(
        f"{row['invoice_id']:<12} "
        f"{int(row['customer_id']):<10} "
        f"₹{int(row['invoice_amount']):>9,} "
        f"{row['late_probability']:>6.1f}% "
        f"{row['estimated_days_late']:>14} "
        f"{str(row['estimated_payment_date'])[:10]:<20} "
        f"{action}"
    )

# ── Save ──────────────────────────────────────────────────────
high_risk.to_csv('data/high_risk_with_days.csv', index=False)
print("\nSaved to data/high_risk_with_days.csv")