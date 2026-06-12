import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import pickle

# ── Load data ─────────────────────────────────────────────────
bucket1 = pd.read_csv('data/bucket1_settled.csv',
                       parse_dates=['invoice_date', 'due_date', 'payment_date'])
bucket3 = pd.read_csv('data/bucket3_partial.csv',
                       parse_dates=['invoice_date', 'due_date', 'payment_date'])

print("Bucket 3 size:", len(bucket3))

# ── Build Bucket 3 features ───────────────────────────────────
# Compute customer bad debt rate from bucket1
# How often has this customer had late payments historically
customer_stats = bucket1.groupby('customer_id')['days_late'].agg(
    avg_days_to_pay='mean',
    pct_invoices_paid_late=lambda x: (x > 0).mean(),
    max_days_late_ever='max'
).reset_index()

# Join customer stats onto bucket3
bucket3 = bucket3.merge(customer_stats, on='customer_id', how='left')

# ── Bucket 3 specific features ────────────────────────────────
bucket3['total_paid'] = bucket3['total_paid'].fillna(0)
bucket3['pct_paid_so_far'] = bucket3['total_paid'] / bucket3['invoice_amount']
bucket3['remaining_balance'] = bucket3['invoice_amount'] - bucket3['total_paid']
bucket3['days_overdue'] = (bucket3['payment_date'] - bucket3['due_date']).dt.days
bucket3['days_since_last_payment'] = (
    pd.Timestamp('2026-01-01') - bucket3['payment_date']
).dt.days

# ── Target variable ───────────────────────────────────────────
# will_settle = 1 if partial_settled, 0 if partial_bad_debt
bucket3['will_settle'] = (bucket3['status'] == 'partial_settled').astype(int)

print("\nWill settle distribution:")
print(bucket3['will_settle'].value_counts())

# ── Features ──────────────────────────────────────────────────
feature_cols_b3 = [
    'invoice_amount', 'payment_terms_days',
    'pct_paid_so_far', 'remaining_balance',
    'days_overdue', 'days_since_last_payment',
    'avg_days_to_pay', 'pct_invoices_paid_late',
    'max_days_late_ever'
]

# Fill any nulls from new customers
for col in ['avg_days_to_pay', 'pct_invoices_paid_late', 'max_days_late_ever']:
    bucket3[col] = bucket3[col].fillna(0)

X = bucket3[feature_cols_b3]
y = bucket3['will_settle']

print("\nFeatures shape:", X.shape)

# ── Train / test split ────────────────────────────────────────
# Note: with only 20 invoices we use 70/30 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print("Training rows:", len(X_train))
print("Test rows:", len(X_test))

# ── Apply SMOTE only if needed ────────────────────────────────
if y_train.value_counts().min() > 1:
    smote = SMOTE(random_state=42, k_neighbors=min(2, y_train.value_counts().min()-1))
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print("Training rows after SMOTE:", len(X_train))

# ── Train XGBoost Classifier ──────────────────────────────────
model = XGBClassifier(
    n_estimators=50,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
model.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────
y_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

print("\n── Bucket 3 Model Results ──")
if len(set(y_test)) > 1:
    auc = roc_auc_score(y_test, y_proba)
    print("ROC-AUC:", round(auc, 4))
print(classification_report(y_test, y_pred,
      target_names=['Bad Debt', 'Will Settle']))

# ── Score all Bucket 3 invoices ───────────────────────────────
all_proba = model.predict_proba(X)[:, 1]
bucket3['settle_probability'] = (all_proba * 100).round(1)
bucket3['prediction'] = np.where(all_proba >= 0.5, 'Will Settle', 'Bad Debt')

print("\n── All Bucket 3 Invoices ──")
print(bucket3[['invoice_id', 'customer_id', 'invoice_amount',
               'pct_paid_so_far', 'settle_probability',
               'prediction', 'status']].to_string(index=False))

# ── Save model ────────────────────────────────────────────────
with open('models/bucket3_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('models/feature_cols_b3.pkl', 'wb') as f:
    pickle.dump(feature_cols_b3, f)

bucket3.to_csv('data/bucket3_predictions.csv', index=False)
print("\nBucket 3 model saved to models/bucket3_model.pkl")