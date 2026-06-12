import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor
from imblearn.over_sampling import SMOTE

# ── Load feature engineered bucket 1 ─────────────────────────
bucket1 = pd.read_csv('data/bucket1_features.csv')

feature_cols = [
    'invoice_amount', 'payment_terms_days', 'invoice_age_at_due',
    'invoice_month', 'invoice_quarter', 'is_end_of_month',
    'avg_days_to_pay', 'std_days_to_pay', 'max_days_late_ever',
    'pct_invoices_paid_late', 'last_3_avg'
]

X = bucket1[feature_cols]
y = bucket1['is_late']

# ── Balance with SMOTE ────────────────────────────────────────
smote = SMOTE(random_state=42)
X_bal, y_bal = smote.fit_resample(X, y)

# ── Train and save Logistic Regression ────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_bal)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_scaled, y_bal)

# ── Train and save XGBoost Classifier ────────────────────────
xgb_clf = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    random_state=42
)
xgb_clf.fit(X_bal, y_bal)

# ── Train and save XGBoost Regressor ─────────────────────────
late_invoices = bucket1[bucket1['is_late'] == 1]
X_reg = late_invoices[feature_cols]
y_reg = late_invoices['days_late']

xgb_reg = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    random_state=42
)
xgb_reg.fit(X_reg, y_reg)

# ── Save all models and scaler to disk ────────────────────────
with open('models/logistic_regression.pkl', 'wb') as f:
    pickle.dump(lr, f)

with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('models/xgb_classifier.pkl', 'wb') as f:
    pickle.dump(xgb_clf, f)

with open('models/xgb_regressor.pkl', 'wb') as f:
    pickle.dump(xgb_reg, f)

with open('models/feature_cols.pkl', 'wb') as f:
    pickle.dump(feature_cols, f)

print("All models saved to models/")
print("  logistic_regression.pkl")
print("  scaler.pkl")
print("  xgb_classifier.pkl")
print("  xgb_regressor.pkl")
print("  feature_cols.pkl")