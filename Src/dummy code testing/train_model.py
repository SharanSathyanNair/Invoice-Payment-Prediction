import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

#Load engineered data
df=pd.read_csv('data/invoices_engineered.csv')

# Define features and target
feature_cols = [
    'invoice_amount', 'payment_terms_days', 'invoice_age_at_due',
    'invoice_month', 'invoice_quarter', 'is_end_of_month',
    'avg_days_to_pay', 'std_days_to_pay', 'max_days_late_ever',
    'pct_invoices_paid_late'
]
X=df[feature_cols]
y=df['is_late']

#Step 1-Train/test split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
print("Training rows:",len(X_train),"|Test Rows:",len(X_test))
print("Late Invoices in training(before SMOTE):",y_train.sum())

#Step 2- SMOTE on training set only 
smote=SMOTE(random_state=42)
X_train_balanced,y_train_balanced=smote.fit_resample(X_train,y_train)
print("Trining Rows after SMOTE:",len(X_train_balanced))

# Step 3 — Logistic Regression baseline
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train_balanced)
lr_proba = lr.predict_proba(X_test_scaled)[:, 1]
lr_auc = roc_auc_score(y_test, lr_proba)
print("\nLogistic Regression ROC-AUC:", round(lr_auc, 4))

#Step 4- XGBoost
xgb=XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
xgb.fit(X_train_balanced,y_train_balanced)
xgb_proba = xgb.predict_proba(X_test)[:, 1]
xgb_auc = roc_auc_score(y_test, xgb_proba)
print("XGBoost ROC-AUC:", round(xgb_auc, 4))

#Step 5- Compare
print("\nWinner:","XGBoost" if xgb_auc>lr_auc else"Logistic Regression")

from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# ── Precision, Recall, F1 ─────────────────────────────────────
xgb_preds = xgb.predict(X_test)
print("\n── XGBoost Classification Report ──")
print(classification_report(y_test, xgb_preds, target_names=['On Time', 'Late']))

# ── Feature Importance ────────────────────────────────────────
importance = pd.Series(
    xgb.feature_importances_,
    index=feature_cols
).sort_values(ascending=True)

plt.figure(figsize=(8, 6))
importance.plot(kind='barh', color='steelblue')
plt.title('XGBoost Feature Importance')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('models/feature_importance.png')
plt.show()
print("\nFeature importance chart saved to models/")
