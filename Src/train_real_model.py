import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score,classification_report
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# ── Load feature engineered bucket 1 ─────────────────────────
df = pd.read_csv('data/bucket1_features.csv')

feature_cols = [
    'invoice_amount', 'payment_terms_days', 'invoice_age_at_due',
    'invoice_month', 'invoice_quarter', 'is_end_of_month',
    'avg_days_to_pay', 'std_days_to_pay', 'max_days_late_ever',
    'pct_invoices_paid_late', 'last_3_avg'
]

X=df[feature_cols]
y=df['is_late']

print("Total Invoices:",len(df))
print("Late rate:", round(y.mean() * 100, 1), "%")

#________________Train / Test Split_______________________________
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

print("\nTraining Rows:",len(X_train))
print("\nTesting Rows:",len(X_test))

#_______________SMOTE TRAINING_____________________________________
smote=SMOTE(random_state=42)
X_train_bal,y_train_bal=smote.fit_resample(X_train,y_train)
print("Training rows after SMOTE:", len(X_train_bal))

#______________Logistic regression baseline_________________________
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_bal)
X_test_scaled = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train_bal)
lr_proba = lr.predict_proba(X_test_scaled)[:, 1]
lr_auc = roc_auc_score(y_test, lr_proba)
print("\nLogistic Regression ROC-AUC:", round(lr_auc,4))

#_____________XGBoost_______________________________________________
xgb = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    random_state=42
)
xgb.fit(X_train_bal, y_train_bal)
xgb_proba = xgb.predict_proba(X_test)[:, 1]
xgb_auc = roc_auc_score(y_test, xgb_proba)
print("XGBoost ROC-AUC:", round(xgb_auc, 4))

print("\n── Model Comparison ──")
print(f"Logistic Regression : {round(lr_auc, 4)}")
print(f"XGBoost             : {round(xgb_auc, 4)}")
winner = "XGBoost" if xgb_auc > lr_auc else "Logistic Regression"
print(f"Winner              : {winner}")

# ── Classification report ─────────────────────────────────────
best_proba = xgb_proba if xgb_auc > lr_auc else lr_proba
best_model = xgb if xgb_auc > lr_auc else lr
best_preds = best_model.predict(X_test if winner == "XGBoost" else X_test_scaled)

print("\n── Classification Report ──")
print(classification_report(y_test, best_preds, target_names=['On Time', 'Late']))

# ── Feature importance ────────────────────────────────────────
importance = pd.Series(
    xgb.feature_importances_,
    index=feature_cols
).sort_values(ascending=True)

plt.figure(figsize=(8, 6))
importance.plot(kind='barh', color='steelblue')
plt.title('XGBoost Feature Importance — Real Data')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('models/feature_importance_real.png')
plt.show()
print("Feature importance saved to models/")