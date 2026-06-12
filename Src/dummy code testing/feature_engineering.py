import pandas as pd
import numpy as np

# Load the raw data
df = pd.read_csv('data/invoices.csv', parse_dates=['invoice_date', 'due_date', 'payment_date'])

# ── Target variable ──────────────────────────────────────────
df['days_late'] = (df['payment_date'] - df['due_date']).dt.days
df['is_late'] = (df['days_late'] > 0).astype(int)

# ── Invoice level features ───────────────────────────────────
df['invoice_age_at_due'] = (df['due_date'] - df['invoice_date']).dt.days
df['invoice_month'] = df['invoice_date'].dt.month
df['invoice_quarter'] = df['invoice_date'].dt.quarter
df['is_end_of_month'] = (df['invoice_date'].dt.day >= 25).astype(int)

# ── Historical payment features per customer ─────────────────
customer_stats = df.groupby('customer_id')['days_late'].agg(avg_days_to_pay='mean',std_days_to_pay='std',max_days_late_ever='max',pct_invoices_paid_late=lambda x: (x > 0).mean()).reset_index()
df = df.merge(customer_stats, on='customer_id', how='left')
df['std_days_to_pay'] = df['std_days_to_pay'].fillna(0)

# ── Save engineered dataset ──────────────────────────────────
df.to_csv('data/invoices_engineered.csv', index=False)
print("Feature engineering done:", df.shape)
print(df[['invoice_id', 'days_late', 'is_late', 'avg_days_to_pay', 'pct_invoices_paid_late']].head())