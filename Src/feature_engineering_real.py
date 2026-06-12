import pandas as pd
import numpy as np

# ── Load buckets ──────────────────────────────────────────────
bucket1 = pd.read_csv('data/bucket1_settled.csv', 
                       parse_dates=['invoice_date', 'due_date', 'payment_date'])
bucket2 = pd.read_csv('data/bucket2_open.csv', 
                       parse_dates=['invoice_date', 'due_date'])

print("Bucket 1:", bucket1.shape)
print("Bucket 2:", bucket2.shape)

# ── Compute customer stats from Bucket 1 only ─────────────────
customer_stats = bucket1.groupby('customer_id')['days_late'].agg(
    avg_days_to_pay='mean',
    std_days_to_pay='std',
    max_days_late_ever='max',
    pct_invoices_paid_late=lambda x: (x > 0).mean(),
    last_3_avg=lambda x: x.tail(3).mean()
).reset_index()

customer_stats['std_days_to_pay'] = customer_stats['std_days_to_pay'].fillna(0)

print("Customer stats computed for", len(customer_stats), "customers")

# ── Feature engineering function ──────────────────────────────
def add_features(df, customer_stats):
    df = df.merge(customer_stats, on='customer_id', how='left')
    df['invoice_month'] = df['invoice_date'].dt.month
    df['invoice_quarter'] = df['invoice_date'].dt.quarter
    df['is_end_of_month'] = (df['invoice_date'].dt.day >= 25).astype(int)
    df['invoice_age_at_due'] = (df['due_date'] - df['invoice_date']).dt.days
    df['avg_days_to_pay'] = df['avg_days_to_pay'].fillna(0)
    df['max_days_late_ever'] = df['max_days_late_ever'].fillna(0)
    df['pct_invoices_paid_late'] = df['pct_invoices_paid_late'].fillna(0.5)
    df['last_3_avg'] = df['last_3_avg'].fillna(0)
    return df
#____Apply to both Buckets__________________________________________
bucket1=add_features(bucket1,customer_stats)
bucket2=add_features(bucket2,customer_stats)

print("\nBucket 1 after feature engineering:",bucket1.shape)
print("\nBucket 2 after feature engineering:",bucket2.shape)

#_________Save_______________________________________________________________
bucket1.to_csv('data/bucket1_features.csv',index=False)
bucket2.to_csv('data/bucket2_features.csv',index=False)


print("\nFeatured engineered buckets have been saved")
print("\nFeature columns available:")
feature_cols = [
    'invoice_amount', 'payment_terms_days', 'invoice_age_at_due',
    'invoice_month', 'invoice_quarter', 'is_end_of_month',
    'avg_days_to_pay', 'std_days_to_pay', 'max_days_late_ever',
    'pct_invoices_paid_late', 'last_3_avg'
]
print(feature_cols)

















