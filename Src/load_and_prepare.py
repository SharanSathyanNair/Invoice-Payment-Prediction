import pandas as pd

# ── Load both tables ──────────────────────────────────────────
invoices = pd.read_csv('data/zoho_invoices.csv', parse_dates=['invoice_date', 'due_date'])
payments = pd.read_csv('data/zoho_payments.csv', parse_dates=['payment_date'])

print("Invoices loaded:", invoices.shape)
print("Payments loaded:", payments.shape)

# ── Group payments by invoice_id ──────────────────────────────
payments_grouped = payments.groupby('invoice_id').agg(
    payment_date=('payment_date', 'max'),
    total_paid=('amount', 'sum')
).reset_index()

# ── Join invoices with payments ───────────────────────────────
df = invoices.merge(payments_grouped, on='invoice_id', how='left')

print("\nAfter join:", df.shape)
print("Null payment_dates (open invoices):", df['payment_date'].isna().sum())

# ── Sort into 3 buckets ───────────────────────────────────────
bucket1 = df[
    (df['payment_date'].notna()) & 
    (df['total_paid'] >= df['invoice_amount'])
].copy()

bucket2 = df[df['payment_date'].isna()].copy()

bucket3 = df[
    (df['payment_date'].notna()) & 
    (df['total_paid'] < df['invoice_amount'])
].copy()

print("\n── Bucket Summary ──")
print(f"Bucket 1 (Fully settled) : {len(bucket1)} invoices → training data")
print(f"Bucket 2 (Open)          : {len(bucket2)} invoices → prediction targets")
print(f"Bucket 3 (Partial)       : {len(bucket3)} invoices → excluded for now")
print(f"Total accounted for      : {len(bucket1) + len(bucket2) + len(bucket3)}")

# ── Compute target variable on Bucket 1 ──────────────────────
bucket1['days_late'] = (bucket1['payment_date'] - bucket1['due_date']).dt.days
bucket1['is_late'] = (bucket1['days_late'] > 0).astype(int)

print("\n── Bucket 1 class distribution ──")
print(bucket1['is_late'].value_counts())
print(f"Late rate: {bucket1['is_late'].mean():.1%}")



# ── Save buckets ──────────────────────────────────────────────
bucket1.to_csv('data/bucket1_settled.csv', index=False)
bucket2.to_csv('data/bucket2_open.csv', index=False)
bucket3.to_csv('data/bucket3_partial.csv', index=False)

print("\nAll 3 buckets saved to data/")