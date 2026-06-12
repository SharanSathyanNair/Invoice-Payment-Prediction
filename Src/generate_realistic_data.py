import pandas as pd
import numpy as np

np.random.seed(42)

# ── 50 customers with distinct payment personalities ──────────
customer_profiles = []
for i in range(1, 51):
    profile_type = np.random.choice(
        ['reliable', 'always_late', 'seasonal', 'early_payer', 'risky'],
        p=[0.45, 0.15, 0.15, 0.15, 0.10]
    )
    customer_profiles.append({
        'customer_id': i,
        'profile': profile_type,
        'avg_delay': {
            'reliable':    np.random.randint(-5, 2),
            'always_late': np.random.randint(10, 30),
            'seasonal':    np.random.randint(2, 10),
            'early_payer': np.random.randint(-15, -5),
            'risky':       np.random.randint(20, 60)
        }[profile_type],
        'std_delay': {
            'reliable':    np.random.randint(1, 3),
            'always_late': np.random.randint(2, 6),
            'seasonal':    np.random.randint(3, 8),
            'early_payer': np.random.randint(1, 3),
            'risky':       np.random.randint(5, 15)
        }[profile_type]
    })

customers_df = pd.DataFrame(customer_profiles)

# ── Generate invoices ─────────────────────────────────────────
invoices = []
payments = []
invoice_id = 1001
payment_id = 1

start_date = pd.Timestamp('2024-01-01')
end_date = pd.Timestamp('2025-12-31')

for _, customer in customers_df.iterrows():
    n_invoices = np.random.randint(20, 41)
    
    for _ in range(n_invoices):
        days_offset = np.random.randint(0, 730)
        invoice_date = start_date + pd.Timedelta(days=days_offset)
        
        terms = np.random.choice([15, 30, 60], p=[0.2, 0.6, 0.2])
        due_date = invoice_date + pd.Timedelta(days=terms)
        
        amount = np.random.randint(
            5000 if customer['profile'] == 'risky' else 1000,
            50000
        )
        
        base_delay = customer['avg_delay']
        
        if invoice_date.month in [12, 3]:
            if customer['profile'] == 'seasonal':
                base_delay += np.random.randint(5, 12)
        
        actual_delay = int(np.random.normal(base_delay, customer['std_delay']))
        payment_date = due_date + pd.Timedelta(days=actual_delay)
        
        if payment_date <= invoice_date:
            payment_date = invoice_date + pd.Timedelta(days=2)
        
        is_partial = (
            customer['profile'] in ['risky', 'always_late'] and
            np.random.random() < 0.20
        )

        # define will_settle before invoices.append
        will_settle = (np.random.random() < 0.70) if is_partial else False

        invoices.append({
            'invoice_id': f'INV-{invoice_id:04d}',
            'customer_id': int(customer['customer_id']),
            'invoice_date': invoice_date.date(),
            'due_date': due_date.date(),
            'invoice_amount': amount,
            'payment_terms_days': terms,
            'status': 'partial_settled' if (is_partial and will_settle) 
                      else 'partial_bad_debt' if is_partial 
                      else 'paid'
        })
        
        if not is_partial:
            # single full payment
            payments.append({
                'payment_id': f'PAY-{payment_id:04d}',
                'invoice_id': f'INV-{invoice_id:04d}',
                'customer_id': int(customer['customer_id']),
                'payment_date': payment_date.date(),
                'amount': amount
            })
            payment_id += 1
        else:
            # first partial payment
            partial_amount = int(amount * np.random.uniform(0.3, 0.7))
            partial_date = due_date + pd.Timedelta(days=np.random.randint(5, 20))
            
            payments.append({
                'payment_id': f'PAY-{payment_id:04d}',
                'invoice_id': f'INV-{invoice_id:04d}',
                'customer_id': int(customer['customer_id']),
                'payment_date': partial_date.date(),
                'amount': partial_amount
            })
            payment_id += 1
            
            # Type A — second payment that fully settles it
            if will_settle:
                final_amount = amount - partial_amount
                final_date = partial_date + pd.Timedelta(
                    days=np.random.randint(10, 40))
                payments.append({
                    'payment_id': f'PAY-{payment_id:04d}',
                    'invoice_id': f'INV-{invoice_id:04d}',
                    'customer_id': int(customer['customer_id']),
                    'payment_date': final_date.date(),
                    'amount': final_amount
                })
                payment_id += 1

        invoice_id += 1

invoices_df = pd.DataFrame(invoices)
payments_df = pd.DataFrame(payments)

# ── Generate 50 open invoices (no payment yet) ───────────────
open_invoices = []
for i in range(50):
    customer = customers_df.sample(1).iloc[0]
    invoice_date = pd.Timestamp('2026-05-01') + pd.Timedelta(
        days=np.random.randint(0, 30))
    terms = np.random.choice([15, 30, 60], p=[0.2, 0.6, 0.2])
    due_date = invoice_date + pd.Timedelta(days=terms)
    amount = np.random.randint(1000, 50000)
    
    open_invoices.append({
        'invoice_id': f'INV-{invoice_id + i:04d}',
        'customer_id': int(customer['customer_id']),
        'invoice_date': invoice_date.date(),
        'due_date': due_date.date(),
        'invoice_amount': amount,
        'payment_terms_days': terms,
        'status': 'open'
    })

open_df = pd.DataFrame(open_invoices)
invoices_df = pd.concat([invoices_df, open_df], ignore_index=True)

invoices_df.to_csv('data/zoho_invoices.csv', index=False)
payments_df.to_csv('data/zoho_payments.csv', index=False)

print("Invoices:", invoices_df.shape)
print("Payments:", payments_df.shape)
print("\nInvoice status breakdown:")
print(invoices_df['status'].value_counts())
print("\nCustomer profile breakdown:")
print(customers_df['profile'].value_counts())
print("\nSample invoices:")
print(invoices_df.head())
print("\nSample payments:")
print(payments_df.head())