import pandas as pd
import numpy as np 
np.random.seed(42)
n=1000 #number of invoices 
df=pd.DataFrame({'invoice_id': range(1001, 2001),
    'customer_id': np.random.randint(1, 51, n),
    'invoice_date': pd.date_range(start='2022-01-01', periods=n, freq='D'),
    'due_date': pd.date_range(start='2022-01-01', periods=n, freq='D') + pd.Timedelta(days=30),
    'invoice_amount': np.random.randint(1000, 50000, n),
    'payment_terms_days': np.random.choice([15, 30, 60], n),
    'payment_date': pd.date_range(start='2022-01-01', periods=n, freq='D')+ pd.Timedelta(days=30)+ pd.to_timedelta(np.random.randint(-5, 45, n), unit='D')
})
df.to_csv('data/invoices.csv', index=False)
print("Dataset Created",df.shape)
print(df.head())
