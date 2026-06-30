import pandas as pd
import numpy as np
import os
import argparse

def generate_data(output_dir):
    """Generates synthetic customer and transactional data."""
    np.random.seed(42)
    n_customers = 5000
    
    # Customer Data
    customer_ids = np.arange(1, n_customers + 1)
    ages = np.random.randint(18, 70, size=n_customers)
    # Introduce some missing values in ages
    ages = np.where(np.random.rand(n_customers) < 0.05, np.nan, ages)
    
    genders = np.random.choice(['Male', 'Female', 'Other', np.nan], size=n_customers, p=[0.48, 0.48, 0.02, 0.02])
    locations = np.random.choice(['Urban', 'Suburban', 'Rural'], size=n_customers, p=[0.6, 0.3, 0.1])
    tenure_months = np.random.randint(1, 60, size=n_customers)
    
    # Introduce some noise and missing values in income
    income = np.random.normal(60000, 20000, size=n_customers)
    income = np.where(np.random.rand(n_customers) < 0.1, np.nan, income)
    
    df_customers = pd.DataFrame({
        'customer_id': customer_ids,
        'age': ages,
        'gender': genders,
        'location': locations,
        'tenure_months': tenure_months,
        'estimated_income': income
    })
    
    # Transaction Data
    n_transactions = 25000
    txn_customer_ids = np.random.choice(customer_ids, size=n_transactions)
    txn_amounts = np.random.exponential(scale=50, size=n_transactions)
    txn_dates = pd.to_datetime('2023-01-01') + pd.to_timedelta(np.random.randint(0, 365, size=n_transactions), unit='D')
    
    # Introduce anomalies
    txn_amounts[np.random.choice(n_transactions, size=50)] = np.random.uniform(1000, 5000, size=50)
    
    df_transactions = pd.DataFrame({
        'transaction_id': np.arange(1, n_transactions + 1),
        'customer_id': txn_customer_ids,
        'transaction_date': txn_dates,
        'amount': txn_amounts
    })
    
    # Save data
    os.makedirs(output_dir, exist_ok=True)
    customers_path = os.path.join(output_dir, 'customers.csv')
    transactions_path = os.path.join(output_dir, 'transactions.csv')
    
    df_customers.to_csv(customers_path, index=False)
    df_transactions.to_csv(transactions_path, index=False)
    print(f"Data generated at {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='data/raw')
    args = parser.parse_args()
    generate_data(args.output_dir)
