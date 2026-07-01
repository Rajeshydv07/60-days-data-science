import pandas as pd
import numpy as np

def generate_customer_data(n_samples=1000, random_state=42):
    np.random.seed(random_state)
    
    data = {
        'customer_id': range(1, n_samples + 1),
        'age': np.random.randint(18, 70, size=n_samples),
        'tenure_months': np.random.randint(1, 60, size=n_samples),
        'monthly_charges': np.random.uniform(20, 150, size=n_samples),
        'total_purchases': np.random.randint(1, 100, size=n_samples),
        'support_calls': np.random.randint(0, 10, size=n_samples),
        'last_interaction_days': np.random.randint(1, 90, size=n_samples),
        'satisfaction_score': np.random.randint(1, 6, size=n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Introduce some logical relationships for churn
    # High support calls, low satisfaction, and low tenure increase churn probability
    churn_prob = (
        (df['support_calls'] > 5).astype(int) * 0.3 +
        (df['satisfaction_score'] < 3).astype(int) * 0.4 +
        (df['tenure_months'] < 12).astype(int) * 0.2 +
        np.random.uniform(0, 0.2, size=n_samples)
    )
    
    df['churn'] = (churn_prob > 0.6).astype(int)
    
    return df

if __name__ == "__main__":
    df = generate_customer_data()
    df.to_csv("customer_data.csv", index=False)
    print("Generated synthetic customer data.")
