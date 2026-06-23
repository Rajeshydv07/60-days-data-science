import pandas as pd
import numpy as np
import os

def generate_churn_data(n_samples=5000, seed=42):
    np.random.seed(seed)
    
    # Customer basic info
    customer_id = [f"CUST_{i:05d}" for i in range(1, n_samples + 1)]
    age = np.random.normal(40, 12, n_samples).astype(int)
    age = np.clip(age, 18, 80)
    
    # Behavior metrics
    tenure_months = np.random.randint(1, 72, n_samples)
    monthly_charges = np.random.normal(65, 25, n_samples)
    monthly_charges = np.clip(monthly_charges, 15, 150)
    total_charges = tenure_months * monthly_charges * np.random.uniform(0.8, 1.2, n_samples)
    
    # Engagement signals
    support_tickets = np.random.poisson(0.5, n_samples)
    days_since_last_login = np.random.exponential(10, n_samples).astype(int)
    satisfaction_score = np.random.randint(1, 6, n_samples)
    
    # Introducing correlation for churn
    # Churn is more likely if: high support tickets, high days since last login, low satisfaction, high monthly charges, low tenure
    churn_prob = (
        0.1 
        + 0.15 * (support_tickets > 2)
        + 0.2 * (days_since_last_login > 20)
        - 0.1 * (satisfaction_score >= 4)
        + 0.1 * (satisfaction_score <= 2)
        + 0.1 * (monthly_charges > 100)
        - 0.15 * (tenure_months > 36)
    )
    churn_prob = np.clip(churn_prob, 0, 1)
    
    churn = np.random.binomial(1, churn_prob)
    
    # Contract types (correlates with tenure/churn)
    contract_types = ['Month-to-month', 'One year', 'Two year']
    contract_probs = [0.55, 0.25, 0.20]
    contract = np.random.choice(contract_types, size=n_samples, p=contract_probs)
    
    # Override some churn based on contract to make it realistic
    churn = np.where(contract == 'Two year', np.random.binomial(1, churn_prob * 0.2), churn)
    churn = np.where(contract == 'One year', np.random.binomial(1, churn_prob * 0.5), churn)
    
    df = pd.DataFrame({
        'customer_id': customer_id,
        'age': age,
        'tenure_months': tenure_months,
        'monthly_charges': monthly_charges,
        'total_charges': total_charges,
        'contract_type': contract,
        'support_tickets': support_tickets,
        'days_since_last_login': days_since_last_login,
        'satisfaction_score': satisfaction_score,
        'churn': churn
    })
    
    return df

if __name__ == "__main__":
    df = generate_churn_data()
    output_dir = "../data"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, "churn_data.csv")
    df.to_csv(file_path, index=False)
    print(f"Generated {len(df)} records at {file_path}")
