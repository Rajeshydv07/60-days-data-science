import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

def train_and_score():
    print("Loading data...")
    df = pd.read_csv('../data/churn_data.csv')
    
    le = LabelEncoder()
    df['contract_type_encoded'] = le.fit_transform(df['contract_type'])
    
    features = [
        'age', 'tenure_months', 'monthly_charges', 'total_charges', 
        'contract_type_encoded', 'support_tickets', 
        'days_since_last_login', 'satisfaction_score'
    ]
    X = df[features]
    y = df['churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    print("Training model...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_model.fit(X_train_scaled, y_train)
    
    print("Scoring all customers...")
    X_scaled_all = scaler.transform(X)
    df['churn_risk_score'] = rf_model.predict_proba(X_scaled_all)[:, 1]
    
    df['risk_group'] = pd.cut(df['churn_risk_score'], bins=[-0.01, 0.25, 0.5, 0.75, 1.01], labels=['Low', 'Medium', 'High', 'Critical'])
    
    df.to_csv('../data/churn_scored.csv', index=False)
    print("Saved scored customers to ../data/churn_scored.csv")

if __name__ == "__main__":
    train_and_score()
