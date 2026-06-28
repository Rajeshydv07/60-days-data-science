import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

def train_and_save_model():
    print("Loading data...")
    # Assuming run from day43-model-api directory
    data_path = '../day41-churn-prediction/data/churn_data.csv'
    df = pd.read_csv(data_path)
    
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
    
    # Save the model, scaler and label encoder
    os.makedirs('models', exist_ok=True)
    joblib.dump(rf_model, 'models/churn_rf_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(le, 'models/label_encoder.pkl')
    
    print("Model and preprocessors saved to 'models/' directory.")

if __name__ == "__main__":
    train_and_save_model()
