import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = [
            'age', 'tenure_months', 'monthly_charges', 
            'total_purchases', 'support_calls', 
            'last_interaction_days', 'satisfaction_score'
        ]
        
    def prepare_data(self, df):
        # Handle any missing values if they existed
        df = df.dropna()
        
        X = df[self.feature_columns]
        y = df['churn']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale numerical features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, df
        
    def scale_features_for_segmentation(self, df):
        # For segmentation, we might want to use specific behavioral features
        seg_features = ['monthly_charges', 'total_purchases', 'tenure_months']
        X_seg = df[seg_features]
        X_scaled = self.scaler.fit_transform(X_seg)
        return X_scaled, df
