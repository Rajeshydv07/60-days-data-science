import pandas as pd
import numpy as np
import sys
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

sys.path.append('src')
from preprocessing import OutlierClipper, DateFeaturesExtractor, FeatureAggregator

def main():
    # Load Data
    df_customers = pd.read_csv('data/raw/customers.csv')
    df_transactions = pd.read_csv('data/raw/transactions.csv')
    
    # Transaction Pipeline
    transaction_pipeline = Pipeline([
        ('date_features', DateFeaturesExtractor(date_col='transaction_date'))
    ])
    df_transactions_processed = transaction_pipeline.fit_transform(df_transactions)
    
    # Aggregate transactions
    aggregator = FeatureAggregator(
        groupby_col='customer_id',
        agg_dict={
            'amount': ['sum', 'mean', 'count'],
            'transaction_date_year': 'max'
        }
    )
    df_txn_agg = aggregator.fit_transform(df_transactions_processed)
    df_txn_agg.columns = ['customer_id', 'total_spend', 'avg_transaction_value', 'transaction_count', 'last_transaction_year']
    
    # Merge
    df_merged = df_customers.merge(df_txn_agg, on='customer_id', how='left')
    
    # Customer Pipeline
    numeric_features = ['age', 'tenure_months', 'estimated_income', 'total_spend', 'avg_transaction_value', 'transaction_count']
    categorical_features = ['gender', 'location']
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('outlier_clipper', OutlierClipper(factor=1.5)),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
        
    X_processed = preprocessor.fit_transform(df_merged)
    cat_feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
    all_feature_names = numeric_features + list(cat_feature_names)
    
    df_cleaned = pd.DataFrame(X_processed, columns=all_feature_names)
    df_cleaned['customer_id'] = df_merged['customer_id'].values
    
    os.makedirs('data/cleaned', exist_ok=True)
    df_cleaned.to_csv('data/cleaned/customers_preprocessed.csv', index=False)
    print("Pipeline executed successfully. Cleaned data saved.")

if __name__ == '__main__':
    main()
