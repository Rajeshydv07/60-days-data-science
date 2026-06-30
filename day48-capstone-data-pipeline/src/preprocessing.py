import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class OutlierClipper(BaseEstimator, TransformerMixin):
    """Clips outliers based on the IQR method."""
    def __init__(self, factor=1.5, columns=None):
        self.factor = factor
        self.columns = columns
        self.bounds_ = {}
        
    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        cols = self.columns if self.columns else X.select_dtypes(include=np.number).columns
        for col in cols:
            q1 = X[col].quantile(0.25)
            q3 = X[col].quantile(0.75)
            iqr = q3 - q1
            self.bounds_[col] = (q1 - self.factor * iqr, q3 + self.factor * iqr)
        return self
        
    def transform(self, X):
        X = pd.DataFrame(X).copy()
        cols = self.columns if self.columns else self.bounds_.keys()
        for col in cols:
            lower, upper = self.bounds_[col]
            X[col] = X[col].clip(lower=lower, upper=upper)
        return X

class DateFeaturesExtractor(BaseEstimator, TransformerMixin):
    """Extracts features from datetime columns."""
    def __init__(self, date_col):
        self.date_col = date_col
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X = pd.DataFrame(X).copy()
        X[self.date_col] = pd.to_datetime(X[self.date_col])
        X[f'{self.date_col}_year'] = X[self.date_col].dt.year
        X[f'{self.date_col}_month'] = X[self.date_col].dt.month
        X[f'{self.date_col}_day'] = X[self.date_col].dt.day
        X[f'{self.date_col}_dayofweek'] = X[self.date_col].dt.dayofweek
        X = X.drop(columns=[self.date_col])
        return X

class FeatureAggregator(BaseEstimator, TransformerMixin):
    """Aggregates transactional data to customer level."""
    def __init__(self, groupby_col, agg_dict):
        self.groupby_col = groupby_col
        self.agg_dict = agg_dict
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X = pd.DataFrame(X)
        return X.groupby(self.groupby_col).agg(self.agg_dict).reset_index()
