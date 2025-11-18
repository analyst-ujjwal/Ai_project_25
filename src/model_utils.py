import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, comment='#')

def split_features_target(df: pd.DataFrame, target_col='price'):
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, X):
        self.scaler.fit(X)
        self.fitted = True
        return self

    def transform(self, X):
        if not self.fitted:
            raise RuntimeError("Preprocessor not fitted")
        return self.scaler.transform(X)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

def save_model(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)

def load_model(path):
    return joblib.load(path)
