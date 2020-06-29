import os
from pathlib import Path
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

BASE_PATH = Path(__file__).resolve().parents[2]

# Data loading
def load_data(src=BASE_PATH.joinpath('data', 'processed')):
    data = pd.read_csv(src.joinpath('features.csv'))
    return data.drop(columns='TARGET'), data['TARGET']

# Define Pipeline here
# In order to reuse the pipeline with various models, do not add the
# estimator to the pipeline.
# DEBUG:Column transformers!!! 
lgb_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', MinMaxScaler()),
    # ('encoder', LabelEncoder())
])

if __name__ == '__main__':
    X, y = load_data()
    print(lgb_pipeline.fit_transform(X))
