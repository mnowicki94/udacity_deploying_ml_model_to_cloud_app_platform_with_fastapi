"""
Author: Maciej Nowicki
Date: January 2025
Desc: test file for model
"""

import sys

sys.path.append(".")
import pandas as pd
import pytest
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from starter.ml.data import process_data

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


def model_test():
    """Test Random Forest model"""

    model = joblib.load("model/model.pkl")
    assert isinstance(model, RandomForestClassifier)


def data_test():
    """Test data csv"""

    data = pd.read_csv("data/cleaned_census.csv")
    assert data.shape[0] > 0
