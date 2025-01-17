"""
Author: Maciej Nowicki
Date: January 2025
Desc: slice file for machine learning model
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
from ml.data import process_data
from ml.model import compute_model_metrics


def slice_census(data, cat_features):
    """Function to evaluate model on slices of the dataset"""

    train, test = train_test_split(data, test_size=0.20)

    model = joblib.load("model/model.pkl")
    encoder = joblib.load("model/encoder.pkl")
    lb = joblib.load("model/lb.pkl")
    slice_result = []

    for cat in cat_features:
        for cls in test[cat].unique():
            df_temp = test[test[cat] == cls]

            X_test, y_test, _, _ = process_data(
                df_temp,
                categorical_features=cat_features,
                label="salary",
                training=False,
                encoder=encoder,
                lb=lb,
            )

            y_pred = model.predict(X_test)

            precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
            slice_result.append(
                {
                    "feature": cat,
                    "category": cls,
                    "precision": precision,
                    "recall": recall,
                    "Fbeta": fbeta,
                }
            )

    df = pd.DataFrame(slice_result)
    df.to_csv("slice_output.txt", index=False)


if __name__ == "__main__":
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
    data = pd.read_csv("data/census_clean.csv")
    slice_census(data, cat_features)
