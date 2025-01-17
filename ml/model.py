"""
Author: Maciej Nowicki
Date: January 2025
Desc: mdoel file for machine learning model
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Parameters
    ----------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model : RandomForestClassifier
        Trained machine learning model.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y_true, y_pred):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Parameters
    ----------
    y_true : np.array
        Known labels, binarized.
    y_pred : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    precision = precision_score(y_true, y_pred, zero_division=1)
    recall = recall_score(y_true, y_pred, zero_division=1)
    fbeta = fbeta_score(y_true, y_pred, beta=1, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """
    Run model inferences and return the predictions.

    Parameters
    ----------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    if model is None:
        raise ValueError("Model is not provided")
    preds = model.predict(X)
    return preds
