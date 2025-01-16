from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    pass


def performance_on_slices(model, data, cat_features, label):
    """
    Outputs the performance of the model on slices of the data.

    Inputs
    ------
    model : Trained machine learning model.
    data : pd.DataFrame
        Dataframe containing the features and label.
    cat_features: list[str]
        List containing the names of the categorical features.
    label : str
        Name of the label column in `data`.

    Returns
    -------
    performance : dict
        Dictionary containing the performance metrics for each slice.
    """
    performance = {}
    for feature in cat_features:
        for category in data[feature].unique():
            slice_data = data[data[feature] == category]
            X_slice, y_slice, _, _ = process_data(
                slice_data,
                categorical_features=cat_features,
                label=label,
                training=False,
            )
            preds = inference(model, X_slice)
            precision, recall, fbeta = compute_model_metrics(y_slice, preds)
            performance[f"{feature}_{category}"] = {
                "precision": precision,
                "recall": recall,
                "fbeta": fbeta,
            }
    return performance
