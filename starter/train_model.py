"""

This script reads cleaned census data, preprocesses it, trains a machine learning model, evaluates the model, and saves the trained model and encoders.

Functions:
- process_data: Preprocesses the data by encoding categorical features and scaling numerical features.
- train_model: Trains a machine learning model using the provided training data.
- compute_model_metrics: Computes precision, recall, and F-beta score for the model predictions.
- inference: Generates predictions using the trained model.

Workflow:
1. Reads cleaned census data from a CSV file.
2. Splits the data into training and testing sets.
3. Defines categorical features for preprocessing.
4. Preprocesses the training and testing data.
5. Trains a machine learning model using the training data.
6. Evaluates the model using the testing data.
7. Logs the model performance metrics.
8. Saves the trained model and encoders to disk.
Author: Maciej Nowicki
Date: January 2025
Desc: model training file for machine learning model
"""
# Script to train machine learning model.

import pandas as pd
import logging
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference
from sklearn.model_selection import train_test_split
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/training.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Read and preprocess data
logger.info("Reading datathat were celanesd")
df = pd.read_csv("data/cleaned_census.csv")

# Split data into training and testing sets
train, test = train_test_split(df, test_size=0.3)

# Define categorical features
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

# Process training data
logger.info("Preprocessing training data")
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Process testing data
logger.info("Preprocessing testing data")
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# Train the model
logger.info("Training the model")
model = train_model(X_train, y_train)

# Evaluate the model
logger.info("Evaluating the model")
preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)
logger.info(
    f"Model metrics - Precision: {precision:.2f}, Recall: {recall:.2f}, Fbeta: {fbeta:.2f}"
)

# Save the model and encoders
logger.info("Saving the model and encoders")
joblib.dump(model, "model/model.pkl")
joblib.dump(encoder, "model/encoder.pkl")
joblib.dump(lb, "model/lb.pkl")
