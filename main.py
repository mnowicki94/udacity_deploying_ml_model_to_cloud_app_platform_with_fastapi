"""
Author: Maciej Nowicki
Date: January 2025
Description: Main file for FastAPI application
"""

import pandas as pd
import joblib
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from starter.ml.data import process_data
from starter.ml.model import inference

# Define categorical features
categorical_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


# Define the data model for input
class InputData(BaseModel):
    # Define the input data model
    age: int = Field(example=23)
    workclass: str = Field(example="State-gov")
    fnlgt: int = Field(example=55789)
    education: str = Field(example="Bachelors")
    education_num: int = Field(alias="education-num", example=11)
    marital_status: str = Field(alias="marital-status", example="Never-married")
    occupation: str = Field(example="Adm-clerical")
    relationship: str = Field(example="Not-in-family")
    race: str = Field(example="White")
    sex: str = Field(example="Female")
    capital_gain: int = Field(alias="capital-gain", example=2474)
    capital_loss: int = Field(alias="capital-loss", example=2)
    hours_per_week: int = Field(alias="hours-per-week", example=45)
    native_country: str = Field(alias="native-country", example="United-States")
