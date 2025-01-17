"""
Author: Maciej Nowicki
Date: January 2025
Desc: Main file for FastAPI
"""

import pandas as pd
import joblib
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from starter.ml.data import process_data
from starter.ml.model import inference

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


# load model
model = joblib.load("model/model.pkl")
encoder = joblib.load("model/encoder.pkl")
lb = joblib.load("model/lb.pkl")

# create app
app = FastAPI()


@app.get("/")
async def hello():
    return {"message": "Hiiii World"}


@app.post("/prediction")
async def predict(input_data: TaggedItem):
    data = pd.DataFrame(
        {k: v for k, v in input_data.dict(by_alias=True).items()}, index=[0]
    )
    X, _, _, _ = process_data(
        data,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb,
    )
    y_pred = inference(model, X)

    return {"Predicted Income": lb.inverse_transform(y_pred)[0]}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
