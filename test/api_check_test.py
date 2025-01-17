"""
Author: Maciej Nowicki
Date: January 2025
Desc: test file for FastAPI
"""

import sys

sys.path.append(".")
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_sanity():
    """Test the root welcome page"""
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"message": "Hiiii World"}


def test_with_fake_data():
    """Test the output for salary is >50k"""

    r = client.post(
        "/prediction",
        json={
            "age": 33,
            "workclass": "Private",
            "fnlgt": 457481,
            "education": "Masters",
            "education-num": 14,
            "marital-status": "Never-married",
            "occupation": "Prof-specialty",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "capital-gain": 140846,
            "capital-loss": 2,
            "hours-per-week": 52,
            "native-country": "United-States",
        },
    )

    assert r.status_code == 200
