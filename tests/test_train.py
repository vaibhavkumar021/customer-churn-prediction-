
import pandas as pd
from src.train import prepare_data


def test_prepare_data_removes_customer_id():
    df = pd.DataFrame({
        "customerID": ["C001", "C002"],
        "tenure": [12, 24],
        "MonthlyCharges": [50.0, 70.0],
        "Churn": ["No", "Yes"]
    })

    X, y = prepare_data(df)

    assert "customerID" not in X.columns
    assert "Churn" not in X.columns
    assert len(X) == 2


def test_prepare_data_encodes_churn():
    df = pd.DataFrame({
        "customerID": ["C001", "C002"],
        "tenure": [12, 24],
        "Churn": ["No", "Yes"]
    })

    X, y = prepare_data(df)

    assert y.tolist() == [0, 1]


from src.train import build_preprocessor


def test_preprocessor_handles_numeric_and_categorical_data():
    X = pd.DataFrame({
        "tenure": [12, 24, 36, 48],
        "Contract": [
            "Month-to-month",
            "One year",
            "Two year",
            "Month-to-month"
        ]
    })

    preprocessor = build_preprocessor(X)
    transformed = preprocessor.fit_transform(X)

    assert transformed.shape[0] == 4
    assert transformed.shape[1] > 2


import joblib
from pathlib import Path


def test_saved_model_predicts_churn():
    model_path = Path("models/logistic_regression_pipeline.joblib")
    model = joblib.load(model_path)

    sample = pd.DataFrame({
        "gender": ["Female"],
        "SeniorCitizen": [0],
        "Partner": ["Yes"],
        "Dependents": ["No"],
        "tenure": [12],
        "PhoneService": ["Yes"],
        "MultipleLines": ["No"],
        "InternetService": ["DSL"],
        "OnlineSecurity": ["No"],
        "OnlineBackup": ["Yes"],
        "DeviceProtection": ["No"],
        "TechSupport": ["No"],
        "StreamingTV": ["No"],
        "StreamingMovies": ["No"],
        "Contract": ["Month-to-month"],
        "PaperlessBilling": ["Yes"],
        "PaymentMethod": ["Electronic check"],
        "MonthlyCharges": [70.0],
        "TotalCharges": [840.0]
    })

    prediction = model.predict(sample)

    assert len(prediction) == 1
    assert prediction[0] in [0, 1]
