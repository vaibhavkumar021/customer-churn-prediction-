

import joblib
import pandas as pd

MODEL_PATH = "models/logistic_regression_pipeline.joblib"


def load_model():
    """Load the saved churn prediction pipeline."""
    return joblib.load(MODEL_PATH)


def predict_churn(model, customer_data):
    """Predict whether a customer is likely to churn."""
    customer_df = pd.DataFrame([customer_data])

    prediction = model.predict(customer_df)[0]
    probability = model.predict_proba(customer_df)[0][1]

    churn_status = "Yes" if prediction == 1 else "No"

    return {
        "churn_prediction": churn_status,
        "churn_probability": round(float(probability) * 100, 2)
    }


def get_feature_importance(model):
    """Return features with the strongest positive and negative coefficients."""

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    coefficients = classifier.coef_[0]

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coefficients
    })

    importance_df["Effect"] = importance_df["Coefficient"].apply(
        lambda value: "Higher predicted churn"
        if value > 0
        else "Lower predicted churn"
    )

    importance_df["AbsoluteCoefficient"] = (
        importance_df["Coefficient"].abs()
    )

    return importance_df.sort_values(
        by="AbsoluteCoefficient",
        ascending=False
    )


if __name__ == "__main__":
    # Load the saved model.
    model = load_model()

    # Sample customer details.
    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.0,
        "TotalCharges": 170.0
    }

    # Predict customer churn.
    result = predict_churn(model, sample_customer)

    print("Customer Churn Prediction")
    print("-------------------------")
    print("Predicted churn:", result["churn_prediction"])
    print("Churn probability:", f'{result["churn_probability"]}%')

    # Analyze model coefficients.
    importance_df = get_feature_importance(model)

    print("\nTop 5 Features Associated with Higher or Lower Churn")
    print(importance_df[
        ["Feature", "Coefficient", "Effect"]
    ].head(5).to_string(index=False))

    print("\nTop 5 Features Associated with Higher Predicted Churn")
    print(
        importance_df[importance_df["Coefficient"] > 0]
        .head(5)[["Feature", "Coefficient"]]
        .to_string(index=False)
    )

    print("\nTop 5 Features Associated with Lower Predicted Churn")
    print(
        importance_df[importance_df["Coefficient"] < 0]
        .sort_values(by="Coefficient", ascending=True)
        .head(5)[["Feature", "Coefficient"]]
        .to_string(index=False)
    )
