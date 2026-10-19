
import json
from pathlib import Path

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

RAW_DATA_PATH = Path("data/processed/cleaned_customer_data.csv")
MODEL_PATH = Path("models/logistic_regression_pipeline.joblib")
REPORT_PATH = Path("reports/model_evaluation.json")


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

def load_data():
    """Load the cleaned customer churn dataset."""
    return pd.read_csv(RAW_DATA_PATH)


# --------------------------------------------------
# 3. PREPARE FEATURES AND TARGET
# --------------------------------------------------

def prepare_data(df):
    """Separate input features and target variable."""
    df = df.copy()

    X = df.drop(columns=["customerID", "Churn"])
    y = df["Churn"].map({"No": 0, "Yes": 1})

    if y.isna().any():
        raise ValueError("Unexpected values found in the Churn column.")

    return X, y


# --------------------------------------------------
# 4. BUILD PREPROCESSING PIPELINE
# --------------------------------------------------

def build_preprocessor(X):
    """Create preprocessing steps for numerical and categorical data."""

    numeric_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

    return preprocessor


# --------------------------------------------------
# 5. TRAIN LOGISTIC REGRESSION
# --------------------------------------------------

def train_logistic_regression(X_train, y_train, preprocessor):
    """Train a Logistic Regression model."""
    model_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000))
    ])

    model_pipeline.fit(X_train, y_train)

    return model_pipeline


# --------------------------------------------------
# 6. TRAIN DECISION TREE
# --------------------------------------------------

def train_decision_tree(X_train, y_train, preprocessor):
    """Train a Decision Tree model."""
    model_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", DecisionTreeClassifier(
            max_depth=6,
            random_state=42
        ))
    ])

    model_pipeline.fit(X_train, y_train)

    return model_pipeline


# --------------------------------------------------
# 7. TRAIN RANDOM FOREST
# --------------------------------------------------

def train_random_forest(X_train, y_train, preprocessor):
    """Train a Random Forest model."""
    model_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ))
    ])

    model_pipeline.fit(X_train, y_train)

    return model_pipeline


# --------------------------------------------------
# 8. EVALUATE MODEL
# --------------------------------------------------

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate a model and return its metrics and confusion matrix."""

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    print(f"\n--- {model_name} Evaluation ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["No Churn", "Churn"],
        zero_division=0
    ))

    print("Confusion Matrix:")
    print(cm)

    # Return metrics as percentages for dashboard display.
    return {
        "model": model_name,
        "accuracy": round(float(accuracy) * 100, 2),
        "precision": round(float(precision) * 100, 2),
        "recall": round(float(recall) * 100, 2),
        "f1_score": round(float(f1) * 100, 2),
        "confusion_matrix": cm.tolist()
    }


# --------------------------------------------------
# 9. MAIN TRAINING WORKFLOW
# --------------------------------------------------

if __name__ == "__main__":

    # Load and prepare the data
    df = load_data()
    X, y = prepare_data(df)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Dataset shape:", df.shape)
    print("Features shape:", X.shape)
    print("Target shape:", y.shape)
    print("Training customers:", len(X_train))
    print("Testing customers:", len(X_test))

    print("\nTarget distribution:")
    print(y.value_counts())

    # Store results from all three models
    evaluation_results = []

    # Train and evaluate Logistic Regression
    logistic_model = train_logistic_regression(
        X_train,
        y_train,
        build_preprocessor(X_train)
    )

    print("\nLogistic Regression model trained successfully!")

    evaluation_results.append(
        evaluate_model(
            logistic_model,
            X_test,
            y_test,
            "Logistic Regression"
        )
    )

    # Save the Logistic Regression pipeline
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(logistic_model, MODEL_PATH)

    print("\nLogistic Regression model saved to:", MODEL_PATH)

    # Train and evaluate Decision Tree
    decision_tree_model = train_decision_tree(
        X_train,
        y_train,
        build_preprocessor(X_train)
    )

    print("\nDecision Tree model trained successfully!")

    evaluation_results.append(
        evaluate_model(
            decision_tree_model,
            X_test,
            y_test,
            "Decision Tree"
        )
    )

    # Train and evaluate Random Forest
    random_forest_model = train_random_forest(
        X_train,
        y_train,
        build_preprocessor(X_train)
    )

    print("\nRandom Forest model trained successfully!")

    evaluation_results.append(
        evaluate_model(
            random_forest_model,
            X_test,
            y_test,
            "Random Forest"
        )
    )

    # Save evaluation results to JSON
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(evaluation_results, file, indent=4)

    print("\nEvaluation results saved to:", REPORT_PATH)
    print("All three models evaluated successfully!")
