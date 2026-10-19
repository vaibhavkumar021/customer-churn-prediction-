
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "logistic_regression_pipeline.joblib"

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.title("Customer Churn Prediction & Analytics")
st.write(
    "Enter customer details to predict churn probability "
    "using our trained Machine Learning model."
)

st.divider()

# Dataset overview
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Customers", "7,043")

with col2:
    st.metric("Churn Rate", "26.54%")

with col3:
    st.metric("Retention Rate", "73.46%")

st.divider()
st.header("Customer Churn Prediction")

# Customer input form
with st.form("customer_form"):

    st.subheader("Customer Information")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox(
            "Gender", ["Female", "Male"]
        )
        senior_citizen = st.selectbox(
            "Senior Citizen", [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )
        partner = st.selectbox("Has Partner?", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
        tenure = st.number_input(
            "Tenure (months)", min_value=0, max_value=72, value=12
        )
        phone_service = st.selectbox(
            "Phone Service", ["Yes", "No"]
        )
        multiple_lines = st.selectbox(
            "Multiple Lines", ["No", "Yes", "No phone service"]
        )
        internet_service = st.selectbox(
            "Internet Service", ["DSL", "Fiber optic", "No"]
        )
        online_security = st.selectbox(
            "Online Security", ["Yes", "No", "No internet service"]
        )
        online_backup = st.selectbox(
            "Online Backup", ["Yes", "No", "No internet service"]
        )

    with col2:
        device_protection = st.selectbox(
            "Device Protection", ["Yes", "No", "No internet service"]
        )
        tech_support = st.selectbox(
            "Tech Support", ["Yes", "No", "No internet service"]
        )
        streaming_tv = st.selectbox(
            "Streaming TV", ["Yes", "No", "No internet service"]
        )
        streaming_movies = st.selectbox(
            "Streaming Movies", ["Yes", "No", "No internet service"]
        )
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )
        paperless_billing = st.selectbox(
            "Paperless Billing", ["Yes", "No"]
        )
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )
        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            max_value=200.0,
            value=70.0
        )
        total_charges = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            value=840.0
        )

    submitted = st.form_submit_button(
        "Predict Customer Churn",
        type="primary",
        use_container_width=True
    )


# Make prediction
if submitted:
    customer_data = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    try:
        model = load_model()
        customer_df = pd.DataFrame([customer_data])

        prediction = model.predict(customer_df)[0]
        probability = model.predict_proba(customer_df)[0][1] * 100

        st.divider()
        st.header("Prediction Results")

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            if prediction == 1:
                st.error("Prediction: Customer likely to churn")
            else:
                st.success("Prediction: Customer likely to stay")

        with result_col2:
            st.metric(
                "Predicted Churn Probability",
                f"{probability:.2f}%"
            )

        st.progress(float(probability / 100))

        if prediction == 1:
            st.warning(
                "Consider reviewing this customer's contract, "
                "support needs, and service experience."
            )
        else:
            st.info(
                "The model predicts a lower likelihood of churn "
                "for this customer."
            )

    except FileNotFoundError:
        st.error(
            "Trained model not found. Please run "
            "'python src/train.py' from the project root first."
        )
    except Exception as error:
        st.error(f"Prediction failed: {error}")




# Model Comparison Section
st.divider()
st.header("Model Performance Comparison")

st.write(
    "Compare the performance of the three machine learning "
    "models trained on the customer churn dataset."
)

model_results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],
    "Accuracy": [80.55, 79.13, 79.99],
    "Precision": [65.72, 63.42, 65.75],
    "Recall": [55.88, 50.53, 51.34],
    "F1-Score": [60.40, 56.25, 57.66]
})

st.dataframe(
    model_results,
    hide_index=True,
    use_container_width=True
)

st.subheader("Compare Metrics Visually")

selected_metrics = st.multiselect(
    "Choose metrics to compare",
    ["Accuracy", "Precision", "Recall", "F1-Score"],
    default=["Accuracy", "F1-Score"]
)

if selected_metrics:
    chart_data = model_results.set_index("Model")[selected_metrics]
    st.bar_chart(chart_data)

st.caption(
    "Metrics are percentages from the existing baseline evaluation "
    "on the held-out test dataset."
)




# Customer Churn Analytics
st.divider()
st.header("Customer Churn Analytics")

# Load the cleaned dataset
DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_customer_data.csv"


@st.cache_data
def load_analytics_data():
    return pd.read_csv(DATA_PATH)


try:
    analytics_df = load_analytics_data()

    # Chart 1: Churn distribution
    st.subheader("1. Customer Churn Distribution")

    churn_counts = (
        analytics_df["Churn"]
        .value_counts()
        .reindex(["No", "Yes"], fill_value=0)
    )

    col1, col2 = st.columns(2)

    with col1:
        st.bar_chart(churn_counts)

    with col2:
        churn_percentages = (
            analytics_df["Churn"]
            .value_counts(normalize=True)
            .mul(100)
            .reindex(["No", "Yes"], fill_value=0)
        )
        st.write("Customer Distribution (%)")
        st.bar_chart(churn_percentages.round(2))

    # Chart 2: Churn by contract
    st.subheader("2. Churn Rate by Contract Type")

    contract_churn = (
        analytics_df.groupby("Contract", observed=False)["Churn"]
        .apply(lambda values: (values == "Yes").mean() * 100)
        .sort_values(ascending=False)
    )

    st.bar_chart(contract_churn.round(2))
    st.caption("Percentage of customers who churned within each contract type.")

    # Chart 3: Churn by tenure
    st.subheader("3. Churn Rate by Customer Tenure")

    analytics_df["Tenure Group"] = pd.cut(
        analytics_df["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=[
            "0–12 months",
            "13–24 months",
            "25–48 months",
            "49–72 months"
        ]
    )

    tenure_churn = (
        analytics_df.groupby("Tenure Group", observed=False)["Churn"]
        .apply(lambda values: (values == "Yes").mean() * 100)
    )

    st.bar_chart(tenure_churn.round(2))
    st.caption("Percentage of customers who churned in each tenure group.")

except FileNotFoundError:
    st.error(
        "Cleaned dataset not found. Please check "
        "data/processed/cleaned_customer_data.csv."
    )


# Confusion Matrix
st.divider()
st.header("Confusion Matrix")

st.write(
    "This matrix shows how accurately the Logistic Regression "
    "model predicts customer churn on unseen test data."
)

try:
    from sklearn.model_selection import train_test_split

    # Load the same cleaned dataset used for training
    model_data = pd.read_csv(DATA_PATH)

    X = model_data.drop(columns=["customerID", "Churn"])
    y = model_data["Churn"].map({"No": 0, "Yes": 1})

    # Recreate the same test split used in src/train.py
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Use the saved, trained model
    model = load_model()
    y_pred = model.predict(X_test)

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    fig, ax = plt.subplots(figsize=(6, 4))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Stayed", "Churned"]
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=False
    )

    ax.set_title("Logistic Regression Confusion Matrix")
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Rows represent actual outcomes; columns represent "
        "predicted outcomes."
    )

    tn, fp, fn, tp = cm.ravel()

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Correctly Predicted Stay", int(tn))
        st.metric("Incorrectly Predicted Churn", int(fp))

    with col2:
        st.metric("Missed Churn Cases", int(fn))
        st.metric("Correctly Predicted Churn", int(tp))

except Exception as error:
    st.error(f"Could not display confusion matrix: {error}")



# Feature Importance Visualization
st.divider()
st.header("Feature Importance")

st.write(
    "Explore which customer characteristics are associated "
    "with higher or lower predicted churn according to "
    "the trained Logistic Regression model."
)

try:
    model = load_model()

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    coefficients = classifier.coef_[0]

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coefficients
    })

    importance_df["Effect"] = importance_df["Coefficient"].apply(
        lambda value: (
            "Higher predicted churn"
            if value > 0
            else "Lower predicted churn"
            if value < 0
            else "Neutral"
        )
    )

    importance_df["Absolute Coefficient"] = (
        importance_df["Coefficient"].abs()
    )

    importance_df = importance_df.sort_values(
        "Absolute Coefficient",
        ascending=False
    )

    # Select the strongest coefficients
    top_features = importance_df.head(10).sort_values(
        "Coefficient"
    )

    st.subheader("Top 10 Features by Coefficient Magnitude")

    chart_data = top_features.set_index("Feature")[["Coefficient"]]
    st.bar_chart(chart_data)

    st.caption(
        "Positive coefficients are associated with higher predicted "
        "churn; negative coefficients are associated with lower "
        "predicted churn. Numeric features were scaled during training."
    )

    with st.expander("View Feature Details"):
        st.dataframe(
            importance_df[
                ["Feature", "Coefficient", "Effect"]
            ].round({"Coefficient": 4}),
            hide_index=True,
            use_container_width=True
        )

except Exception as error:
    st.error(f"Could not display feature importance: {error}")
