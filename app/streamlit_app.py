
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import json

from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "logistic_regression_pipeline.joblib"
DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_customer_data.csv"
REPORT_PATH = BASE_DIR / "reports" / "model_evaluation.json"

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# --------------------------------------------------
# 2. LOAD MODEL AND DATA
# --------------------------------------------------

@st.cache_resource
def load_model():
    """Load the trained Logistic Regression pipeline."""
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_analytics_data():
    """Load the cleaned customer dataset."""
    return pd.read_csv(DATA_PATH)

@st.cache_data
def load_evaluation_results():
    """Load saved model evaluation results."""
    with REPORT_PATH.open("r", encoding="utf-8") as file:
        return pd.DataFrame(json.load(file))


# --------------------------------------------------
# 3. DASHBOARD HEADER
# --------------------------------------------------

st.title("Customer Churn Prediction & Analytics")

st.write(
    "Analyze customer churn, predict customer retention, "
    "compare machine learning models, and explore important features."
)

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Customers", "7,043")

with col2:
    st.metric("Churn Rate", "26.54%")

with col3:
    st.metric("Retention Rate", "73.46%")


# --------------------------------------------------
# 4. NAVIGATION TABS
# --------------------------------------------------

prediction_tab, analytics_tab, comparison_tab, importance_tab = (
    st.tabs([
        "🔮 Prediction",
        "📊 Analytics",
        "🤖 Model Comparison",
        "🎯 Feature Importance"
    ])
)


# ==================================================
# TAB 1: CUSTOMER CHURN PREDICTION
# ==================================================

with prediction_tab:

    st.header("Customer Churn Prediction")

    st.write(
        "Enter customer details to estimate the probability "
        "that the customer will leave the service."
    )

    with st.form("customer_form"):

        st.subheader("Customer Information")

        col1, col2 = st.columns(2)

        with col1:

            gender = st.selectbox(
                "Gender", ["Female", "Male"]
            )

            senior_citizen = st.selectbox(
                "Senior Citizen",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

            partner = st.selectbox(
                "Has Partner?", ["Yes", "No"]
            )

            dependents = st.selectbox(
                "Has Dependents?", ["Yes", "No"]
            )

            tenure = st.number_input(
                "Tenure (months)",
                min_value=0,
                max_value=72,
                value=12
            )

            phone_service = st.selectbox(
                "Phone Service", ["Yes", "No"]
            )

            multiple_lines = st.selectbox(
                "Multiple Lines",
                ["No", "Yes", "No phone service"]
            )

            internet_service = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"]
            )

            online_security = st.selectbox(
                "Online Security",
                ["Yes", "No", "No internet service"]
            )

            online_backup = st.selectbox(
                "Online Backup",
                ["Yes", "No", "No internet service"]
            )

        with col2:

            device_protection = st.selectbox(
                "Device Protection",
                ["Yes", "No", "No internet service"]
            )

            tech_support = st.selectbox(
                "Tech Support",
                ["Yes", "No", "No internet service"]
            )

            streaming_tv = st.selectbox(
                "Streaming TV",
                ["Yes", "No", "No internet service"]
            )

            streaming_movies = st.selectbox(
                "Streaming Movies",
                ["Yes", "No", "No internet service"]
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

            st.subheader("Prediction Results")

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
                "Trained model not found. Run "
                "'python src/train.py' from the project root first."
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")


# ==================================================
# TAB 2: CUSTOMER CHURN ANALYTICS
# ==================================================

with analytics_tab:

    st.header("Customer Churn Analytics")

    try:
        analytics_df = load_analytics_data()

        # Churn distribution
        st.subheader("1. Customer Churn Distribution")

        churn_counts = (
            analytics_df["Churn"]
            .value_counts()
            .reindex(["No", "Yes"], fill_value=0)
        )

        col1, col2 = st.columns(2)

        with col1:
            st.write("Number of Customers")
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

        # Churn by contract
        st.subheader("2. Churn Rate by Contract Type")

        contract_churn = (
            analytics_df.groupby("Contract")["Churn"]
            .apply(lambda values: (values == "Yes").mean() * 100)
            .sort_values(ascending=False)
        )

        st.bar_chart(contract_churn.round(2))

        st.caption(
            "Percentage of customers who churned within each contract type."
        )

        # Churn by tenure
        st.subheader("3. Churn Rate by Customer Tenure")

        tenure_df = analytics_df.copy()

        tenure_df["Tenure Group"] = pd.cut(
            tenure_df["tenure"],
            bins=[-1, 12, 24, 48, 72],
            labels=[
                "0–12 months",
                "13–24 months",
                "25–48 months",
                "49–72 months"
            ]
        )

        tenure_churn = (
            tenure_df.groupby("Tenure Group", observed=False)["Churn"]
            .apply(lambda values: (values == "Yes").mean() * 100)
        )

        st.bar_chart(tenure_churn.round(2))

        st.caption(
            "Percentage of customers who churned in each tenure group."
        )

    except FileNotFoundError:
        st.error(
            "Cleaned dataset not found. Check "
            "data/processed/cleaned_customer_data.csv."
        )

    except Exception as error:
        st.error(f"Could not load analytics: {error}")


# ==================================================
# TAB 3: MODEL COMPARISON AND CONFUSION MATRIX
# ==================================================

with comparison_tab:

    st.header("Model Performance Comparison")

    st.write(
        "Compare the baseline performance of the three trained "
        "machine learning models."
    )

    
    try:
        model_results = load_evaluation_results()

        display_results = model_results[
            ["model", "accuracy", "precision", "recall", "f1_score"]
        ].rename(columns={
            "model": "Model",
            "accuracy": "Accuracy (%)",
            "precision": "Precision (%)",
            "recall": "Recall (%)",
            "f1_score": "F1-Score (%)"
        })

        st.dataframe(
            display_results,
            hide_index=True,
            use_container_width=True
        )

        st.subheader("Compare Model Metrics")

        metric_columns = {
            "Accuracy": "accuracy",
            "Precision": "precision",
            "Recall": "recall",
            "F1-Score": "f1_score"
        }

        selected_metrics = st.multiselect(
            "Choose metrics to compare",
            list(metric_columns.keys()),
            default=["Accuracy", "F1-Score"],
            key="comparison_metrics"
        )

        if selected_metrics:
            chart_data = model_results.set_index("model")[
                [metric_columns[metric] for metric in selected_metrics]
            ].rename(columns={
                metric_columns[metric]: metric
                for metric in selected_metrics
            })

            st.bar_chart(chart_data)

        st.caption(
            "Metrics are percentages calculated during model training "
            "on the held-out test dataset."
        )

    except FileNotFoundError:
        st.error(
            "Evaluation results not found. Run "
            "'python src/train.py' first."
        )

    except Exception as error:
        st.error(f"Could not load model evaluation results: {error}")


    # Confusion matrix
    st.divider()
    st.subheader("Logistic Regression Confusion Matrix")

    st.write(
        "Rows show actual outcomes; columns show predicted outcomes."
    )

    try:
        model_data = load_analytics_data()

        X = model_data.drop(columns=["customerID", "Churn"])
        y = model_data["Churn"].map({"No": 0, "Yes": 1})

        _, X_test, _, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        model = load_model()
        y_pred = model.predict(X_test)

        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1]
        )

        fig, ax = plt.subplots(figsize=(5, 3.5))

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

        ax.set_title("Logistic Regression Confusion Matrix",fontsize=11)
        ax.set_xlabel("Predicted Outcome",fontsize=9)

        ax.set_ylabel("Actual Outcome",fontsize=9)  
        ax.tick_params(axis="both", labelsize=8)

        fig.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

        

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


# ==================================================
# TAB 4: FEATURE IMPORTANCE
# ==================================================

with importance_tab:

    st.header("Feature Importance")

    st.write(
        "Explore which customer characteristics are associated "
        "with higher or lower predicted churn in the Logistic "
        "Regression model."
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

        top_features = importance_df.head(10).sort_values(
            "Coefficient"
        )

        st.subheader("Top 10 Features by Coefficient Magnitude")

        chart_data = top_features.set_index("Feature")[["Coefficient"]]
        st.bar_chart(chart_data)

        st.caption(
            "Positive coefficients are associated with higher predicted "
            "churn, while negative coefficients are associated with lower "
            "predicted churn. Numeric features were scaled during training. "
            "These associations do not establish causation."
        )

        with st.expander("View All Feature Details"):

            st.dataframe(
                importance_df[
                    ["Feature", "Coefficient", "Effect"]
                ].round({"Coefficient": 4}),
                hide_index=True,
                use_container_width=True
            )

    except Exception as error:
        st.error(f"Could not display feature importance: {error}")
