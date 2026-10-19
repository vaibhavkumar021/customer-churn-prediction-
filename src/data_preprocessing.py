
import pandas as pd

RAW_DATA_PATH = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
PROCESSED_DATA_PATH = "data/processed/cleaned_customer_data.csv"


def load_data():
    """Load the original customer churn dataset."""
    return pd.read_csv(RAW_DATA_PATH)


def clean_data(df):
    """Clean customer churn data without changing the original DataFrame."""
    df = df.copy()

    # Convert TotalCharges to numeric.
    # Blank strings become NaN.
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].astype("string").str.strip(),
        errors="coerce"
    )

    # The blank TotalCharges records have tenure = 0.
    zero_tenure_mask = df["tenure"].eq(0)

    df.loc[
        zero_tenure_mask & df["TotalCharges"].isna(),
        "TotalCharges"
    ] = 0.0

    return df


def save_clean_data(df):
    """Save the cleaned dataset in data/processed/."""
    df.to_csv(PROCESSED_DATA_PATH, index=False)


if __name__ == "__main__":
    df = load_data()

    print("Original shape:", df.shape)
    print("Blank TotalCharges before cleaning:",
          df["TotalCharges"].astype("string").str.strip().eq("").sum())

    cleaned_df = clean_data(df)

    print("\nCleaned shape:", cleaned_df.shape)
    print("TotalCharges data type:", cleaned_df["TotalCharges"].dtype)
    print("Missing values after cleaning:")
    print(cleaned_df.isnull().sum())
    print("TotalCharges values still missing:",
          cleaned_df["TotalCharges"].isna().sum())

    save_clean_data(cleaned_df)
   
# Overall customer churn analysis
churn_counts = cleaned_df["Churn"].value_counts()

total_customers = len(cleaned_df)
churned_customers = (cleaned_df["Churn"] == "Yes").sum()

churn_rate = (churned_customers / total_customers) * 100

print("\n--- Overall Churn Analysis ---")
print("Total customers:", total_customers)
print("Churned customers:", churned_customers)
print(f"Churn rate: {churn_rate:.2f}%")
print(f"Retention rate: {100 - churn_rate:.2f}%")
print(f"Cleaned dataset saved to: {PROCESSED_DATA_PATH}")