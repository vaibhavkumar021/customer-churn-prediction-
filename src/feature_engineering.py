
import pandas as pd

PROCESSED_DATA_PATH = "data/processed/cleaned_customer_data.csv"
FEATURES_OUTPUT_PATH = "data/processed/features.csv"



def load_data():
    """Load the cleaned customer churn dataset."""
    return pd.read_csv(PROCESSED_DATA_PATH)


def separate_features_target(df):
    """Separate input features from the target column."""
    df = df.copy()

    # Remove customer ID because it is only an identifier.
    df = df.drop(columns=["customerID"])

    # Separate the target variable.
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    return X, y


def encode_features(X):
    """Convert categorical features into numerical columns."""
    X_encoded = pd.get_dummies(
        X,
        columns=X.select_dtypes(include=["object", "string"]).columns,
        drop_first=True,
        dtype=int
    )

    return X_encoded


def save_features(X_encoded, y):
    """Save encoded features and target labels."""
    X_encoded.to_csv(FEATURES_OUTPUT_PATH, index=False)

    target_output_path = "data/processed/target.csv"
    y.to_frame(name="Churn").to_csv(target_output_path, index=False)

    print("\nEncoded features saved to:", FEATURES_OUTPUT_PATH)
    print("Target labels saved to:", target_output_path)





if __name__ == "__main__":
    df = load_data()
    X, y = separate_features_target(df)
    print("\nFeatures shape:", X.shape)
    print("Target shape:", y.shape)

    print("\nFeatures columns:")
    print(X.columns.tolist())

    print("\nTarget distribution:")
    print(y.value_counts())

    X_encoded = encode_features(X)
    save_features(X_encoded, y)
    
    print("\nEncoded features shape:", X_encoded.shape)

    print("\nEncoded feature columns:")
    print(X_encoded.columns.tolist())

    print("\nFirst 5 encoded rows:")
    print(X_encoded.head())

    print("\nData types after encoding:")
    print(X_encoded.dtypes)


    print("Dataset shape:", df.shape)
    print("\nDataset columns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())
