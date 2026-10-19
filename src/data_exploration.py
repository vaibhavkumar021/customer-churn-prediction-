
import pandas as pd

# Load the customer churn dataset
file_path = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"

df = pd.read_csv(file_path)

# Display the first 5 rows
print("First 5 rows:")
print(df.head())

# Display the number of rows and columns
print("\nDataset shape:")
print(df.shape)

# Display column names
print("\nColumn names:")
print(df.columns.tolist())



# 1. Check dataset information
print("\nDataset Information:")
df.info()

# 2. Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# 3. Check duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# 4. Check target column distribution
print("\nChurn Distribution:")
print(df["Churn"].value_counts())

# 5. Check numerical statistics
print("\nNumerical Statistics:")
print(df.describe())


# 1. Inspect dataset information
print("\n--- Dataset Information ---")
df.info()

# 2. Check missing values
print("\n--- Missing Values ---")
print(df.isnull().sum())

# 3. Check duplicate rows
print("\n--- Duplicate Rows ---")
print(df.duplicated().sum())

# 4. Check churn distribution
print("\n--- Churn Distribution ---")
print(df["Churn"].value_counts())

# 5. Check numerical statistics
print("\n--- Numerical Statistics ---")
print(df.describe())




# Investigate TotalCharges
print("\n--- TotalCharges Investigation ---")

print("Data type:", df["TotalCharges"].dtype)

# Count blank or whitespace-only values
blank_charges = (
    df["TotalCharges"]
    .astype("string")
    .str.strip()
    .eq("")
    .sum()
)

print("Blank TotalCharges values:", blank_charges)

# Display examples of blank values
print("\nRows with blank TotalCharges:")
print(
    df.loc[
        df["TotalCharges"].astype("string").str.strip().eq(""),
        ["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]
    ].head(15)
)
