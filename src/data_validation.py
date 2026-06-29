import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
TRAINING_DATA_PATH = BASE_DIR / "data" / "zara_training.csv"

def validate_data(df):
    print("--- Performing Data Validation ---")

    # Check for duplicate rows
    duplicate_rows = df.duplicated().sum()
    print(f"Number of duplicate rows: {duplicate_rows}")
    if duplicate_rows > 0:
        print("Warning: Duplicate rows found.")

    # Check for missing values
    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0]
    print(f"Number of columns with missing values: {len(missing_values)}")
    if not missing_values.empty:
        print("Missing values per column:")
        print(missing_values)
        print("Warning: Missing values found in the dataset.")
    else:
        print("No missing values found.")

    print("--- Data Validation Complete ---")
    return duplicate_rows, missing_values

if __name__ == "__main__":
    if not TRAINING_DATA_PATH.exists():
        raise FileNotFoundError(f"Training data not found: {TRAINING_DATA_PATH}")

    df_train = pd.read_csv(TRAINING_DATA_PATH, sep=';')
    validate_data(df_train)
