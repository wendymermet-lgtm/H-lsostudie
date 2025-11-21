import pandas as pd

REQUIRED = [
    "age",
    "sex",
    "height",
    "weight",
    "systolic_bp",
    "cholesterol",
    "smoker",
    "disease"]

#age (år), sex (M/F), height (cm), weight (kg), systolic_bp (mmHg), cholesterol (mmol/L), smoker (Yes/No), disease (0/1).


def load_data(file_path: str) -> pd.DataFrame:  
    """
    Reads data from a CSV file and creates a DataFrame.
    """
    df = pd.read_csv(file_path, encoding="utf-8")
    missing = [col for col in REQUIRED if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataframe: {missing}") #Check for required columns
    return df

df = load_data("Data\health_study_dataset.csv")

def check_data(df: pd.DataFrame) -> None:
    """Function to check data quality."""
    print(df.info())  # Check data types
    print("Missing values per column:\n", df.isna().sum())  # Check for missing values
    print("Number of duplicate rows:", df.duplicated().sum())  # Check for duplicates

    print("Statistical summary:\n", df.describe())  # Summary statistics
    print("First 5 rows of the dataframe:\n", df.head())  # Preview first 5 rows


