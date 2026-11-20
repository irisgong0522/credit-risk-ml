import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_data(file_path, random_state=42):
    """
    Load data from a CSV file and split it into training and testing sets.

    Parameters:
    - file_path: str, path to the CSV file.
    - random_state: int, random seed for reproducibility.
    """
    # Load the dataset
    df = pd.read_csv(file_path)

    # Separate features and target
    X = df.drop(columns=["TARGET", "SK_ID_CURR"])
    y = df["TARGET"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_temp = train_test_split(
        X, y, 
        test_size = 0.4, 
        stratify = y,
        random_state = random_state)

    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_temp,
        test_size = 0.5,
        stratify = y_temp,
        random_state = random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test