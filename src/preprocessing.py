"""
Data preprocessing functions for the Customer Churn Prediction project.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw customer churn dataset.

    Steps performed:
    ---------------
    1. Remove customerID column.
    2. Replace blank values in TotalCharges with NaN.
    3. Convert TotalCharges to numeric.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset.
    """

    df = df.copy()

    # Remove identifier column
    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)

    # Replace blank strings
    df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan)

    # Convert to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows containing missing values.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """

    df = df.copy()

    df.dropna(inplace=True)

    return df

def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode the target variable (Churn).

    No -> 0
    Yes -> 1
    """

    df = df.copy()

    encoder = LabelEncoder()

    df["Churn"] = encoder.fit_transform(df["Churn"])

    return df

def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Split dataset into training and testing sets.

    Parameters
    ----------
    df : pd.DataFrame
        Clean dataset.

    test_size : float
        Percentage of testing data.

    random_state : int
        Random seed.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


def create_preprocessor():
    """
    Create a preprocessing pipeline.

    Returns
    -------
    ColumnTransformer
        Preprocessing pipeline.
    """

    numeric_features = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    categorical_features = [
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod"
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_transformer,
                numeric_features
            ),
            (
                "cat",
                categorical_transformer,
                categorical_features
            )
        ],
        remainder="passthrough"
    )

    return preprocessor


def apply_smote(
    X_train,
    y_train,
    random_state: int = 42
):
    """
    Apply SMOTE to the training dataset.

    Parameters
    ----------
    X_train : array-like
        Training features.

    y_train : array-like
        Training labels.

    random_state : int
        Random seed.

    Returns
    -------
    X_train_smote, y_train_smote
    """

    smote = SMOTE(random_state=random_state)

    X_train_smote, y_train_smote = smote.fit_resample(
        X_train,
        y_train
    )

    return X_train_smote, y_train_smote