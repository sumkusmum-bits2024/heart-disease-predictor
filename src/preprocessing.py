"""
Data preprocessing pipeline for Heart Disease dataset.

This module provides functions and sklearn pipelines for data cleaning,
feature engineering, and transformation.
"""
import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    RANDOM_SEED,
    TARGET_COLUMN,
    TEST_SIZE,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw dataset by handling missing values and data types.

    Args:
        df: Raw DataFrame to clean.

    Returns:
        Cleaned DataFrame.
    """
    df = df.copy()

    # Log initial missing values
    missing_before = df.isnull().sum().sum()
    logger.info(f"Missing values before cleaning: {missing_before}")

    # Fill missing values for categorical columns with mode, then convert to string
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            if df[col].isnull().any():
                mode_value = df[col].mode()[0]
                df[col] = df[col].fillna(mode_value)
                logger.info(f"Filled {col} missing values with mode: {mode_value}")
            # Convert to string for consistent one-hot encoding
            df[col] = df[col].astype(str)

    missing_after = df.isnull().sum().sum()
    logger.info(f"Missing values after cleaning: {missing_after}")

    return df


def create_preprocessing_pipeline() -> ColumnTransformer:
    """
    Create a sklearn preprocessing pipeline for feature transformation.

    The pipeline handles:
    - Numerical features: Imputation with median, then StandardScaler
    - Categorical features: Imputation with most frequent, then OneHotEncoder

    Returns:
        ColumnTransformer preprocessing pipeline.
    """
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, NUMERICAL_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor


def split_data(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and test sets with stratification.

    Args:
        df: DataFrame containing features and target.
        test_size: Proportion of data for testing.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info(f"Training set size: {len(X_train)}")
    logger.info(f"Test set size: {len(X_test)}")
    logger.info(f"Training target distribution:\n{y_train.value_counts(normalize=True)}")

    return X_train, X_test, y_train, y_test


def prepare_data(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, ColumnTransformer]:
    """
    Full data preparation pipeline: clean, split, and transform.

    Args:
        df: Raw DataFrame.

    Returns:
        Tuple of (X_train_transformed, X_test_transformed, y_train, y_test, preprocessor).
    """
    # Clean data
    df_clean = clean_data(df)

    # Split data
    X_train, X_test, y_train, y_test = split_data(df_clean)

    # Create and fit preprocessor
    preprocessor = create_preprocessing_pipeline()
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    logger.info(f"Transformed training shape: {X_train_transformed.shape}")
    logger.info(f"Transformed test shape: {X_test_transformed.shape}")

    return X_train_transformed, X_test_transformed, y_train, y_test, preprocessor


if __name__ == "__main__":
    from src.data_loader import load_raw_data

    # Test the preprocessing pipeline
    df = load_raw_data()
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(df)

    print(f"\nPreprocessed training data shape: {X_train.shape}")
    print(f"Preprocessed test data shape: {X_test.shape}")
