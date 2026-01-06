"""
Data loading and download utilities for Heart Disease dataset.

This module provides functions to download the Heart Disease UCI dataset
and load it into pandas DataFrames.
"""
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from src.config import RAW_DATA_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# UCI Heart Disease dataset URL (Cleveland dataset)
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

# Column names for the dataset
COLUMN_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]


def download_dataset(url: str = DATASET_URL, save_path: Optional[Path] = None) -> Path:
    """
    Download the Heart Disease UCI dataset from the source.

    Args:
        url: URL to download the dataset from.
        save_path: Path to save the downloaded file. Defaults to RAW_DATA_PATH.

    Returns:
        Path to the saved dataset file.

    Raises:
        requests.RequestException: If download fails.
    """
    if save_path is None:
        save_path = RAW_DATA_PATH

    # Create directory if it doesn't exist
    save_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading dataset from {url}")
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    # Parse the CSV data (no headers in original file)
    lines = response.text.strip().split("\n")

    # Create proper CSV with headers
    csv_content = ",".join(COLUMN_NAMES) + "\n"
    csv_content += "\n".join(lines)

    with open(save_path, "w") as f:
        f.write(csv_content)

    logger.info(f"Dataset saved to {save_path}")
    return save_path


def load_raw_data(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the raw Heart Disease dataset.

    If the dataset doesn't exist, it will be downloaded first.

    Args:
        path: Path to the dataset. Defaults to RAW_DATA_PATH.

    Returns:
        pandas DataFrame containing the raw dataset.
    """
    if path is None:
        path = RAW_DATA_PATH

    if not path.exists():
        logger.info("Dataset not found. Downloading...")
        download_dataset(save_path=path)

    logger.info(f"Loading dataset from {path}")
    df = pd.read_csv(path, na_values=["?"])

    # Convert target to binary (0 = no disease, 1 = disease)
    # Original dataset has values 0-4, where 0 = no disease
    df["target"] = (df["target"] > 0).astype(int)

    logger.info(f"Loaded {len(df)} samples with {len(df.columns)} features")
    return df


if __name__ == "__main__":
    # Download and verify dataset
    df = load_raw_data()
    print(f"\nDataset Shape: {df.shape}")
    print(f"\nColumn Types:\n{df.dtypes}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    print(f"\nTarget Distribution:\n{df['target'].value_counts()}")
