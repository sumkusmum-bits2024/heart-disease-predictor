"""Project configuration and constants."""
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model directory
MODELS_DIR = PROJECT_ROOT / "models"

# MLflow
MLFLOW_TRACKING_URI = f"file://{PROJECT_ROOT / 'mlruns'}"
EXPERIMENT_NAME = "heart-disease-classification"

# Dataset
DATASET_FILENAME = "heart.csv"
RAW_DATA_PATH = RAW_DATA_DIR / DATASET_FILENAME
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "heart_processed.csv"

# Model settings
RANDOM_SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Feature definitions
TARGET_COLUMN = "target"

NUMERICAL_FEATURES = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
]

CATEGORICAL_FEATURES = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
