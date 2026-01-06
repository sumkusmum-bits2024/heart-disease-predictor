"""
Model inference module for Heart Disease prediction.

Provides functions to load the trained model and make predictions.
"""
import json
import logging
from typing import Any, Dict, List, Union

import joblib
import pandas as pd

from src.config import ALL_FEATURES, MODELS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache for loaded model and preprocessor
_model_cache: Dict[str, Any] = {}


def load_model() -> Any:
    """
    Load the trained model from disk.

    Returns:
        Trained sklearn model.

    Raises:
        FileNotFoundError: If model file doesn't exist.
    """
    if "model" not in _model_cache:
        model_path = MODELS_DIR / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        _model_cache["model"] = joblib.load(model_path)
        logger.info("Model loaded successfully")
    return _model_cache["model"]


def load_preprocessor() -> Any:
    """
    Load the preprocessor from disk.

    Returns:
        Fitted sklearn preprocessor.

    Raises:
        FileNotFoundError: If preprocessor file doesn't exist.
    """
    if "preprocessor" not in _model_cache:
        preprocessor_path = MODELS_DIR / "preprocessor.joblib"
        if not preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}")
        _model_cache["preprocessor"] = joblib.load(preprocessor_path)
        logger.info("Preprocessor loaded successfully")
    return _model_cache["preprocessor"]


def load_metadata() -> Dict[str, Any]:
    """
    Load model metadata.

    Returns:
        Dictionary containing model metadata.
    """
    if "metadata" not in _model_cache:
        metadata_path = MODELS_DIR / "model_metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                _model_cache["metadata"] = json.load(f)
        else:
            _model_cache["metadata"] = {}
    return _model_cache["metadata"]


def predict(data: Union[Dict, pd.DataFrame]) -> Dict[str, Any]:
    """
    Make a prediction for the given input data.

    Args:
        data: Dictionary or DataFrame containing feature values.

    Returns:
        Dictionary with prediction, probability, and risk level.
    """
    # Load model and preprocessor
    model = load_model()
    preprocessor = load_preprocessor()

    # Convert to DataFrame if dict
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = data.copy()

    # Ensure all features are present
    for feature in ALL_FEATURES:
        if feature not in df.columns:
            raise ValueError(f"Missing feature: {feature}")

    # Import CATEGORICAL_FEATURES for type conversion
    from src.config import CATEGORICAL_FEATURES

    # Convert categorical features to match training data format
    # Training data had floats converted to strings (e.g., "1.0", "2.0")
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(float).astype(str)

    # Preprocess
    X = preprocessor.transform(df[ALL_FEATURES])

    # Predict
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0, 1]

    # Determine risk level
    if probability < 0.3:
        risk_level = "Low"
    elif probability < 0.6:
        risk_level = "Moderate"
    else:
        risk_level = "High"

    return {
        "prediction": int(prediction),
        "probability": float(probability),
        "risk_level": risk_level,
    }


def predict_batch(data: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Make predictions for multiple samples.

    Args:
        data: DataFrame containing feature values for multiple samples.

    Returns:
        List of prediction dictionaries.
    """
    # Load model and preprocessor
    model = load_model()
    preprocessor = load_preprocessor()

    # Import CATEGORICAL_FEATURES for type conversion
    from src.config import CATEGORICAL_FEATURES

    # Convert categorical features to match training data format
    # Training data had floats converted to strings (e.g., "1.0", "2.0")
    data = data.copy()
    for col in CATEGORICAL_FEATURES:
        data[col] = data[col].astype(float).astype(str)

    # Preprocess
    X = preprocessor.transform(data[ALL_FEATURES])

    # Predict
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    # Build results
    results = []
    for pred, prob in zip(predictions, probabilities, strict=False):
        if prob < 0.3:
            risk_level = "Low"
        elif prob < 0.6:
            risk_level = "Moderate"
        else:
            risk_level = "High"

        results.append(
            {
                "prediction": int(pred),
                "probability": float(prob),
                "risk_level": risk_level,
            }
        )

    return results


if __name__ == "__main__":
    # Test prediction
    sample_input = {
        "age": 63,
        "sex": 1,
        "cp": 4,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 2,
        "ca": 0,
        "thal": 6,
    }

    result = predict(sample_input)
    print(f"Prediction: {result['prediction']}")
    print(f"Probability: {result['probability']:.4f}")
    print(f"Risk Level: {result['risk_level']}")
