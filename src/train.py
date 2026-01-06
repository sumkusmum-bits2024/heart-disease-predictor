"""
Model training script with MLflow experiment tracking.

This script trains multiple classification models on the Heart Disease dataset
and logs all parameters, metrics, and artifacts to MLflow.
"""
import json
import logging
from typing import Any, Dict, Tuple

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

from src.config import (
    CV_FOLDS,
    EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    RANDOM_SEED,
)
from src.data_loader import load_raw_data
from src.preprocessing import prepare_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_models() -> Dict[str, Any]:
    """
    Return a dictionary of models to train.

    Returns:
        Dictionary mapping model names to model instances.
    """
    return {
        "logistic_regression": LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            random_state=RANDOM_SEED,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=RANDOM_SEED,
            use_label_encoder=False,
            eval_metric="logloss",
        ),
    }


def evaluate_model(
    model: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Tuple[Any, Dict[str, float]]:
    """
    Train and evaluate a model using cross-validation and holdout test set.

    Args:
        model: sklearn-compatible classifier.
        X_train: Training feature matrix.
        X_test: Test feature matrix.
        y_train: Training target vector.
        y_test: Test target vector.

    Returns:
        Tuple of (trained_model, metrics_dict).
    """
    # Cross-validation
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")

    # Train on full training set
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "cv_mean_roc_auc": cv_scores.mean(),
        "cv_std_roc_auc": cv_scores.std(),
    }

    return model, metrics


def train_all_models(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Dict[str, Dict[str, Any]]:
    """
    Train all models and log to MLflow.

    Args:
        X_train: Training feature matrix.
        X_test: Test feature matrix.
        y_train: Training target vector.
        y_test: Test target vector.

    Returns:
        Dictionary of results for each model.
    """
    # Set up MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    models = get_models()
    results = {}

    for name, model in models.items():
        logger.info(f"Training {name}...")

        with mlflow.start_run(run_name=name):
            # Train and evaluate
            trained_model, metrics = evaluate_model(model, X_train, X_test, y_train, y_test)

            # Log parameters
            mlflow.log_params(model.get_params())

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log model
            mlflow.sklearn.log_model(trained_model, "model")

            # Store results
            results[name] = {
                "model": trained_model,
                "metrics": metrics,
            }

            logger.info(f"{name} - ROC-AUC: {metrics['roc_auc']:.4f}")

    return results


def save_best_model(
    results: Dict[str, Dict[str, Any]],
    preprocessor: Any,
) -> str:
    """
    Save the best model and preprocessor to disk.

    Args:
        results: Dictionary of training results.
        preprocessor: Fitted preprocessor pipeline.

    Returns:
        Name of the best model.
    """
    # Find best model by ROC-AUC
    best_name = max(results, key=lambda x: results[x]["metrics"]["roc_auc"])
    best_result = results[best_name]

    logger.info(f"Best model: {best_name} (ROC-AUC: {best_result['metrics']['roc_auc']:.4f})")

    # Create models directory
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Save preprocessor
    preprocessor_path = MODELS_DIR / "preprocessor.joblib"
    joblib.dump(preprocessor, preprocessor_path)
    logger.info(f"Preprocessor saved to {preprocessor_path}")

    # Save model
    model_path = MODELS_DIR / "model.joblib"
    joblib.dump(best_result["model"], model_path)
    logger.info(f"Model saved to {model_path}")

    # Save metadata
    metadata = {
        "model_name": best_name,
        "metrics": best_result["metrics"],
    }
    metadata_path = MODELS_DIR / "model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to {metadata_path}")

    return best_name


def main() -> None:
    """Main training pipeline."""
    logger.info("Starting training pipeline...")

    # Load data
    logger.info("Loading dataset...")
    df = load_raw_data()

    # Prepare data
    logger.info("Preparing data...")
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(df)

    # Train models
    logger.info("Training models...")
    results = train_all_models(X_train, X_test, y_train, y_test)

    # Save best model
    logger.info("Saving best model...")
    best_model = save_best_model(results, preprocessor)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Best Model: {best_model}")
    logger.info(f"Test ROC-AUC: {results[best_model]['metrics']['roc_auc']:.4f}")
    logger.info(f"Test Accuracy: {results[best_model]['metrics']['accuracy']:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
