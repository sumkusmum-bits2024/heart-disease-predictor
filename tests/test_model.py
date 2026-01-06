"""
Unit tests for model training and prediction.
"""
import json

import pytest

from src.config import MODELS_DIR


class TestModelFiles:
    """Tests for model file existence and loading."""

    def test_model_file_exists(self):
        """Verify that the trained model file exists."""
        model_path = MODELS_DIR / "model.joblib"
        assert model_path.exists(), f"Model not found at {model_path}"

    def test_preprocessor_file_exists(self):
        """Verify that the preprocessor file exists."""
        preprocessor_path = MODELS_DIR / "preprocessor.joblib"
        assert preprocessor_path.exists(), f"Preprocessor not found at {preprocessor_path}"

    def test_metadata_file_exists(self):
        """Verify that the model metadata file exists."""
        metadata_path = MODELS_DIR / "model_metadata.json"
        assert metadata_path.exists(), f"Metadata not found at {metadata_path}"

    def test_metadata_has_required_fields(self):
        """Verify metadata contains required fields."""
        metadata_path = MODELS_DIR / "model_metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)

        assert "model_name" in metadata
        assert "metrics" in metadata
        assert "roc_auc" in metadata["metrics"]


class TestModelLoading:
    """Tests for model and preprocessor loading."""

    def test_model_loads_successfully(self):
        """Verify that the model loads without errors."""
        from src.predict import load_model

        model = load_model()
        assert model is not None
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")

    def test_preprocessor_loads_successfully(self):
        """Verify that the preprocessor loads without errors."""
        from src.predict import load_preprocessor

        preprocessor = load_preprocessor()
        assert preprocessor is not None
        assert hasattr(preprocessor, "transform")


class TestPrediction:
    """Tests for model prediction functionality."""

    @pytest.fixture
    def sample_input(self):
        """Sample patient data for testing."""
        return {
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

    def test_predict_returns_dict(self, sample_input):
        """Verify that predict returns a dictionary."""
        from src.predict import predict

        result = predict(sample_input)
        assert isinstance(result, dict)

    def test_predict_has_required_keys(self, sample_input):
        """Verify that prediction result has required keys."""
        from src.predict import predict

        result = predict(sample_input)
        assert "prediction" in result
        assert "probability" in result
        assert "risk_level" in result

    def test_prediction_is_binary(self, sample_input):
        """Verify that prediction is 0 or 1."""
        from src.predict import predict

        result = predict(sample_input)
        assert result["prediction"] in [0, 1]

    def test_probability_in_valid_range(self, sample_input):
        """Verify that probability is between 0 and 1."""
        from src.predict import predict

        result = predict(sample_input)
        assert 0 <= result["probability"] <= 1

    def test_risk_level_is_valid(self, sample_input):
        """Verify that risk level is one of expected values."""
        from src.predict import predict

        result = predict(sample_input)
        assert result["risk_level"] in ["Low", "Moderate", "High"]

    def test_predict_handles_edge_case_young(self):
        """Test prediction for young patient with good metrics."""
        from src.predict import predict

        young_healthy = {
            "age": 25,
            "sex": 0,
            "cp": 1,
            "trestbps": 110,
            "chol": 180,
            "fbs": 0,
            "restecg": 0,
            "thalach": 180,
            "exang": 0,
            "oldpeak": 0.0,
            "slope": 1,
            "ca": 0,
            "thal": 3,
        }

        result = predict(young_healthy)
        assert result["prediction"] in [0, 1]
        assert 0 <= result["probability"] <= 1

    def test_predict_handles_edge_case_elderly(self):
        """Test prediction for elderly patient with concerning metrics."""
        from src.predict import predict

        elderly_risky = {
            "age": 75,
            "sex": 1,
            "cp": 4,
            "trestbps": 180,
            "chol": 300,
            "fbs": 1,
            "restecg": 2,
            "thalach": 100,
            "exang": 1,
            "oldpeak": 4.0,
            "slope": 3,
            "ca": 3,
            "thal": 7,
        }

        result = predict(elderly_risky)
        assert result["prediction"] in [0, 1]
        assert 0 <= result["probability"] <= 1

    def test_predict_missing_feature_raises_error(self):
        """Verify that missing features raise ValueError."""
        from src.predict import predict

        incomplete_input = {
            "age": 63,
            "sex": 1,
            # Missing other features
        }

        with pytest.raises(ValueError):
            predict(incomplete_input)


class TestModelMetrics:
    """Tests for model performance metrics."""

    def test_model_has_acceptable_roc_auc(self):
        """Verify that model ROC-AUC is above threshold."""
        metadata_path = MODELS_DIR / "model_metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)

        roc_auc = metadata["metrics"]["roc_auc"]
        assert roc_auc >= 0.75, f"ROC-AUC {roc_auc} is below threshold 0.75"

    def test_model_has_acceptable_accuracy(self):
        """Verify that model accuracy is above threshold."""
        metadata_path = MODELS_DIR / "model_metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)

        accuracy = metadata["metrics"]["accuracy"]
        assert accuracy >= 0.70, f"Accuracy {accuracy} is below threshold 0.70"
