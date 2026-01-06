"""
Unit tests for API endpoints.
"""
import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(autouse=True)
def clear_model_cache():
    """Clear model cache before each test to ensure fresh state."""
    from src.predict import _model_cache

    _model_cache.clear()
    yield
    _model_cache.clear()


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_returns_200(self, client):
        """Verify health endpoint returns 200 status."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_status(self, client):
        """Verify health response contains status field."""
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root_returns_200(self, client):
        """Verify root endpoint returns 200 status."""
        response = client.get("/")
        assert response.status_code == 200

    def test_root_returns_message(self, client):
        """Verify root response contains message."""
        response = client.get("/")
        data = response.json()
        assert "message" in data


class TestPredictEndpoint:
    """Tests for prediction endpoint."""

    @pytest.fixture
    def valid_input(self):
        """Valid patient data for testing."""
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

    def test_predict_returns_200(self, client, valid_input):
        """Verify predict endpoint returns 200 for valid input."""
        response = client.post("/predict", json=valid_input)
        assert response.status_code == 200

    def test_predict_returns_prediction(self, client, valid_input):
        """Verify predict response has required fields."""
        response = client.post("/predict", json=valid_input)
        data = response.json()

        assert "prediction" in data
        assert "probability" in data
        assert "risk_level" in data

    def test_predict_prediction_is_binary(self, client, valid_input):
        """Verify prediction is 0 or 1."""
        response = client.post("/predict", json=valid_input)
        data = response.json()

        assert data["prediction"] in [0, 1]

    def test_predict_probability_in_range(self, client, valid_input):
        """Verify probability is between 0 and 1."""
        response = client.post("/predict", json=valid_input)
        data = response.json()

        assert 0 <= data["probability"] <= 1

    def test_predict_invalid_age_fails(self, client, valid_input):
        """Verify invalid age is rejected."""
        valid_input["age"] = -5
        response = client.post("/predict", json=valid_input)
        assert response.status_code == 422

    def test_predict_missing_field_fails(self, client):
        """Verify missing fields are rejected."""
        incomplete = {"age": 63, "sex": 1}
        response = client.post("/predict", json=incomplete)
        assert response.status_code == 422


class TestMetricsEndpoint:
    """Tests for metrics endpoint."""

    def test_metrics_returns_200(self, client):
        """Verify metrics endpoint returns 200 status."""
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_returns_prometheus_format(self, client):
        """Verify metrics are in Prometheus format."""
        response = client.get("/metrics")
        content = response.text

        # Should contain prometheus metric definitions
        assert "prediction" in content or "HELP" in content or "#" in content
