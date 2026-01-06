"""
FastAPI application for Heart Disease Prediction API.

Provides endpoints for health checks, predictions, and metrics.
"""
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from starlette.responses import Response

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    "prediction_requests_total",
    "Total number of prediction requests",
    ["status"],
)
REQUEST_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Request latency in seconds",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)
PREDICTION_CLASSES = Counter(
    "predictions_by_class",
    "Number of predictions by class",
    ["prediction_class"],
)


class PatientData(BaseModel):
    """Input schema for patient health data."""

    age: int = Field(..., ge=0, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="Sex (0=female, 1=male)")
    cp: int = Field(
        ...,
        ge=1,
        le=4,
        description="Chest pain type (1=typical angina, 2=atypical angina, 3=non-anginal, 4=asymptomatic)",
    )
    trestbps: int = Field(..., ge=0, description="Resting blood pressure (mm Hg)")
    chol: int = Field(..., ge=0, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(
        ..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (1=true, 0=false)"
    )
    restecg: int = Field(
        ...,
        ge=0,
        le=2,
        description="Resting ECG results (0=normal, 1=ST-T abnormality, 2=LV hypertrophy)",
    )
    thalach: int = Field(..., ge=0, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina (1=yes, 0=no)")
    oldpeak: float = Field(..., ge=0, description="ST depression induced by exercise")
    slope: int = Field(
        ...,
        ge=1,
        le=3,
        description="Slope of peak exercise ST (1=upsloping, 2=flat, 3=downsloping)",
    )
    ca: int = Field(
        ..., ge=0, le=3, description="Number of major vessels colored by fluoroscopy (0-3)"
    )
    thal: int = Field(
        ..., description="Thalassemia (3=normal, 6=fixed defect, 7=reversible defect)"
    )

    class Config:
        json_schema_extra = {
            "example": {
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
        }


class PredictionResponse(BaseModel):
    """Output schema for prediction results."""

    prediction: int = Field(..., description="0=No disease, 1=Heart disease")
    probability: float = Field(..., description="Probability of heart disease")
    risk_level: str = Field(..., description="Risk level (Low/Moderate/High)")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    version: str


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""

    patients: List[PatientData]


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""

    predictions: List[PredictionResponse]


# Initialize model on startup
model_state: Dict[str, Any] = {"loaded": False}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    try:
        # Import here to avoid circular imports
        from src.predict import load_model, load_preprocessor

        load_model()
        load_preprocessor()
        model_state["loaded"] = True
        logger.info("Model loaded successfully on startup")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model_state["loaded"] = False

    yield

    logger.info("Shutting down API")


# Create FastAPI app
app = FastAPI(
    title="Heart Disease Prediction API",
    description="ML-powered API for predicting heart disease risk based on patient health data.",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
async def root() -> Dict[str, str]:
    """Root endpoint with API information."""
    return {
        "message": "Heart Disease Prediction API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns the current status of the API and model.
    """
    return HealthResponse(
        status="healthy" if model_state["loaded"] else "degraded",
        model_loaded=model_state["loaded"],
        version="1.0.0",
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(data: PatientData) -> PredictionResponse:
    """
    Make a heart disease prediction for a single patient.

    Args:
        data: Patient health data.

    Returns:
        Prediction result with probability and risk level.
    """
    start_time = time.time()

    try:
        from src.predict import predict as make_prediction

        result = make_prediction(data.model_dump())

        # Record metrics
        REQUEST_COUNT.labels(status="success").inc()
        PREDICTION_CLASSES.labels(prediction_class=str(result["prediction"])).inc()
        REQUEST_LATENCY.observe(time.time() - start_time)

        logger.info(
            f"Prediction made: class={result['prediction']}, "
            f"probability={result['probability']:.4f}"
        )

        return PredictionResponse(**result)

    except Exception as e:
        import traceback

        REQUEST_COUNT.labels(status="error").inc()
        logger.error(f"Prediction error: {e}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e)) from None


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Prediction"],
)
async def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """
    Make predictions for multiple patients.

    Args:
        request: List of patient health data.

    Returns:
        List of prediction results.
    """
    start_time = time.time()

    try:
        from src.predict import predict as make_prediction

        predictions = []
        for patient in request.patients:
            result = make_prediction(patient.model_dump())
            predictions.append(PredictionResponse(**result))
            PREDICTION_CLASSES.labels(prediction_class=str(result["prediction"])).inc()

        REQUEST_COUNT.labels(status="success").inc()
        REQUEST_LATENCY.observe(time.time() - start_time)

        logger.info(f"Batch prediction made for {len(predictions)} patients")

        return BatchPredictionResponse(predictions=predictions)

    except Exception as e:
        REQUEST_COUNT.labels(status="error").inc()
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from None


@app.get("/metrics", tags=["Monitoring"])
async def metrics() -> Response:
    """
    Prometheus metrics endpoint.

    Returns metrics in Prometheus format.
    """
    return Response(content=generate_latest(), media_type="text/plain")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
