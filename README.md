# Heart Disease Prediction - MLOps Pipeline

A production-ready machine learning pipeline for predicting heart disease risk, built with modern MLOps best practices including CI/CD, containerization, Kubernetes deployment, and monitoring.

## Project Overview

This project implements an end-to-end ML solution that:
- Predicts heart disease risk based on patient health data
- Tracks experiments with MLflow
- Deploys as a REST API with FastAPI
- Runs in Docker containers on Kubernetes
- Includes automated testing and CI/CD with GitHub Actions

### Dataset
**Heart Disease UCI Dataset** from the UCI Machine Learning Repository
- 303 patient records
- 13 features (age, sex, chest pain type, blood pressure, cholesterol, etc.)
- Binary target: presence/absence of heart disease

## Quick Start

```bash
# 1. Clone and setup
git clone <repository-url>
cd mlops-assignment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Download data and train model
python -c "from src.data_loader import load_raw_data; load_raw_data()"
python -m src.train

# 3. Run the API
uvicorn api.main:app --reload

# 4. Test the endpoint
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":4,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":2,"ca":0,"thal":6}'
```

## Project Structure

```
mlops-assignment/
├── .github/workflows/       # CI/CD pipeline
│   └── ci-cd.yml
├── api/                     # FastAPI application
│   └── main.py
├── data/                    # Dataset storage
│   ├── raw/
│   └── processed/
├── deployment/              # Docker & Kubernetes
│   ├── Dockerfile
│   └── k8s/
│       ├── deployment.yaml
│       └── service.yaml
├── models/                  # Trained models
├── mlruns/                  # MLflow tracking
├── notebooks/               # EDA & Modeling
│   └── 01_eda_and_modeling.ipynb
├── screenshots/             # Documentation images
├── src/                     # Source code
│   ├── config.py           # Configuration
│   ├── data_loader.py      # Data acquisition
│   ├── preprocessing.py    # Feature engineering
│   ├── train.py            # Model training
│   └── predict.py          # Inference
├── tests/                   # Unit tests
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── requirements.txt         # Dependencies
├── pyproject.toml          # Tool configuration
└── README.md
```

## Features

### Data Processing
- Automated dataset download from UCI repository
- Missing value handling with imputation
- Feature scaling and categorical encoding
- Stratified train/test split

### Model Training
- Three models evaluated: Logistic Regression, Random Forest, XGBoost
- 5-fold stratified cross-validation
- Comprehensive metrics: Accuracy, Precision, Recall, F1, ROC-AUC
- Best model selection based on ROC-AUC

### Experiment Tracking
All experiments are tracked with MLflow:
```bash
# View MLflow UI
mlflow ui --backend-store-uri file:./mlruns
# Open http://localhost:5000
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/predict` | POST | Single prediction |
| `/predict/batch` | POST | Batch predictions |
| `/metrics` | GET | Prometheus metrics |

### API Request Example
```json
{
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
  "thal": 6
}
```

### API Response
```json
{
  "prediction": 1,
  "probability": 0.78,
  "risk_level": "High"
}
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=src --cov=api --cov-report=term-missing

# Run specific test file
pytest tests/test_api.py -v
```

## Docker

```bash
# Build image
docker build -f deployment/Dockerfile -t heart-disease-api:latest .

# Run container
docker run -p 8000:8000 heart-disease-api:latest

# Test
curl http://localhost:8000/health
```

## Kubernetes Deployment

### Using Minikube
```bash
# Start Minikube
minikube start

# Load local image
minikube image load heart-disease-api:latest

# Deploy
kubectl apply -f deployment/k8s/deployment.yaml
kubectl apply -f deployment/k8s/service.yaml

# Get service URL
minikube service heart-disease-api-service --url

# Verify deployment
kubectl get pods
kubectl get services
```

### Using Docker Desktop
```bash
# Enable Kubernetes in Docker Desktop settings
kubectl apply -f deployment/k8s/deployment.yaml
kubectl apply -f deployment/k8s/service.yaml

# Access at http://localhost:80
```

## CI/CD Pipeline

The GitHub Actions workflow includes:
1. **Lint**: Code formatting (Black) and linting (Ruff)
2. **Test**: Unit tests with coverage reporting
3. **Build**: Docker image build

Triggered on push/PR to `main` branch.

## Architecture

See [docs/architecture.md](docs/architecture.md) for detailed architecture diagrams and component descriptions.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CI/CD Pipeline                               │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────────────┐  │
│  │ GitHub  │───►│  Lint   │───►│  Test   │───►│  Docker Build   │  │
│  └─────────┘    └─────────┘    └─────────┘    └─────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Training Pipeline                            │
│  ┌─────────┐    ┌─────────────┐    ┌─────────┐    ┌─────────────┐  │
│  │Raw Data │───►│Preprocessing│───►│ Models  │───►│Best Model   │  │
│  └─────────┘    └─────────────┘    └─────────┘    └─────────────┘  │
│                                         │                           │
│                                         ▼                           │
│                                   ┌─────────┐                       │
│                                   │ MLflow  │                       │
│                                   └─────────┘                       │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Serving Layer                                │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                     FastAPI Application                      │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │   │
│  │  │ /predict │  │ /health  │  │ /metrics │  │ /docs    │    │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Deployment                                   │
│  ┌─────────────┐         ┌──────────────────────────────────────┐  │
│  │   Docker    │────────►│         Kubernetes Cluster           │  │
│  │  Container  │         │  ┌─────────┐  ┌─────────┐           │  │
│  └─────────────┘         │  │  Pod 1  │  │  Pod 2  │           │  │
│                          │  └─────────┘  └─────────┘           │  │
│                          │        ▲            ▲                │  │
│                          │        └─────┬──────┘                │  │
│                          │              │                       │  │
│                          │     ┌────────────────┐               │  │
│                          │     │ LoadBalancer   │               │  │
│                          │     └────────────────┘               │  │
│                          └──────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Monitoring                                   │
│  ┌─────────────┐         ┌─────────────┐                           │
│  │ Prometheus  │────────►│   Grafana   │                           │
│  │  /metrics   │         │  Dashboard  │                           │
│  └─────────────┘         └─────────────┘                           │
└─────────────────────────────────────────────────────────────────────┘
```

## Monitoring

### Quick Start with Docker Compose
```bash
# Start the full monitoring stack
cd deployment/monitoring
docker compose up -d

# Access services:
# - API: http://localhost:8000
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000 (admin/admin)
```

### Prometheus Metrics
Access at `/metrics` endpoint:
- `prediction_requests_total` - Total prediction requests
- `prediction_latency_seconds` - Request latency histogram
- `predictions_by_class` - Counter by prediction class

### Grafana Dashboard
The dashboard is auto-provisioned and includes:
- Request rate graph
- Latency percentile chart (p50, p95, p99)
- Prediction class distribution pie chart
- Total predictions counter
- Average latency stat

You can also manually import `deployment/monitoring/grafana-dashboard.json`.

## Model Performance

| Model | Accuracy | ROC-AUC | F1 Score |
|-------|----------|---------|----------|
| Logistic Regression | 0.89 | 0.97 | 0.88 |
| Random Forest | 0.87 | 0.95 | 0.86 |
| XGBoost | 0.85 | 0.94 | 0.84 |

## Feature Importance

Top predictive features:
1. `cp` (Chest Pain Type)
2. `thalach` (Max Heart Rate)
3. `oldpeak` (ST Depression)
4. `ca` (Number of Major Vessels)
5. `thal` (Thalassemia)

## License

This project is for educational purposes as part of the MLOps course assignment.

## Author

MLOps Assignment - S1-25_AIMLCZG523
