# Heart Disease Prediction - System Architecture

## High-Level Architecture

```mermaid
flowchart TB
    subgraph DataPipeline [Data Pipeline]
        UCI[(UCI Repository)]
        Raw[Raw Data]
        Preprocessor[Preprocessing Pipeline]
        Features[Feature Engineering]
    end

    subgraph Training [Training Pipeline]
        EDA[EDA Notebook]
        Models[Model Training]
        MLflow[(MLflow Tracking)]
        BestModel[Best Model Selection]
    end

    subgraph Serving [Model Serving]
        API[FastAPI Application]
        Predict[Prediction Endpoint]
        Health[Health Endpoint]
        Metrics[Metrics Endpoint]
    end

    subgraph Deployment [Deployment]
        Docker[Docker Container]
        K8s[Kubernetes Cluster]
        Service[LoadBalancer Service]
    end

    subgraph Monitoring [Monitoring Stack]
        Prometheus[(Prometheus)]
        Grafana[Grafana Dashboard]
    end

    subgraph CICD [CI/CD Pipeline]
        GitHub[GitHub Repository]
        Actions[GitHub Actions]
        Lint[Lint & Format]
        Tests[Unit Tests]
        Build[Docker Build]
    end

    UCI --> Raw
    Raw --> Preprocessor
    Preprocessor --> Features
    Features --> EDA
    Features --> Models
    Models --> MLflow
    Models --> BestModel
    BestModel --> API

    API --> Predict
    API --> Health
    API --> Metrics
    API --> Docker
    Docker --> K8s
    K8s --> Service

    Metrics --> Prometheus
    Prometheus --> Grafana

    GitHub --> Actions
    Actions --> Lint
    Lint --> Tests
    Tests --> Build
```

## Component Details

### 1. Data Pipeline
- **Data Source**: UCI Heart Disease Dataset (Cleveland)
- **Preprocessing**: Missing value handling, feature scaling, categorical encoding
- **Output**: Cleaned dataset ready for model training

### 2. Training Pipeline
- **EDA**: Comprehensive exploratory data analysis with visualizations
- **Models**: Logistic Regression, Random Forest, XGBoost
- **Tracking**: All experiments tracked in MLflow
- **Selection**: Best model selected based on ROC-AUC metric

### 3. Model Serving
- **FastAPI**: Modern, fast Python web framework
- **Endpoints**:
  - `/predict`: Make heart disease predictions
  - `/health`: API health check
  - `/metrics`: Prometheus metrics
- **Features**: Input validation, structured logging, error handling

### 4. Deployment
- **Docker**: Multi-stage build for optimized images
- **Kubernetes**: 
  - Deployment with 2 replicas
  - LoadBalancer service
  - Health probes (liveness/readiness)
  - Resource limits

### 5. Monitoring
- **Prometheus**: Metrics collection
- **Grafana**: Visualization dashboard
- **Metrics**:
  - Request rate
  - Latency percentiles
  - Prediction distribution

### 6. CI/CD Pipeline
- **Trigger**: Push/PR to main branch
- **Stages**:
  1. Lint (Black, Ruff)
  2. Test (pytest with coverage)
  3. Build (Docker image)
- **Artifacts**: Trained model, Docker image

## Data Flow

```mermaid
sequenceDiagram
    participant Client
    participant API as FastAPI
    participant Preprocessor
    participant Model
    participant Prometheus

    Client->>API: POST /predict (patient data)
    API->>API: Validate input
    API->>Preprocessor: Transform features
    Preprocessor->>Model: Predict
    Model->>API: Prediction + Probability
    API->>Prometheus: Record metrics
    API->>Client: Response (prediction, risk_level)
```

## Technology Stack

| Layer | Technology |
|-------|------------|
| Data Processing | pandas, scikit-learn |
| Machine Learning | scikit-learn, XGBoost |
| Experiment Tracking | MLflow |
| API Framework | FastAPI, Pydantic |
| Containerization | Docker |
| Orchestration | Kubernetes |
| Monitoring | Prometheus, Grafana |
| CI/CD | GitHub Actions |
| Testing | pytest, pytest-cov |
| Code Quality | Black, Ruff |


