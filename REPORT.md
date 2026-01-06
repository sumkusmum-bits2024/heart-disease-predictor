# Heart Disease Prediction: An End-to-End MLOps Pipeline

**Course**: MLOps (S1-25_AIMLCZG523)  
**Assignment**: MLOps Pipeline Implementation  
**Date**: January 2026

---

## Table of Contents

1. [Introduction and Problem Statement](#1-introduction-and-problem-statement)
2. [Exploratory Data Analysis](#2-exploratory-data-analysis)
3. [Model Selection and Results](#3-model-selection-and-results)
4. [MLflow Experiment Tracking](#4-mlflow-experiment-tracking)
5. [CI/CD Pipeline Architecture](#5-cicd-pipeline-architecture)
6. [Containerization and Deployment](#6-containerization-and-deployment)
7. [Monitoring Setup](#7-monitoring-setup)
8. [Conclusion and Future Work](#8-conclusion-and-future-work)

---

## 1. Introduction and Problem Statement

### 1.1 Background

Cardiovascular diseases (CVDs) are the leading cause of death globally, claiming an estimated 17.9 million lives each year according to the World Health Organization. Early detection and prediction of heart disease can significantly improve patient outcomes and reduce healthcare costs. Machine learning offers a powerful approach to analyze patient health data and predict heart disease risk.

### 1.2 Problem Statement

The objective of this project is to build a **production-ready machine learning pipeline** that:

1. Predicts the presence of heart disease based on clinical and demographic features
2. Provides a reliable, scalable API for real-time predictions
3. Implements MLOps best practices including:
   - Experiment tracking
   - Continuous Integration/Continuous Deployment (CI/CD)
   - Containerization
   - Kubernetes deployment
   - Monitoring and observability

### 1.3 Dataset

We use the **Heart Disease UCI Dataset** from the UCI Machine Learning Repository (Cleveland database). The dataset contains:

- **303 patient records**
- **13 clinical features** including:
  - Demographics: age, sex
  - Cardiac measurements: resting blood pressure, cholesterol, max heart rate
  - Symptoms: chest pain type, exercise-induced angina
  - ECG results: resting ECG, ST depression
  - Other: number of major vessels colored by fluoroscopy, thalassemia type
- **Binary target**: Presence (1) or absence (0) of heart disease

### 1.4 Project Scope

The project implements a complete MLOps pipeline from data acquisition to production deployment, demonstrating industry-standard practices for machine learning system design and operations.

---

## 2. Exploratory Data Analysis

### 2.1 Dataset Overview

The initial data exploration revealed the following characteristics:

| Statistic          | Value                              |
|--------------------|-------------------------------------|
| Total samples      | 303                                 |
| Features           | 13                                  |
| Missing values     | 6 (in `ca` and `thal` columns)      |
| Class distribution | 46% No Disease, 54% Heart Disease   |

The dataset is reasonably balanced with a slight majority of positive cases, reducing the need for specialized handling of class imbalance.

### 2.2 Feature Analysis

#### Numerical Features Distribution

Analysis of numerical features revealed distinct patterns between heart disease positive and negative cases:

1. **Age**: Patients with heart disease tend to be older (mean: 56.6 years) compared to healthy patients (mean: 52.5 years)

2. **Maximum Heart Rate (thalach)**: Healthy patients achieve significantly higher maximum heart rates (average: 158 bpm) compared to those with heart disease (average: 139 bpm)

3. **ST Depression (oldpeak)**: Patients with heart disease show higher ST depression values, indicating exercise-induced cardiac stress

4. **Resting Blood Pressure**: While elevated in heart disease patients, the difference is less pronounced than other features

5. **Cholesterol**: Surprisingly, cholesterol showed the weakest correlation with heart disease in this dataset

#### Categorical Features Analysis

Categorical features showed strong predictive signals:

1. **Chest Pain Type (cp)**: Asymptomatic chest pain (type 4) showed the highest association with heart disease (77% positive rate)

2. **Exercise-Induced Angina (exang)**: Present in 55% of heart disease cases vs only 18% of healthy cases

3. **Thalassemia (thal)**: Fixed defect (type 6) and reversible defect (type 7) showed strong associations with heart disease

4. **Number of Major Vessels (ca)**: Higher values correlated strongly with heart disease presence

![Target Distribution](screenshots/target_distribution.png)

![Correlation Heatmap](screenshots/correlation_heatmap.png)

### 2.3 Correlation Analysis

The correlation matrix revealed key insights:

**Top 5 features correlated with target:**

| Feature | Correlation |
|---------|-------------|
| thal    | 0.53        |
| ca      | 0.46        |
| exang   | 0.43        |
| oldpeak | 0.42        |
| thalach | -0.42       |

Notable negative correlation of `thalach` indicates that lower maximum heart rate is associated with heart disease.

### 2.4 Missing Value Treatment

Missing values were found in:
- `ca`: 4 missing values (1.3%)
- `thal`: 2 missing values (0.7%)

These were handled during preprocessing using most frequent value imputation, as the missing percentage was minimal and the features were categorical.

### 2.5 Key Insights

1. **Strong Predictors**: Chest pain type, maximum heart rate, and thalassemia status are the most discriminative features
2. **Age Factor**: Heart disease prevalence increases with age
3. **Exercise Response**: Exercise-induced symptoms (angina, ST depression) are highly indicative
4. **Balanced Classes**: No significant class imbalance requiring specialized handling

---

## 3. Model Selection and Results

### 3.1 Preprocessing Pipeline

A robust sklearn preprocessing pipeline was implemented:

```python
ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numerical_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])
```

**Features:**
- Numerical features: StandardScaler normalization
- Categorical features: OneHotEncoder transformation
- Missing values: Median imputation (numerical), Mode imputation (categorical)
- Train/Test split: 80/20 with stratification

### 3.2 Models Evaluated

Three classification algorithms were evaluated:

1. **Logistic Regression**: Baseline linear model, highly interpretable
2. **Random Forest**: Ensemble method with feature importance
3. **XGBoost**: Gradient boosting for optimal performance

### 3.3 Evaluation Methodology

- **Cross-validation**: 5-fold stratified K-fold
- **Metrics**: Accuracy, Precision, Recall, F1-score, ROC-AUC
- **Final evaluation**: Holdout test set (20%)

### 3.4 Results

*All metrics computed using 5-fold cross-validation for more robust evaluation.*

| Model                   | Accuracy  | Precision | Recall | F1-Score | ROC-AUC |
|-------------------------|-----------|-----------|--------|----------|---------|
| **Logistic Regression** | **0.847** | **0.877** | **0.783** | **0.825** | **0.902** |
| Random Forest           | 0.831     | 0.856     | 0.770  | 0.807    | 0.889   |
| XGBoost                 | 0.819     | 0.841     | 0.761  | 0.795    | 0.876   |

![Model Comparison](screenshots/model_comparison.png)

### 3.5 Model Selection

**Logistic Regression** was selected as the best model based on:

1. **Highest ROC-AUC (0.902)**: Good discrimination capability
2. **Best F1-Score (0.825)**: Balanced precision-recall trade-off
3. **Highest Precision (0.877)**: Important for reducing false positives
4. **Interpretability**: Coefficients provide clinical insights
5. **Simplicity**: Lower computational requirements for production

### 3.6 Feature Importance Analysis

From the trained Logistic Regression model, the most important features (by absolute coefficient magnitude) were:

1. Chest Pain Type 4 (Asymptomatic)
2. Thalassemia Type 7 (Reversible Defect)
3. Number of Major Vessels (ca)
4. Maximum Heart Rate (thalach)
5. ST Depression (oldpeak)

These align with clinical knowledge about heart disease indicators.

---

## 4. MLflow Experiment Tracking

### 4.1 MLflow Setup

MLflow was configured for comprehensive experiment tracking:

```python
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("heart-disease-classification")
```

### 4.2 Logged Artifacts

For each model run, the following were tracked:

**Parameters:**
- All hyperparameters (C, solver, n_estimators, max_depth, etc.)
- Random seed (42)
- Test size (0.2)
- Cross-validation folds (5)

**Metrics:**
- Training and test accuracy
- Precision, Recall, F1-Score
- ROC-AUC
- Cross-validation mean and standard deviation

**Artifacts:**
- Trained model (sklearn format)
- Preprocessing pipeline
- Confusion matrix visualization
- ROC curve plot
- Model metadata JSON

### 4.3 Benefits Achieved

1. **Reproducibility**: All experiments can be recreated with logged parameters
2. **Comparison**: Easy comparison of model performance across runs
3. **Versioning**: Model artifacts versioned automatically
4. **Collaboration**: Shareable experiment results

### 4.4 MLflow UI

![MLflow Experiments](screenshots/mlflow_models.png)

![MLflow Metrics Comparison](screenshots/mlflow_comparison_metrics.png)

The MLflow UI provides:
- Run comparison tables
- Metric visualization over time
- Artifact browsing
- Model registry integration

---

## 5. CI/CD Pipeline Architecture

### 5.1 GitHub Actions Workflow

A comprehensive CI/CD pipeline was implemented using GitHub Actions:

```yaml
name: MLOps CI/CD Pipeline
on:
  push:
    branches: [main, master]
  pull_request:
    branches: [main, master]
```

### 5.2 Pipeline Stages

#### Stage 1: Lint (Code Quality)
- **Black**: Code formatting check
- **Ruff**: Static code analysis
- Ensures consistent code style across the codebase

#### Stage 2: Test (Quality Assurance)
- Install dependencies
- Download dataset
- Train model
- Run pytest with coverage
- Upload coverage reports

#### Stage 3: Build (Artifact Creation)
- Build Docker image
- Cache layers for efficiency
- Upload trained model artifacts

### 5.3 Test Coverage

The test suite includes:

| Test File      | Tests  | Coverage                    |
|----------------|--------|------------------------------|
| test_data.py   | 13     | Data loading, preprocessing  |
| test_model.py  | 16     | Model loading, predictions   |
| test_api.py    | 12     | API endpoints                |
| **Total**      | **41** | **54%**                      |

### 5.4 Pipeline Benefits

1. **Automated Quality Gates**: No merge without passing tests
2. **Reproducible Builds**: Consistent environment via containers
3. **Fast Feedback**: Issues caught early in development
4. **Documentation**: Pipeline as code serves as documentation

---

## 6. Containerization and Deployment

### 6.1 Docker Implementation

A multi-stage Dockerfile was created for optimal image size:

```dockerfile
# Build stage - install dependencies
FROM python:3.10-slim AS builder
# ... build steps ...

# Production stage - minimal runtime
FROM python:3.10-slim AS production
# ... production setup ...
```

**Features:**
- Multi-stage build (~50% smaller image)
- Non-root user for security
- Health check endpoint integration
- Layer caching for fast rebuilds

### 6.2 FastAPI Application

The REST API provides:

| Endpoint         | Method | Description                     |
|------------------|--------|---------------------------------|
| `/`              | GET    | API information                 |
| `/health`        | GET    | Health check with model status  |
| `/predict`       | POST   | Single patient prediction       |
| `/predict/batch` | POST   | Batch predictions               |
| `/metrics`       | GET    | Prometheus metrics              |
| `/docs`          | GET    | Interactive API documentation   |

**Request Validation:**
- Pydantic models for input validation
- Clear error messages for invalid inputs
- Documented field constraints

![FastAPI Documentation](screenshots/fastapi.png)

### 6.3 Kubernetes Deployment

Kubernetes manifests were created for production deployment:

**Deployment Features:**
- 2 replicas for high availability
- Rolling update strategy
- Resource limits (256Mi-512Mi memory, 250m-500m CPU)
- Liveness and readiness probes
- Prometheus scraping annotations

**Service Configuration:**
- LoadBalancer type for external access
- Port 80 → 8000 mapping
- Label-based pod selection

### 6.4 Deployment Script

A deployment script (`scripts/deploy.sh`) automates:
1. Docker image building
2. Kubernetes cluster verification
3. Image loading to cluster
4. Manifest application
5. Rollout status monitoring
6. Service URL retrieval

---

## 7. Monitoring Setup

### 7.1 Prometheus Metrics

The API exposes custom Prometheus metrics:

```python
# Request counter by status (success/error)
prediction_requests_total

# Latency histogram with buckets
prediction_latency_seconds

# Prediction class distribution
predictions_by_class
```

### 7.2 Grafana Dashboard

A pre-configured Grafana dashboard includes:

1. **Request Rate Panel**: Time series of requests/second by status
2. **Latency Percentiles**: p50, p95, p99 latency tracking
3. **Prediction Distribution**: Pie chart of class predictions
4. **Total Predictions**: Counter stat panel
5. **Average Latency**: Real-time latency indicator

### 7.3 Monitoring Stack Deployment

Docker Compose configuration enables one-command deployment:

```bash
cd deployment/monitoring
docker compose up -d
```

Services deployed:
- Heart Disease API (port 8000)
- Prometheus (port 9090)
- Grafana (port 3000)

![Grafana Dashboard](screenshots/grafana_report.png)

### 7.4 Alerting Considerations

The monitoring setup enables future alerting on:
- High error rates (>1%)
- Elevated latency (p99 > 500ms)
- Pod availability issues
- Resource utilization thresholds

---

## 8. Conclusion and Future Work

### 8.1 Achievements

This project successfully implemented a complete MLOps pipeline:

1. **Data Pipeline**: Automated data acquisition and preprocessing
2. **Model Training**: Reproducible training with experiment tracking
3. **API Development**: Production-ready FastAPI application
4. **CI/CD**: Automated testing and deployment pipeline
5. **Containerization**: Optimized Docker images
6. **Orchestration**: Kubernetes deployment configurations
7. **Monitoring**: Prometheus metrics and Grafana dashboards

### 8.2 Key Learnings

1. **MLOps Practices**: The importance of reproducibility, versioning, and automation
2. **Model Selection**: Simple models can outperform complex ones with proper preprocessing
3. **Production Considerations**: Security, scalability, and observability are crucial
4. **CI/CD Benefits**: Automated pipelines catch issues early and ensure consistency

### 8.3 Future Improvements

1. **Model Enhancements**:
   - Hyperparameter optimization with GridSearchCV/Optuna
   - Ensemble methods combining multiple algorithms
   - Feature engineering (interaction terms, polynomial features)

2. **Infrastructure**:
   - Kubernetes autoscaling based on load
   - A/B testing framework for model comparison
   - Model serving with MLflow or Seldon

3. **Monitoring**:
   - Data drift detection
   - Model performance monitoring in production
   - Automated retraining triggers

4. **Security**:
   - API authentication and authorization
   - Input sanitization and rate limiting
   - Audit logging

### 8.4 Final Remarks

This project demonstrates that building production-ready ML systems requires much more than just training a model. The MLOps approach ensures that models can be reliably deployed, monitored, and maintained in production environments. The practices implemented here—version control, automated testing, containerization, and monitoring—are essential for any organization deploying machine learning at scale.

---

## References

1. UCI Machine Learning Repository - Heart Disease Dataset
2. scikit-learn Documentation
3. FastAPI Documentation
4. MLflow Documentation
5. Kubernetes Documentation
6. Prometheus and Grafana Documentation

---

**Appendix A: Project Structure**

```
mlops-assignment/
|-- .github/workflows/ci-cd.yml
|-- api/main.py
|-- data/raw/heart.csv
|-- deployment/
|   |-- Dockerfile
|   |-- k8s/
|   +-- monitoring/
|-- docs/architecture.md
|-- models/
|-- mlruns/
|-- notebooks/01_eda_and_modeling.ipynb
|-- scripts/deploy.sh
|-- src/
|   |-- config.py
|   |-- data_loader.py
|   |-- preprocessing.py
|   |-- train.py
|   +-- predict.py
|-- tests/
|-- requirements.txt
+-- README.md
```

**Appendix B: API Usage Example**

```bash
# Health check
curl http://localhost:8000/health

# Make prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63, "sex": 1, "cp": 4, "trestbps": 145,
    "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150,
    "exang": 0, "oldpeak": 2.3, "slope": 2, "ca": 0, "thal": 6
  }'

# Response
{
  "prediction": 1,
  "probability": 0.61,
  "risk_level": "High"
}
```


