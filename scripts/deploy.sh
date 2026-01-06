#!/bin/bash
# Deployment script for Heart Disease Prediction API
# This script builds and deploys the API to Kubernetes

set -e

echo "============================================"
echo "Heart Disease API - Deployment Script"
echo "============================================"

# Configuration
IMAGE_NAME="heart-disease-api"
IMAGE_TAG="latest"
NAMESPACE="default"

# Check for required tools
echo ""
echo "Checking required tools..."
command -v docker >/dev/null 2>&1 || { echo "Docker is required but not installed. Aborting." >&2; exit 1; }
command -v kubectl >/dev/null 2>&1 || { echo "kubectl is required but not installed. Aborting." >&2; exit 1; }

# Step 1: Build Docker image
echo ""
echo "Step 1: Building Docker image..."
docker build -f deployment/Dockerfile -t ${IMAGE_NAME}:${IMAGE_TAG} .
echo "✓ Docker image built: ${IMAGE_NAME}:${IMAGE_TAG}"

# Step 2: Check for Kubernetes cluster
echo ""
echo "Step 2: Checking Kubernetes cluster..."
if kubectl cluster-info >/dev/null 2>&1; then
    echo "✓ Kubernetes cluster is available"
else
    echo "No Kubernetes cluster detected."
    echo ""
    echo "Options:"
    echo "  1. Start Minikube: minikube start"
    echo "  2. Enable Docker Desktop Kubernetes"
    echo "  3. Use a cloud Kubernetes cluster"
    echo ""
    echo "After starting a cluster, re-run this script."
    exit 1
fi

# Step 3: Load image to cluster (for Minikube)
echo ""
echo "Step 3: Loading image to cluster..."
if command -v minikube >/dev/null 2>&1 && minikube status >/dev/null 2>&1; then
    echo "Loading image to Minikube..."
    minikube image load ${IMAGE_NAME}:${IMAGE_TAG}
    echo "✓ Image loaded to Minikube"
else
    echo "Using local Docker daemon (Docker Desktop K8s or similar)"
fi

# Step 4: Deploy to Kubernetes
echo ""
echo "Step 4: Deploying to Kubernetes..."
kubectl apply -f deployment/k8s/deployment.yaml
kubectl apply -f deployment/k8s/service.yaml
echo "✓ Kubernetes manifests applied"

# Step 5: Wait for deployment
echo ""
echo "Step 5: Waiting for deployment to be ready..."
kubectl rollout status deployment/heart-disease-api --timeout=120s
echo "✓ Deployment is ready"

# Step 6: Get service information
echo ""
echo "Step 6: Getting service information..."
echo ""
kubectl get pods -l app=heart-disease-api
echo ""
kubectl get services heart-disease-api-service
echo ""

# Get service URL
if command -v minikube >/dev/null 2>&1 && minikube status >/dev/null 2>&1; then
    SERVICE_URL=$(minikube service heart-disease-api-service --url 2>/dev/null)
    echo "Service URL: ${SERVICE_URL}"
else
    echo "Service URL: http://localhost:80 (if using LoadBalancer on Docker Desktop)"
fi

echo ""
echo "============================================"
echo "Deployment Complete!"
echo "============================================"
echo ""
echo "Test the API with:"
echo "  curl \${SERVICE_URL}/health"
echo "  curl -X POST \${SERVICE_URL}/predict -H 'Content-Type: application/json' -d '{\"age\":63,\"sex\":1,\"cp\":4,\"trestbps\":145,\"chol\":233,\"fbs\":1,\"restecg\":0,\"thalach\":150,\"exang\":0,\"oldpeak\":2.3,\"slope\":2,\"ca\":0,\"thal\":6}'"


