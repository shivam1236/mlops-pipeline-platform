# MLOps Pipeline Platform 🚀

An end-to-end MLOps pipeline for training, versioning, and deploying ML models at scale using Kubeflow, MLflow, and Kubernetes.

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![Kubeflow](https://img.shields.io/badge/Kubeflow-1.8-326CE5?style=flat-square&logo=kubernetes&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-2.x-0194E2?style=flat-square&logo=mlflow&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)
![Kubernetes](https://img.shields.io/badge/Kubernetes-326CE5?style=flat-square&logo=kubernetes&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

</div>

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        MLOps Pipeline Platform                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────────────┐    │
│  │  Data    │──▶│  Feature │──▶│  Model   │──▶│  Model Registry  │    │
│  │  Ingest  │   │  Store   │   │ Training │   │    (MLflow)      │    │
│  └──────────┘   └──────────┘   └──────────┘   └────────┬─────────┘    │
│                                                          │              │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐            ▼              │
│  │ Monitor  │◀──│  A/B     │◀──│  Model   │◀── Auto-Deploy            │
│  │ & Alert  │   │  Testing │   │  Serving │    (if metrics pass)      │
│  └──────────┘   └──────────┘   └──────────┘                           │
│                                                                         │
│  Orchestration: Kubeflow Pipelines + Argo Workflows                     │
│  Infrastructure: Kubernetes + Terraform                                 │
│  CI/CD: GitHub Actions + ArgoCD                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

- **Automated Training Pipelines** — Kubeflow pipelines for reproducible model training
- **Experiment Tracking** — MLflow for metrics, parameters, and artifact logging
- **Model Versioning** — DVC for data versioning + MLflow Model Registry
- **Auto-Retraining** — Drift detection triggers automatic retraining
- **Canary Deployments** — Gradual rollout with automatic rollback
- **Model Monitoring** — Prometheus + Grafana dashboards for inference metrics
- **Feature Store** — Centralized feature computation and serving
- **GitOps Deployment** — ArgoCD syncs model deployments from Git

---

## 📁 Project Structure

```
mlops-pipeline-platform/
├── pipelines/                  # Kubeflow pipeline definitions
│   ├── training_pipeline.py
│   ├── inference_pipeline.py
│   └── retraining_pipeline.py
├── src/
│   ├── data/                   # Data ingestion & preprocessing
│   │   ├── ingest.py
│   │   └── preprocess.py
│   ├── features/               # Feature engineering
│   │   ├── feature_store.py
│   │   └── transformers.py
│   ├── models/                 # Model training & evaluation
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── registry.py
│   ├── serving/                # Model serving
│   │   ├── predictor.py
│   │   └── api.py
│   └── monitoring/             # Drift detection & alerting
│       ├── drift_detector.py
│       └── metrics.py
├── k8s/                        # Kubernetes manifests
│   ├── base/
│   ├── overlays/
│   └── serving/
├── terraform/                  # Infrastructure as Code
│   ├── modules/
│   ├── environments/
│   └── main.tf
├── docker/                     # Dockerfiles
│   ├── training.Dockerfile
│   ├── serving.Dockerfile
│   └── monitoring.Dockerfile
├── tests/                      # Unit & integration tests
├── .github/workflows/          # CI/CD pipelines
│   ├── ci.yml
│   ├── train.yml
│   └── deploy.yml
├── configs/                    # Pipeline & model configs
│   ├── pipeline_config.yaml
│   └── model_config.yaml
├── Makefile
├── pyproject.toml
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- kubectl configured
- Helm 3.x

### Local Development

```bash
# Clone the repo
git clone https://github.com/shivam1236/mlops-pipeline-platform.git
cd mlops-pipeline-platform

# Setup virtual environment
make setup

# Run tests
make test

# Start local MLflow server
make mlflow-up

# Run training pipeline locally
make train MODEL=xgboost CONFIG=configs/model_config.yaml
```

### Deploy to Kubernetes

```bash
# Apply infrastructure
make infra-apply ENV=dev

# Deploy pipeline components
make deploy ENV=dev

# Trigger training pipeline
make pipeline-run PIPELINE=training
```

---

## 🔄 Pipeline Stages

### 1. Data Ingestion
```python
@pipeline_step(name="data-ingestion")
def ingest_data(source_config: dict) -> pd.DataFrame:
    """Pull data from multiple sources with validation."""
    ...
```

### 2. Feature Engineering
```python
@pipeline_step(name="feature-engineering")
def compute_features(df: pd.DataFrame, feature_config: dict) -> pd.DataFrame:
    """Compute and store features in feature store."""
    ...
```

### 3. Model Training
```python
@pipeline_step(name="model-training")
def train_model(features: pd.DataFrame, config: ModelConfig) -> MLModel:
    """Train model with hyperparameter tuning and MLflow tracking."""
    with mlflow.start_run():
        model = train(features, config)
        mlflow.log_metrics(evaluate(model))
        mlflow.sklearn.log_model(model, "model")
    return model
```

### 4. Model Deployment
```python
@pipeline_step(name="model-deployment")
def deploy_model(model_uri: str, strategy: str = "canary") -> None:
    """Deploy model with canary strategy and auto-rollback."""
    ...
```

---

## 📊 Monitoring Dashboard

The platform includes Grafana dashboards for:

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| Prediction Latency (p99) | End-to-end inference time | > 200ms |
| Model Drift Score | PSI/KL divergence | > 0.2 |
| Prediction Volume | Requests per second | < 10 rps |
| Error Rate | Failed predictions | > 1% |
| Feature Drift | Input distribution shift | > 0.15 |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Pipeline Orchestration | Kubeflow Pipelines |
| Experiment Tracking | MLflow |
| Data Versioning | DVC |
| Feature Store | Feast |
| Model Serving | KServe / Seldon Core |
| Infrastructure | Terraform + Kubernetes |
| CI/CD | GitHub Actions + ArgoCD |
| Monitoring | Prometheus + Grafana |
| Alerting | PagerDuty |

---

## 📈 Results

| Model | Accuracy | F1 Score | Latency (p95) | Status |
|-------|----------|----------|---------------|--------|
| XGBoost v3.2 | 94.2% | 0.93 | 45ms | ✅ Production |
| LightGBM v2.1 | 93.8% | 0.92 | 38ms | 🔄 Canary (20%) |
| Neural Net v1.0 | 95.1% | 0.94 | 120ms | 🧪 Staging |

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with ❤️ by [Shivam Shukla](https://github.com/shivam1236)**

If you found this useful, give it a ⭐!

</div>
