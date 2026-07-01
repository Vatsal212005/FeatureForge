from pathlib import Path

ROOT = Path.cwd()

README = r"""# FeatureForge

**FeatureForge** is a lightweight ML Feature Store and MLOps platform for managing datasets, reusable feature definitions, offline training tables, online feature serving, model training, prediction APIs, and feature drift monitoring.

This project is designed as an industry-style ML infrastructure system rather than a standalone notebook or single-model experiment.

---

## Overview

FeatureForge demonstrates a complete ML platform workflow:

```text
Raw Dataset
    ↓
Dataset Registry
    ↓
Feature Registry
    ↓
Offline Feature Materialization
    ↓
Model Training + Model Registry
    ↓
Prediction API
    ↓
Online Feature Store
    ↓
Drift Monitoring
```

Modern ML teams often face these problems:

- Feature logic is duplicated across notebooks, training scripts, and production services.
- Training and inference pipelines use slightly different transformations.
- Datasets, features, materializations, and models are not centrally tracked.
- Predictions are hard to trace back to the exact feature versions used during training.
- Production feature distributions drift away from training distributions.

FeatureForge solves these through a small but complete MLOps workflow.

---

## Features

### Dataset Registry

- Upload CSV datasets.
- Infer column schema automatically.
- Track dataset name, version, row count, column count, file hash, and storage path.
- Preview registered datasets through API and dashboard.

### Feature Registry

- Register reusable feature definitions.
- Supports column features, aggregate features, transformations, feature versioning, and schema validation.
- Prevents invalid features by checking source columns, entity columns, and aggregation settings.

### Offline Feature Materialization

- Converts registered feature definitions into training-ready feature tables.
- Stores materialized feature tables as CSV files.
- Tracks feature IDs, feature names, label column, rows, columns, and storage location.

### Model Training and Registry

- Trains models using materialized feature tables.
- Supports Random Forest, Logistic Regression, and XGBoost.
- Stores model artifacts using Joblib.
- Tracks feature columns, metrics, algorithm, label column, train/test split, and materialization lineage.

### Prediction API

- Loads trained model artifacts.
- Accepts JSON feature records.
- Returns predictions and class probabilities when available.
- Includes model input schema endpoint.

### Online Feature Store

- Pushes materialized feature vectors into an online serving table.
- Supports lookup by entity ID.
- Supports batch feature lookup.
- Supports prediction directly from online feature vectors.

### Drift Monitoring

- Compares reference and current materialized feature tables.
- Computes numeric drift using PSI and normalized mean shift.
- Computes categorical drift using total variation distance.
- Generates drift reports with low, medium, or high drift levels.

### React Dashboard

- Dataset upload
- Dataset registry view
- Feature creation
- Feature registry
- Offline materialization
- Model registry
- Online store operation
- Drift report generation
- Project statistics

### Dockerized Deployment

- Backend container
- Frontend container
- Docker Compose setup
- Demo seed pipeline

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI |
| Frontend | React, Vite |
| Database | SQLite |
| ML | scikit-learn, XGBoost |
| Data Processing | pandas, NumPy |
| Model Artifacts | Joblib |
| API Validation | Pydantic |
| Containerization | Docker, Docker Compose |
| Dashboard UI | React + CSS |
| Feature Serving | SQLite-backed online feature table |

---

## Project Structure

```text
FeatureForge/
├── backend/
│   ├── app/
│   │   ├── core/
│   │   ├── db/
│   │   ├── models/
│   │   ├── routers/
│   │   ├── schemas/
│   │   ├── services/
│   │   └── main.py
│   ├── tests/
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── main.jsx
│   │   └── styles.css
│   ├── Dockerfile
│   └── package.json
├── data/
│   ├── raw/
│   └── processed/
├── artifacts/
│   ├── models/
│   └── reports/
├── scripts/
│   └── seed_demo.py
├── docs/
├── screenshots/
├── docker-compose.yml
└── README.md
```

---

## Architecture

```mermaid
flowchart TD
    A[CSV Dataset Upload] --> B[Dataset Registry]
    B --> C[Feature Registry]
    C --> D[Offline Feature Materialization]
    D --> E[Training Feature Table]
    E --> F[Model Training]
    F --> G[Model Registry]
    G --> H[Prediction API]

    D --> I[Online Feature Store]
    I --> J[Entity Feature Lookup]
    I --> K[Online Prediction]

    D --> L[Drift Monitoring]
    L --> M[Drift Report]

    N[React Dashboard] --> B
    N --> C
    N --> D
    N --> G
    N --> I
    N --> L
```

---

## Local Run

### Backend

```bash
cd backend
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run API:

```bash
uvicorn app.main:app --reload
```

API docs:

```text
http://127.0.0.1:8000/docs
```

### Frontend

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Dashboard:

```text
http://localhost:5173
```

---

## Docker Run

Make sure Docker Desktop is running.

```bash
docker compose up --build
```

Dashboard:

```text
http://localhost:5173
```

API docs:

```text
http://127.0.0.1:8000/docs
```

If Docker is installed but not recognized in PowerShell, use:

```powershell
& "C:\Program Files\Docker\Docker\resources\bin\docker.exe" compose up --build
```

---

## Seed Demo Data

After the backend is running:

```bash
python scripts/seed_demo.py
```

The demo creates:

- reference transaction dataset
- current transaction dataset
- feature definitions
- offline materializations
- trained fraud model
- online feature store entries
- drift report

---

## Demo Workflow

### Upload Dataset

```text
POST /api/datasets/upload
```

### Create Feature

```text
POST /api/features
```

Example:

```json
{
  "name": "amount_log",
  "dataset_id": 1,
  "description": "Log-transformed transaction amount.",
  "entity_column": "user_id",
  "source_column": "amount",
  "feature_kind": "column",
  "transformation": "log1p",
  "output_dtype": "float"
}
```

### Materialize Features

```text
POST /api/materializations
```

### Train Model

```text
POST /api/models/train
```

Example:

```json
{
  "materialization_id": 1,
  "name": "fraud_detection_rf",
  "label_column": "is_fraud",
  "algorithm": "random_forest",
  "problem_type": "classification",
  "test_size": 0.3,
  "random_state": 42
}
```

### Push Features to Online Store

```text
POST /api/online-store/materialize
```

### Run Online Prediction

```text
POST /api/online-store/models/{model_id}/predict
```

Example:

```json
{
  "materialization_id": 1,
  "entity_column": "user_id",
  "entity_values": [1, 11, 15]
}
```

### Generate Drift Report

```text
POST /api/drift/reports
```

Example:

```json
{
  "reference_materialization_id": 1,
  "current_materialization_id": 2,
  "name": "reference_vs_current_drift",
  "feature_columns": null
}
```

---

## API Modules

| Module | Purpose |
|---|---|
| `/api/health` | API health check |
| `/api/datasets` | Dataset registry |
| `/api/features` | Feature registry |
| `/api/materializations` | Offline feature materialization |
| `/api/models` | Model training and registry |
| `/api/predictions` | Direct prediction API |
| `/api/online-store` | Online feature serving |
| `/api/drift` | Drift monitoring |

---

## Screenshots

Add screenshots to the `screenshots/` folder.

Recommended files:

```text
screenshots/dashboard-home.png
screenshots/dataset-registry.png
screenshots/feature-registry.png
screenshots/model-registry.png
screenshots/drift-report.png
screenshots/swagger-prediction.png
```

Then update this section with image links:

```md
![Dashboard](screenshots/dashboard-home.png)
![Drift Report](screenshots/drift-report.png)
```

---

## Testing

Run backend tests:

```bash
cd backend
pytest
```

---

## Resume Bullets

```text
FeatureForge | ML Feature Store & MLOps Platform | FastAPI, React, Docker, SQLite, scikit-learn
• Built an end-to-end ML feature store with dataset registration, reusable feature definitions, offline materialization, model training, online feature serving, and drift monitoring.
• Implemented a full ML lifecycle workflow with feature versioning, model registry, prediction APIs, Dockerized deployment, and a React dashboard for operational visibility.
```

Alternative shorter version:

```text
Built FeatureForge, a Dockerized ML feature store and MLOps platform with dataset registry, feature registry, offline/online feature serving, model training, prediction APIs, drift monitoring, and React dashboard.
```

---

## Future Improvements

- Replace SQLite online store with Redis.
- Replace SQLite metadata database with PostgreSQL.
- Add scheduled feature materialization jobs.
- Add authentication and workspace-level access control.
- Add feature lineage graph visualization.
- Add model approval and rollback workflow.
- Add SHAP-based model explainability.
- Add CI/CD with GitHub Actions.
- Add cloud deployment on AWS/GCP/Azure.

---

## Status

FeatureForge is a working MVP demonstrating:

```text
Dataset → Feature Registry → Materialization → Model Training → Online Serving → Drift Monitoring
```
"""

API_REFERENCE = r"""# FeatureForge API Reference

## Health

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Check API health |

## Datasets

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/datasets/upload` | Upload and register CSV dataset |
| GET | `/api/datasets` | List datasets |
| GET | `/api/datasets/{dataset_id}` | Get dataset details |
| GET | `/api/datasets/{dataset_id}/preview` | Preview dataset rows |

## Features

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/features` | Create feature definition |
| POST | `/api/features/validate` | Validate feature definition |
| GET | `/api/features` | List feature definitions |
| GET | `/api/features/{feature_id}` | Get feature definition |
| GET | `/api/features/{feature_id}/preview` | Preview computed feature |

## Materializations

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/materializations` | Materialize features into offline table |
| GET | `/api/materializations` | List materializations |
| GET | `/api/materializations/{materialization_id}` | Get materialization metadata |
| GET | `/api/materializations/{materialization_id}/preview` | Preview materialized table |

## Models

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/models/train` | Train model from materialization |
| GET | `/api/models` | List trained models |
| GET | `/api/models/{model_id}` | Get model metadata |
| GET | `/api/models/{model_id}/metrics` | Get model metrics |

## Predictions

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/predictions/models/{model_id}` | Predict from JSON records |
| POST | `/api/predictions/models/{model_id}/batch` | Predict from materialized table |
| GET | `/api/predictions/models/{model_id}/input-schema` | Get expected feature columns |

## Online Store

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/online-store/materialize` | Push materialization to online store |
| GET | `/api/online-store/{materialization_id}/features/{entity_value}` | Lookup online feature vector |
| POST | `/api/online-store/{materialization_id}/batch-lookup` | Batch online feature lookup |
| GET | `/api/online-store/{materialization_id}/stats` | Online store stats |
| POST | `/api/online-store/models/{model_id}/predict` | Predict using online features |

## Drift Monitoring

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/drift/reports` | Generate drift report |
| GET | `/api/drift/reports` | List drift reports |
| GET | `/api/drift/reports/{report_id}` | Get drift report |
"""

ARCHITECTURE = r"""# FeatureForge Architecture

## System Goal

FeatureForge simulates the core infrastructure used in production ML systems:

- central dataset registry
- reusable feature definitions
- offline training data generation
- model training and registry
- online feature serving
- prediction APIs
- drift monitoring

---

## High-Level Architecture

```mermaid
flowchart LR
    CSV[CSV Upload] --> DatasetRegistry[Dataset Registry]
    DatasetRegistry --> FeatureRegistry[Feature Registry]
    FeatureRegistry --> OfflineMaterializer[Offline Materializer]
    OfflineMaterializer --> TrainingTable[Training Table]
    TrainingTable --> Trainer[Model Trainer]
    Trainer --> ModelRegistry[Model Registry]
    ModelRegistry --> PredictionAPI[Prediction API]

    OfflineMaterializer --> OnlineStore[Online Feature Store]
    OnlineStore --> OnlinePrediction[Online Prediction]

    OfflineMaterializer --> DriftService[Drift Monitoring]
    DriftService --> DriftReports[Drift Reports]

    Dashboard[React Dashboard] --> DatasetRegistry
    Dashboard --> FeatureRegistry
    Dashboard --> OfflineMaterializer
    Dashboard --> ModelRegistry
    Dashboard --> OnlineStore
    Dashboard --> DriftService
```

---

## Backend Design

| Folder | Purpose |
|---|---|
| `models/` | SQLAlchemy database models |
| `schemas/` | Pydantic request/response schemas |
| `services/` | Business logic |
| `routers/` | FastAPI routes |
| `db/` | Database engine/session |
| `core/` | Configuration |

---

## Storage Design

| Storage | Used For |
|---|---|
| SQLite DB | metadata registry |
| `data/raw/` | uploaded datasets |
| `data/processed/` | materialized feature tables |
| `artifacts/models/` | trained model artifacts |
| Online feature table | serving entity feature vectors |

---

## Model Lifecycle

```text
Materialized Feature Table
    ↓
Feature/label split
    ↓
Train/test split
    ↓
Model training
    ↓
Metric calculation
    ↓
Joblib artifact save
    ↓
Model registry entry
```

---

## Drift Monitoring

Numeric drift:

- Population Stability Index
- normalized mean shift

Categorical drift:

- total variation distance
- new categories
- missing categories
- top-value distribution shift

Overall drift:

```text
mean(column drift scores)
```

Drift levels:

| Score | Level |
|---:|---|
| `< 0.10` | low |
| `0.10 – 0.25` | medium |
| `>= 0.25` | high |
"""

RESUME_BULLETS = r"""# Resume Bullets

## Main Version

```text
FeatureForge | ML Feature Store & MLOps Platform | FastAPI, React, Docker, SQLite, scikit-learn
• Built an end-to-end ML feature store with dataset registration, reusable feature definitions, offline materialization, model training, online feature serving, and drift monitoring.
• Implemented a full ML lifecycle workflow with feature versioning, model registry, prediction APIs, Dockerized deployment, and a React dashboard for operational visibility.
```

## Short Version

```text
FeatureForge | ML Feature Store & MLOps Platform | FastAPI, React, Docker, scikit-learn
• Built a Dockerized ML feature store with dataset registry, feature registry, offline/online feature serving, model training, prediction APIs, drift monitoring, and React dashboard.
```

## Interview Explanation

FeatureForge is a lightweight version of the kind of feature-store and MLOps infrastructure used by ML platform teams. It solves training-serving skew by registering reusable feature definitions, materializing offline feature tables for training, serving entity-level feature vectors for online inference, and monitoring feature drift between reference and current datasets.

## Strong One-Liner

```text
FeatureForge turns raw CSV datasets into reusable feature definitions, training-ready materializations, trained model artifacts, online feature vectors, and drift reports through a full-stack FastAPI + React platform.
```
"""

GITIGNORE_APPEND = r"""
# Screenshots
!screenshots/.gitkeep

# Keep generated folders but ignore generated demo data/artifacts
sample_data/
*.db
data/raw/*.csv
data/processed/*.csv
artifacts/models/*.joblib
artifacts/reports/*
"""

files = {
    "README.md": README,
    "docs/API_REFERENCE.md": API_REFERENCE,
    "docs/ARCHITECTURE.md": ARCHITECTURE,
    "docs/RESUME_BULLETS.md": RESUME_BULLETS,
    "screenshots/.gitkeep": "",
}


def write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")
    print(f"wrote {path}")


def append_gitignore() -> None:
    gitignore_path = ROOT / ".gitignore"

    if not gitignore_path.exists():
        write_file(gitignore_path, GITIGNORE_APPEND)
        return

    existing = gitignore_path.read_text(encoding="utf-8")

    if "# Keep generated folders but ignore generated demo data/artifacts" not in existing:
        gitignore_path.write_text(existing.rstrip() + "\n" + GITIGNORE_APPEND, encoding="utf-8")
        print(f"updated {gitignore_path}")
    else:
        print(".gitignore already contains generated-data rules")


def main() -> None:
    print("Generating final FeatureForge documentation...")

    for relative_path, content in files.items():
        write_file(ROOT / relative_path, content)

    append_gitignore()

    print()
    print("Documentation generated successfully.")
    print()
    print("Created/updated:")
    print("- README.md")
    print("- docs/API_REFERENCE.md")
    print("- docs/ARCHITECTURE.md")
    print("- docs/RESUME_BULLETS.md")
    print("- screenshots/.gitkeep")


if __name__ == "__main__":
    main()
